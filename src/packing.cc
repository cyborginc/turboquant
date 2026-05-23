#include "packing.h"

#include <cassert>
#include <cstring>

namespace turboquant {
namespace {

void PackBitstream(const uint16_t* codes, size_t n, int bits, uint8_t* out) {
  const size_t total_bits = n * static_cast<size_t>(bits);
  const size_t total_bytes = (total_bits + 7) / 8;
  std::memset(out, 0, total_bytes);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t value = codes[i];
    const size_t bit_off = i * static_cast<size_t>(bits);
    const size_t byte_off = bit_off >> 3;
    const int shift = static_cast<int>(bit_off & 7);
    // A code spans at most 12 bits, so it touches at most 3 bytes when shifted.
    uint32_t shifted = value << shift;
    out[byte_off] |= static_cast<uint8_t>(shifted & 0xFFu);
    if (shift + bits > 8) {
      out[byte_off + 1] |= static_cast<uint8_t>((shifted >> 8) & 0xFFu);
      if (shift + bits > 16) {
        out[byte_off + 2] |= static_cast<uint8_t>((shifted >> 16) & 0xFFu);
      }
    }
  }
}

}  // namespace

void PackCodes(const uint16_t* codes, size_t n, QuantBits bits, uint8_t* out) {
  switch (bits) {
    case QuantBits::B8: {
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(codes[i]);
      return;
    }
    case QuantBits::B4: {
      const size_t pairs = n / 2;
      for (size_t i = 0; i < pairs; ++i) {
        const uint8_t lo = static_cast<uint8_t>(codes[2 * i] & 0xF);
        const uint8_t hi = static_cast<uint8_t>(codes[2 * i + 1] & 0xF);
        out[i] = static_cast<uint8_t>(lo | (hi << 4));
      }
      if (n & 1) {
        out[pairs] = static_cast<uint8_t>(codes[n - 1] & 0xF);
      }
      return;
    }
    case QuantBits::B2: {
      const size_t groups = n / 4;
      for (size_t i = 0; i < groups; ++i) {
        const uint8_t b = static_cast<uint8_t>(
            (codes[4 * i] & 0x3) | ((codes[4 * i + 1] & 0x3) << 2) |
            ((codes[4 * i + 2] & 0x3) << 4) | ((codes[4 * i + 3] & 0x3) << 6));
        out[i] = b;
      }
      const size_t rem = n & 3;
      if (rem) {
        uint8_t b = 0;
        for (size_t k = 0; k < rem; ++k) {
          b |= static_cast<uint8_t>((codes[4 * groups + k] & 0x3) << (2 * k));
        }
        out[groups] = b;
      }
      return;
    }
    case QuantBits::B1: {
      const size_t groups = n / 8;
      for (size_t i = 0; i < groups; ++i) {
        uint8_t b = 0;
        for (int k = 0; k < 8; ++k) {
          b |= static_cast<uint8_t>((codes[8 * i + k] & 0x1) << k);
        }
        out[i] = b;
      }
      const size_t rem = n & 7;
      if (rem) {
        uint8_t b = 0;
        for (size_t k = 0; k < rem; ++k) {
          b |= static_cast<uint8_t>((codes[8 * groups + k] & 0x1) << k);
        }
        out[groups] = b;
      }
      return;
    }
    case QuantBits::B6:
    case QuantBits::B12:
      PackBitstream(codes, n, static_cast<int>(bits), out);
      return;
  }
}

namespace {

// 8 codes occupy 48 bits = 6 bytes. The bulk loop loads 8 bytes per iter
// (reading 2 bytes more than needed); the scalar tail handles the last group
// when an 8-byte load would walk past the packed buffer.
void UnpackB6(const uint8_t* in, size_t n, uint16_t* codes) {
  const size_t tot_bytes = (n * 6 + 7) / 8;
  size_t i = 0;
  while (i + 8 <= n) {
    const size_t byte_off = 6 * (i / 8);
    if (byte_off + 8 > tot_bytes) break;
    uint64_t v;
    std::memcpy(&v, in + byte_off, 8);
    codes[i + 0] = static_cast<uint16_t>((v >> 0) & 0x3F);
    codes[i + 1] = static_cast<uint16_t>((v >> 6) & 0x3F);
    codes[i + 2] = static_cast<uint16_t>((v >> 12) & 0x3F);
    codes[i + 3] = static_cast<uint16_t>((v >> 18) & 0x3F);
    codes[i + 4] = static_cast<uint16_t>((v >> 24) & 0x3F);
    codes[i + 5] = static_cast<uint16_t>((v >> 30) & 0x3F);
    codes[i + 6] = static_cast<uint16_t>((v >> 36) & 0x3F);
    codes[i + 7] = static_cast<uint16_t>((v >> 42) & 0x3F);
    i += 8;
  }
  for (; i < n; ++i) {
    const size_t bit_off = i * 6;
    const size_t byte_off = bit_off >> 3;
    const int shift = static_cast<int>(bit_off & 7);
    uint32_t v32 = in[byte_off];
    if (shift + 6 > 8) v32 |= static_cast<uint32_t>(in[byte_off + 1]) << 8;
    codes[i] = static_cast<uint16_t>((v32 >> shift) & 0x3F);
  }
}

// 4 codes occupy 48 bits = 6 bytes. Same idea: u64 load gives 4 codes.
void UnpackB12(const uint8_t* in, size_t n, uint16_t* codes) {
  const size_t tot_bytes = (n * 12 + 7) / 8;
  size_t i = 0;
  while (i + 4 <= n) {
    const size_t byte_off = 6 * (i / 4);
    if (byte_off + 8 > tot_bytes) break;
    uint64_t v;
    std::memcpy(&v, in + byte_off, 8);
    codes[i + 0] = static_cast<uint16_t>((v >> 0) & 0xFFF);
    codes[i + 1] = static_cast<uint16_t>((v >> 12) & 0xFFF);
    codes[i + 2] = static_cast<uint16_t>((v >> 24) & 0xFFF);
    codes[i + 3] = static_cast<uint16_t>((v >> 36) & 0xFFF);
    i += 4;
  }
  for (; i < n; ++i) {
    const size_t bit_off = i * 12;
    const size_t byte_off = bit_off >> 3;
    const int shift = static_cast<int>(bit_off & 7);
    uint32_t v32 = static_cast<uint32_t>(in[byte_off]) |
                   (static_cast<uint32_t>(in[byte_off + 1]) << 8);
    if (shift + 12 > 16)
      v32 |= static_cast<uint32_t>(in[byte_off + 2]) << 16;
    codes[i] = static_cast<uint16_t>((v32 >> shift) & 0xFFF);
  }
}

}  // namespace

void UnpackCodes(const uint8_t* in, size_t n, QuantBits bits, uint16_t* codes) {
  switch (bits) {
    case QuantBits::B8: {
      for (size_t i = 0; i < n; ++i) codes[i] = in[i];
      return;
    }
    case QuantBits::B4: {
      const size_t pairs = n / 2;
      for (size_t i = 0; i < pairs; ++i) {
        const uint8_t b = in[i];
        codes[2 * i] = static_cast<uint16_t>(b & 0xF);
        codes[2 * i + 1] = static_cast<uint16_t>((b >> 4) & 0xF);
      }
      if (n & 1) codes[n - 1] = static_cast<uint16_t>(in[pairs] & 0xF);
      return;
    }
    case QuantBits::B2: {
      const size_t groups = n / 4;
      for (size_t i = 0; i < groups; ++i) {
        const uint8_t b = in[i];
        codes[4 * i] = static_cast<uint16_t>(b & 0x3);
        codes[4 * i + 1] = static_cast<uint16_t>((b >> 2) & 0x3);
        codes[4 * i + 2] = static_cast<uint16_t>((b >> 4) & 0x3);
        codes[4 * i + 3] = static_cast<uint16_t>((b >> 6) & 0x3);
      }
      const size_t rem = n & 3;
      if (rem) {
        const uint8_t b = in[groups];
        for (size_t k = 0; k < rem; ++k) {
          codes[4 * groups + k] = static_cast<uint16_t>((b >> (2 * k)) & 0x3);
        }
      }
      return;
    }
    case QuantBits::B1: {
      const size_t groups = n / 8;
      for (size_t i = 0; i < groups; ++i) {
        const uint8_t b = in[i];
        for (int k = 0; k < 8; ++k) {
          codes[8 * i + k] = static_cast<uint16_t>((b >> k) & 0x1);
        }
      }
      const size_t rem = n & 7;
      if (rem) {
        const uint8_t b = in[groups];
        for (size_t k = 0; k < rem; ++k) {
          codes[8 * groups + k] = static_cast<uint16_t>((b >> k) & 0x1);
        }
      }
      return;
    }
    case QuantBits::B6:
      UnpackB6(in, n, codes);
      return;
    case QuantBits::B12:
      UnpackB12(in, n, codes);
      return;
  }
}

}  // namespace turboquant
