#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "packing.h"
#include "turboquant/turboquant.h"

using turboquant::PackCodes;
using turboquant::QuantBits;
using turboquant::UnpackCodes;

namespace {

void RoundtripBits(QuantBits bits, size_t n) {
  const int b = static_cast<int>(bits);
  const uint32_t mask = (1u << b) - 1u;
  std::mt19937_64 rng(0xC0FFEEULL + b * 1000 + n);
  std::vector<uint16_t> codes(n);
  for (size_t i = 0; i < n; ++i) {
    codes[i] = static_cast<uint16_t>(rng() & mask);
  }
  const size_t bytes = (n * b + 7) / 8;
  std::vector<uint8_t> packed(bytes, 0xAA);  // garbage init
  PackCodes(codes.data(), n, bits, packed.data());

  std::vector<uint16_t> decoded(n, 0xFFFF);
  UnpackCodes(packed.data(), n, bits, decoded.data());
  ASSERT_EQ(codes, decoded) << "bits=" << b << " n=" << n;
}

}  // namespace

TEST(Packing, Roundtrip_1bit) {
  for (size_t n : {1ul, 7ul, 8ul, 9ul, 31ul, 64ul, 65ul, 1023ul}) {
    RoundtripBits(QuantBits::B1, n);
  }
}
TEST(Packing, Roundtrip_3bit) {
  for (size_t n : {1ul, 7ul, 8ul, 9ul, 15ul, 16ul, 17ul, 64ul, 257ul}) {
    RoundtripBits(QuantBits::B3, n);
  }
}
TEST(Packing, Roundtrip_2bit) {
  for (size_t n : {1ul, 3ul, 4ul, 5ul, 16ul, 17ul, 31ul, 1023ul}) {
    RoundtripBits(QuantBits::B2, n);
  }
}
TEST(Packing, Roundtrip_4bit) {
  for (size_t n : {1ul, 2ul, 3ul, 15ul, 16ul, 17ul, 256ul}) {
    RoundtripBits(QuantBits::B4, n);
  }
}
TEST(Packing, Roundtrip_6bit) {
  for (size_t n : {1ul, 4ul, 7ul, 12ul, 13ul, 128ul, 257ul}) {
    RoundtripBits(QuantBits::B6, n);
  }
}
TEST(Packing, Roundtrip_8bit) {
  for (size_t n : {1ul, 8ul, 16ul, 1024ul}) {
    RoundtripBits(QuantBits::B8, n);
  }
}
TEST(Packing, Roundtrip_12bit) {
  for (size_t n : {1ul, 2ul, 3ul, 16ul, 17ul, 256ul, 257ul}) {
    RoundtripBits(QuantBits::B12, n);
  }
}

TEST(Packing, BitstreamLayoutMatchesSpec_4bit) {
  // Codes: 0x3, 0xA, 0x5, 0xC
  std::vector<uint16_t> codes = {0x3, 0xA, 0x5, 0xC};
  std::vector<uint8_t> packed(2, 0);
  PackCodes(codes.data(), codes.size(), QuantBits::B4, packed.data());
  // dim 0 low nibble, dim 1 high nibble of byte 0 -> 0xA3.
  // dim 2 low nibble, dim 3 high nibble of byte 1 -> 0xC5.
  EXPECT_EQ(packed[0], 0xA3);
  EXPECT_EQ(packed[1], 0xC5);
}

TEST(Packing, BitstreamLayoutMatchesSpec_6bit) {
  // 4 codes -> 3 bytes.
  // Codes (each <64): 0b000001, 0b101010, 0b111100, 0b010101.
  std::vector<uint16_t> codes = {0x01, 0x2A, 0x3C, 0x15};
  std::vector<uint8_t> packed(3, 0);
  PackCodes(codes.data(), codes.size(), QuantBits::B6, packed.data());
  // Build expected little-endian bitstream manually.
  uint32_t bitstream = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    bitstream |= (static_cast<uint32_t>(codes[i]) & 0x3F) << (i * 6);
  }
  EXPECT_EQ(packed[0], (bitstream >> 0) & 0xFF);
  EXPECT_EQ(packed[1], (bitstream >> 8) & 0xFF);
  EXPECT_EQ(packed[2], (bitstream >> 16) & 0xFF);
}
