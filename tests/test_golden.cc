// Format-stability tests.
//
// TurboQuant payloads are written to disk by downstream consumers (cyborgdb-core
// stores them as the Stage-2 rerank vectors) and decoded by whatever library
// version is linked at read time. The encoded bytes are therefore a persisted
// format, not an implementation detail: changing the rotation, the Lloyd-Max
// codebook, or the bit packing silently invalidates every stored index.
//
// These tests pin that format. A failure here is not necessarily a bug — it may
// be a deliberate format change — but it MUST be accompanied by a version bump
// and a migration story for already-written data. Do not "fix" a failure by
// regenerating the constants.

#include "turboquant/turboquant.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <hwy/targets.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

using turboquant::QuantBits;
using turboquant::Quantizer;

// Deterministic input generation, deliberately NOT using <random> distributions.
// std::normal_distribution and friends are implementation-defined, so goldens
// built with libstdc++ would not reproduce under libc++. SplitMix64 plus a
// fixed int->float mapping is portable across compilers and platforms.
uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// `n` row-major vectors with coordinates uniform in [-1, 1).
std::vector<float> MakeVectors(size_t n, size_t dim, uint64_t seed) {
  std::vector<float> v(n * dim);
  uint64_t s = seed;
  for (auto& f : v) {
    const uint32_t bits = static_cast<uint32_t>(SplitMix64(s) >> 32);
    f = static_cast<float>(static_cast<int32_t>(bits)) * (1.0f / 2147483648.0f);
  }
  return v;
}

uint64_t Fnv1a(const uint8_t* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

constexpr size_t kGoldenBatch = 8;
uint64_t GoldenSeed(size_t dim) { return 0x5EED0000ULL + dim; }

// Encode kGoldenBatch fixed vectors and hash the whole payload buffer.
uint64_t EncodeHash(size_t dim, QuantBits bits, size_t* payload_bytes_out) {
  Quantizer q(dim, bits);
  const std::vector<float> x = MakeVectors(kGoldenBatch, dim, GoldenSeed(dim));
  std::vector<uint8_t> payloads(kGoldenBatch * q.payload_bytes());
  q.Quantize(x.data(), kGoldenBatch, payloads.data());
  if (payload_bytes_out) *payload_bytes_out = q.payload_bytes();
  return Fnv1a(payloads.data(), payloads.size());
}

struct GoldenCase {
  size_t dim;
  QuantBits bits;
  size_t payload_bytes;
  uint64_t hash;
};

// dim=768 exercises the mixed-radix path (dim = 3 * 2^k, no padding);
// dim=1000 exercises the padded Walsh-Hadamard path (padded to 1024).
// Generated on 2026-09-01 against NEON and NEON_BF16, which agree byte-for-byte.
const GoldenCase kGoldens[] = {
    { 768, QuantBits::B1,   100, 0xcb4a051656fee950ULL},
    { 768, QuantBits::B2,   196, 0xd62a6d4f26be77f2ULL},
    { 768, QuantBits::B3,   292, 0x7b3e21c3c8917a38ULL},
    { 768, QuantBits::B4,   388, 0x3660bf9b46eebf1eULL},
    { 768, QuantBits::B6,   580, 0x64cf5dc2348f3dccULL},
    { 768, QuantBits::B8,   772, 0x0d551a221d5c1083ULL},
    { 768, QuantBits::B12, 1156, 0xaaf57915702bfbe1ULL},
    {1000, QuantBits::B1,   132, 0x610f86d29d2be84bULL},
    {1000, QuantBits::B2,   260, 0x2730ddfc518f4467ULL},
    {1000, QuantBits::B3,   388, 0x08065455cbfe5539ULL},
    {1000, QuantBits::B4,   516, 0x5ff28fa94fa6fb3cULL},
    {1000, QuantBits::B6,   772, 0x870af48f78625134ULL},
    {1000, QuantBits::B8,  1028, 0xa8740ef639b69c21ULL},
    {1000, QuantBits::B12, 1540, 0x0715d4afb343192dULL},
};

}  // namespace

// The encoded bytes for a fixed (dim, bits, seed) and fixed input are frozen.
// See the file header before changing any constant here.
TEST(GoldenFormat, EncodedBytesAreStable) {
  for (const GoldenCase& g : kGoldens) {
    SCOPED_TRACE("dim=" + std::to_string(g.dim) +
                 " bits=" + std::to_string(static_cast<int>(g.bits)));
    size_t payload_bytes = 0;
    const uint64_t hash = EncodeHash(g.dim, g.bits, &payload_bytes);
    EXPECT_EQ(payload_bytes, g.payload_bytes) << "payload size changed";
    EXPECT_EQ(hash, g.hash)
        << "encoded bytes changed — this is an on-disk format break. Bump the "
           "format version and provide a migration path; do not just update "
           "this constant.";
  }
}

// Payload size is part of the format contract: downstream sizes fixed-width
// storage rows from it before any vector is encoded.
TEST(GoldenFormat, PayloadSizeFormula) {
  // Mixed-radix dims (3 * 2^k) pack the true dim with no padding.
  for (size_t dim : {size_t{768}, size_t{1536}, size_t{3072}}) {
    for (QuantBits b : {QuantBits::B4, QuantBits::B6, QuantBits::B8,
                        QuantBits::B12}) {
      const size_t expect =
          4 + (dim * static_cast<size_t>(b) + 7) / 8;
      EXPECT_EQ(Quantizer::PayloadBytes(dim, b), expect)
          << "dim=" << dim << " bits=" << static_cast<int>(b);
    }
  }
  // Padded dims round the code region up to next_pow2(dim).
  EXPECT_EQ(Quantizer::PayloadBytes(1000, QuantBits::B8), 4u + 1024u);
  EXPECT_EQ(Quantizer::PayloadBytes(1024, QuantBits::B8), 4u + 1024u);
}

// ---------------------------------------------------------------------------
// On-disk byte layout
//
// The hash tests above detect that the format moved; these pin what the format
// *is*, so a failure says which part broke and the layout stays documented for
// anyone reading stored payloads without linking TurboQuant.
// ---------------------------------------------------------------------------

// Layout of one payload: [0,4) little-endian float32 scale, then ceil(code_dim
// * bits / 8) bytes of packed codes. code_dim is dim for the mixed-radix path
// and next_pow2(dim) for the padded path.
TEST(GoldenLayout, PayloadIsLittleEndianScaleThenPackedCodes) {
  constexpr size_t dim = 768;  // mixed-radix: code_dim == dim
  for (QuantBits bits : {QuantBits::B4, QuantBits::B8, QuantBits::B12}) {
    Quantizer q(dim, bits);
    const std::vector<float> x = MakeVectors(1, dim, 0xABCDEF);
    std::vector<uint8_t> payload(q.payload_bytes());
    q.Quantize(x.data(), 1, payload.data());

    // Header is exactly 4 bytes and the rest is the code region.
    const size_t code_bytes = (dim * static_cast<size_t>(bits) + 7) / 8;
    ASSERT_EQ(q.payload_bytes(), 4u + code_bytes);

    // The scale reads back as a finite, positive float32 via a byte copy —
    // i.e. it is stored in native little-endian order with no tag or padding.
    float scale = 0.0f;
    std::memcpy(&scale, payload.data(), sizeof(float));
    EXPECT_TRUE(std::isfinite(scale))
        << "bits=" << static_cast<int>(bits) << " scale is not finite";
    EXPECT_GT(scale, 0.0f) << "bits=" << static_cast<int>(bits);

    // Spell out the little-endian contract independently of the host, so a
    // big-endian port fails here rather than silently writing swapped bytes.
    uint32_t raw = 0;
    std::memcpy(&raw, payload.data(), sizeof(uint32_t));
    const uint32_t rebuilt = static_cast<uint32_t>(payload[0]) |
                             (static_cast<uint32_t>(payload[1]) << 8) |
                             (static_cast<uint32_t>(payload[2]) << 16) |
                             (static_cast<uint32_t>(payload[3]) << 24);
    EXPECT_EQ(raw, rebuilt) << "scale header is not little-endian";
  }
}

// Batched payloads are contiguous and fixed-stride, and encoding a vector alone
// gives the same bytes as its slice of a batch. Downstream indexes rows by
// multiplying by payload_bytes(), so this is load-bearing.
TEST(GoldenLayout, BatchIsContiguousFixedStride) {
  constexpr size_t dim = 768;
  constexpr size_t n = 5;
  for (QuantBits bits : {QuantBits::B4, QuantBits::B8}) {
    Quantizer q(dim, bits);
    const size_t ps = q.payload_bytes();
    const std::vector<float> x = MakeVectors(n, dim, 0x1234);

    std::vector<uint8_t> batch(n * ps);
    q.Quantize(x.data(), n, batch.data());

    for (size_t i = 0; i < n; ++i) {
      std::vector<uint8_t> single(ps);
      q.Quantize(x.data() + i * dim, 1, single.data());
      const std::vector<uint8_t> slice(batch.begin() + i * ps,
                                       batch.begin() + (i + 1) * ps);
      EXPECT_EQ(single, slice)
          << "row " << i << " differs between single and batch encode (bits="
          << static_cast<int>(bits) << ")";
    }
  }
}

// A literal byte-for-byte golden for one configuration. The hashes above cover
// every case compactly; this one spells the bytes out so a format break shows
// exactly which bytes moved instead of just a changed digest.
TEST(GoldenLayout, LiteralBytesForOneVector) {
  constexpr size_t dim = 768;
  Quantizer q(dim, QuantBits::B4);
  const std::vector<float> x = MakeVectors(1, dim, 0xFEED);
  std::vector<uint8_t> payload(q.payload_bytes());
  q.Quantize(x.data(), 1, payload.data());

  ASSERT_EQ(payload.size(), 388u);
  // First 16 bytes: 4-byte scale followed by the first 12 bytes of codes.
  static const uint8_t kExpectedPrefix[16] = {
      0x27, 0x61, 0x81, 0x41, 0xAD, 0x7A, 0x4B, 0x79,
      0x88, 0x68, 0x78, 0x44, 0x72, 0x88, 0x3B, 0xC8,
  };
  const std::vector<uint8_t> got(payload.begin(), payload.begin() + 16);
  const std::vector<uint8_t> want(kExpectedPrefix, kExpectedPrefix + 16);
  EXPECT_EQ(got, want) << "payload prefix changed";
}

// The encode path runs under Highway dynamic dispatch, so the SIMD target is
// chosen at runtime from whatever the host CPU supports. A persisted format
// must not depend on that choice: an index written on an AVX-512 host has to
// decode identically to one written on AVX2 or NEON.
//
// The scale field is the value at risk — on the Beta path it derives from
// CentroidInnerProduct, a horizontal reduction whose partial-sum grouping
// varies with vector width. This test encodes the same input under every
// target the build generated and the host supports, and requires byte
// equality. On a single-ISA host it only covers the targets available there
// (e.g. NEON and NEON_BF16 on Apple Silicon); on x86 CI it covers the
// SSE4/AVX2/AVX-512 spread, which is where a lane-width difference would show.
TEST(GoldenFormat, EncodingIsIdenticalAcrossSimdTargets) {
  const std::vector<int64_t> targets = hwy::SupportedAndGeneratedTargets();

  // Log the coverage: which targets a given CI runner exercises depends on its
  // CPU, so a green check is only meaningful alongside the list it compared.
  std::string target_list;
  for (int64_t t : targets) {
    if (!target_list.empty()) target_list += ", ";
    target_list += hwy::TargetName(t);
  }
  std::cout << "[ TARGETS  ] comparing encodings across: " << target_list
            << std::endl;

  if (targets.size() < 2) {
    GTEST_SKIP() << "only one SIMD target available on this host ("
                 << target_list << ")";
  }

  struct Restore {
    ~Restore() { hwy::SetSupportedTargetsForTest(0); }
  } restore;

  for (const GoldenCase& g : kGoldens) {
    std::vector<uint8_t> reference;
    std::string reference_target;
    for (int64_t t : targets) {
      hwy::SetSupportedTargetsForTest(t);
      Quantizer q(g.dim, g.bits);
      const std::vector<float> x =
          MakeVectors(kGoldenBatch, g.dim, GoldenSeed(g.dim));
      std::vector<uint8_t> payloads(kGoldenBatch * q.payload_bytes());
      q.Quantize(x.data(), kGoldenBatch, payloads.data());

      if (reference.empty()) {
        reference = std::move(payloads);
        reference_target = hwy::TargetName(t);
      } else {
        EXPECT_EQ(payloads, reference)
            << "target " << hwy::TargetName(t) << " disagrees with "
            << reference_target << " at dim=" << g.dim
            << " bits=" << static_cast<int>(g.bits)
            << " — the on-disk format depends on the host CPU";
      }
    }
  }
}
