#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "turboquant/turboquant.h"

#include "internal.h"
#include "rotator_padded.h"

using turboquant::QuantBits;
using turboquant::Quantizer;
using turboquant::internal::DequantizeBeta;
using turboquant::internal::QuantizeBeta;
using turboquant::internal::RotatorPadded;

namespace {

double Mse(const float* a, const float* b, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    s += d * d;
  }
  return s / n;
}

double Variance(const float* a, size_t n) {
  double mean = 0;
  for (size_t i = 0; i < n; ++i) mean += a[i];
  mean /= n;
  double s = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = a[i] - mean;
    s += d * d;
  }
  return s / n;
}

}  // namespace

// Public-API roundtrip: construct a Quantizer, encode + decode, check SNR.
// The Quantizer auto-routes based on (dim, bits) — these tests verify the
// composite path works end-to-end at every supported bit width.
class RoundtripTest : public ::testing::TestWithParam<QuantBits> {};

TEST_P(RoundtripTest, MseDecreasesWithBits) {
  const QuantBits bits = GetParam();
  const size_t dim = 128;  // power of two → padded path
  Quantizer q(dim, bits);
  std::mt19937_64 rng(0xABCD0001);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(q.payload_bytes());
  q.Quantize(x.data(), 1, payload.data());

  std::vector<float> y(dim);
  q.Dequantize(payload.data(), 1, y.data());

  const double mse = Mse(x.data(), y.data(), dim);
  const double var = Variance(x.data(), dim);
  const double snr = var / std::max(mse, 1e-30);
  switch (bits) {
    case QuantBits::B1:  EXPECT_GT(snr, 0.3);  break;
    case QuantBits::B2:  EXPECT_GT(snr, 1.5);  break;
    case QuantBits::B3:  EXPECT_GT(snr, 6.0);  break;
    case QuantBits::B4:  EXPECT_GT(snr, 30.0); break;
    case QuantBits::B6:  EXPECT_GT(snr, 200.0); break;
    case QuantBits::B8:  EXPECT_GT(snr, 5000.0); break;
    case QuantBits::B12: EXPECT_GT(snr, 1e5);  break;
  }
}

INSTANTIATE_TEST_SUITE_P(AllBitWidths, RoundtripTest,
                         ::testing::Values(QuantBits::B1, QuantBits::B2,
                                           QuantBits::B3, QuantBits::B4,
                                           QuantBits::B6, QuantBits::B8,
                                           QuantBits::B12));

TEST(Roundtrip, PayloadHeaderIsScale) {
  const size_t dim = 100;  // not 3*2^k, not pow2 → padded path with padded=128.
  Quantizer q(dim, QuantBits::B8, /*seed=*/1);
  std::mt19937_64 rng(0xABCD0002);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(q.payload_bytes());
  q.Quantize(x.data(), 1, payload.data());

  float scale;
  std::memcpy(&scale, payload.data(), sizeof(float));
  EXPECT_GT(scale, 0.0f);
  // For dim=100 (padded to 128) at b8: 4 (scale) + 128 (codes) = 132 bytes.
  EXPECT_EQ(q.payload_bytes(), 4u + 128u);
}

// Routing sanity check: dim = 3 * 2^k uses the mixed-radix path (smaller
// payload because no padding).
TEST(Roundtrip, Mixed3PathChosenForSupportedDim) {
  // d=768: padded would be 1024; mixed-radix uses 768. At b4 that's a
  // 384-byte code region (+4 header) vs 512 for padded.
  EXPECT_EQ(Quantizer::PayloadBytes(768, QuantBits::B4), 4u + 384u);
  // d=1024 (already pow2, not 3*2^k): padded → still 1024 codes.
  EXPECT_EQ(Quantizer::PayloadBytes(1024, QuantBits::B4), 4u + 512u);
  // d=1000: not 3*2^k → padded to 1024.
  EXPECT_EQ(Quantizer::PayloadBytes(1000, QuantBits::B4), 4u + 512u);
}

TEST(Roundtrip, NonPowerOf2Dim) {
  const size_t dim = 100;
  Quantizer q(dim, QuantBits::B8, /*seed=*/9);
  std::mt19937_64 rng(0xABCD0003);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(q.payload_bytes());
  q.Quantize(x.data(), 1, payload.data());
  std::vector<float> y(dim);
  q.Dequantize(payload.data(), 1, y.data());

  const double mse = Mse(x.data(), y.data(), dim);
  EXPECT_LT(mse, 1e-2);
}

// Batched API: encoding many vectors in one call must be bit-equivalent to
// encoding them one at a time.
TEST(Roundtrip, BatchedMatchesSingle) {
  const size_t dim = 768;  // mixed3-supported
  const size_t n = 16;
  Quantizer q(dim, QuantBits::B4, /*seed=*/0xCAFE);
  std::mt19937_64 rng(0xD00D);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(n * dim);
  for (auto& v : x) v = dist(rng);

  std::vector<uint8_t> batched(n * q.payload_bytes());
  q.Quantize(x.data(), n, batched.data());

  std::vector<uint8_t> per_vec(n * q.payload_bytes());
  for (size_t i = 0; i < n; ++i) {
    q.Quantize(x.data() + i * dim, 1, per_vec.data() + i * q.payload_bytes());
  }
  EXPECT_EQ(batched, per_vec);

  // Dequantize batched and compare.
  std::vector<float> rec_batched(n * dim);
  q.Dequantize(batched.data(), n, rec_batched.data());
  std::vector<float> rec_single(n * dim);
  for (size_t i = 0; i < n; ++i) {
    q.Dequantize(batched.data() + i * q.payload_bytes(), 1,
                 rec_single.data() + i * dim);
  }
  for (size_t k = 0; k < n * dim; ++k) {
    EXPECT_FLOAT_EQ(rec_batched[k], rec_single[k]) << "k=" << k;
  }
}

// Beta-codebook variant: SNR should improve with bit width. Verifies the
// internal beta path directly (used by the bench head-to-head).
class BetaRoundtripTest : public ::testing::TestWithParam<QuantBits> {};

TEST_P(BetaRoundtripTest, MseDecreasesWithBits) {
  const QuantBits bits = GetParam();
  const size_t dim = 768;
  RotatorPadded R(dim, 0xCAFE);
  std::mt19937_64 rng(0xABCD0001);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(turboquant::internal::PayloadSizePadded(dim, bits));
  QuantizeBeta(R, bits, x.data(), payload.data());
  std::vector<float> y(dim);
  DequantizeBeta(R, bits, payload.data(), y.data());

  const double mse = Mse(x.data(), y.data(), dim);
  const double var = Variance(x.data(), dim);
  const double snr = var / std::max(mse, 1e-30);
  switch (bits) {
    case QuantBits::B1:  EXPECT_GT(snr, 0.3);  break;
    case QuantBits::B2:  EXPECT_GT(snr, 1.5);  break;
    case QuantBits::B3:  EXPECT_GT(snr, 6.0);  break;
    case QuantBits::B4:  EXPECT_GT(snr, 30.0); break;
    case QuantBits::B6:  EXPECT_GT(snr, 200.0); break;
    case QuantBits::B8:  EXPECT_GT(snr, 1000.0); break;
    case QuantBits::B12: EXPECT_GT(snr, 1e4); break;
  }
}

INSTANTIATE_TEST_SUITE_P(BetaWidths, BetaRoundtripTest,
                         ::testing::Values(QuantBits::B1, QuantBits::B2,
                                           QuantBits::B3, QuantBits::B4,
                                           QuantBits::B6, QuantBits::B8,
                                           QuantBits::B12));
