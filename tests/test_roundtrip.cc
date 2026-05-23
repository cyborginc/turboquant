#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "turboquant/turboquant.h"

#include "internal.h"

using turboquant::Dequantize;
using turboquant::PayloadSize;
using turboquant::QuantBits;
using turboquant::Quantize;
using turboquant::Rotator;
using turboquant::internal::DequantizeBeta;
using turboquant::internal::QuantizeBeta;

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

class RoundtripTest : public ::testing::TestWithParam<QuantBits> {};

TEST_P(RoundtripTest, MseDecreasesWithBits) {
  const QuantBits bits = GetParam();
  const size_t dim = 128;
  Rotator R(dim, 0xCAFE);
  std::mt19937_64 rng(0xABCD0001);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(PayloadSize(dim, bits));
  Quantize(R, bits, x.data(), payload.data());

  std::vector<float> y(dim);
  Dequantize(R, bits, payload.data(), y.data());

  const double mse = Mse(x.data(), y.data(), dim);
  const double var = Variance(x.data(), dim);
  // SNR ~ var/mse should rise sharply with bit width.
  const double snr = var / std::max(mse, 1e-30);
  // Loose lower bounds; just sanity-check the trend.
  // Quantize/Dequantize auto-route by bit width (beta for b1/2/3/4/6, affine
  // for b8/12). The SNR floors below cover both paths conservatively.
  switch (bits) {
    case QuantBits::B1:
      EXPECT_GT(snr, 0.3);
      break;
    case QuantBits::B2:
      EXPECT_GT(snr, 1.5);
      break;
    case QuantBits::B3:
      EXPECT_GT(snr, 6.0);
      break;
    case QuantBits::B4:
      EXPECT_GT(snr, 30.0);
      break;
    case QuantBits::B6:
      EXPECT_GT(snr, 200.0);
      break;
    case QuantBits::B8:
      EXPECT_GT(snr, 5000.0);
      break;
    case QuantBits::B12:
      EXPECT_GT(snr, 1e5);
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(AllBitWidths, RoundtripTest,
                         ::testing::Values(QuantBits::B1, QuantBits::B2,
                                           QuantBits::B3, QuantBits::B4,
                                           QuantBits::B6, QuantBits::B8,
                                           QuantBits::B12));

TEST(Roundtrip, PayloadHeaderIsScale) {
  const size_t dim = 96;
  Rotator R(dim, 1);
  std::mt19937_64 rng(0xABCD0002);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(PayloadSize(dim, QuantBits::B8));
  Quantize(R, QuantBits::B8, x.data(), payload.data());

  float scale;
  std::memcpy(&scale, payload.data(), sizeof(float));
  EXPECT_GT(scale, 0.0f);
  // PayloadSize is 4 (scale) + ceil(padded_dim * bits / 8).
  EXPECT_EQ(PayloadSize(dim, QuantBits::B8), 4u + 128u);
}

// Beta-codebook variant: roundtrip MSE should decrease with bit width, and at
// each bit width Beta's SNR should be at least as good as affine on rotated
// Gaussian inputs (where the rotated coord distribution matches the codebook).
class BetaRoundtripTest : public ::testing::TestWithParam<QuantBits> {};

TEST_P(BetaRoundtripTest, MseDecreasesWithBits) {
  const QuantBits bits = GetParam();
  const size_t dim = 768;
  Rotator R(dim, 0xCAFE);
  std::mt19937_64 rng(0xABCD0001);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(PayloadSize(dim, bits));
  QuantizeBeta(R, bits, x.data(), payload.data());
  std::vector<float> y(dim);
  DequantizeBeta(R, bits, payload.data(), y.data());

  const double mse = Mse(x.data(), y.data(), dim);
  const double var = Variance(x.data(), dim);
  const double snr = var / std::max(mse, 1e-30);

  // Loose lower bounds — sanity-check that quality improves with bits. We do
  // not assert vs the affine path here since recall depends on dataset shape,
  // not synthetic Gaussians.
  switch (bits) {
    case QuantBits::B1: EXPECT_GT(snr, 0.3); break;
    case QuantBits::B2: EXPECT_GT(snr, 1.5); break;
    case QuantBits::B3: EXPECT_GT(snr, 6.0); break;
    case QuantBits::B4: EXPECT_GT(snr, 30.0); break;
    case QuantBits::B6: EXPECT_GT(snr, 200.0); break;
    case QuantBits::B8: EXPECT_GT(snr, 1000.0); break;
    case QuantBits::B12: EXPECT_GT(snr, 1e4); break;
  }
}

INSTANTIATE_TEST_SUITE_P(BetaWidths, BetaRoundtripTest,
                         ::testing::Values(QuantBits::B1, QuantBits::B2,
                                           QuantBits::B3, QuantBits::B4,
                                           QuantBits::B6, QuantBits::B8,
                                           QuantBits::B12));

TEST(Roundtrip, NonPowerOf2Dim) {
  const size_t dim = 100;
  Rotator R(dim, 9);
  std::mt19937_64 rng(0xABCD0003);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim);
  for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

  std::vector<uint8_t> payload(PayloadSize(dim, QuantBits::B8));
  Quantize(R, QuantBits::B8, x.data(), payload.data());

  std::vector<float> y(dim);
  Dequantize(R, QuantBits::B8, payload.data(), y.data());

  const double mse = Mse(x.data(), y.data(), dim);
  EXPECT_LT(mse, 1e-2);
}
