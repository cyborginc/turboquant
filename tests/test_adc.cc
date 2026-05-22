#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "turboquant/turboquant.h"

using turboquant::AdcScore;
using turboquant::AdcStats;
using turboquant::Dequantize;
using turboquant::PayloadSize;
using turboquant::QuantBits;
using turboquant::Quantize;
using turboquant::RotateQuery;
using turboquant::Rotator;

namespace {
double DotF(const float* a, const float* b, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i)
    s += static_cast<double>(a[i]) * b[i];
  return s;
}
double NormSq(const float* a, size_t n) { return DotF(a, a, n); }

double RelErr(double a, double b) {
  const double s = std::max(1e-6, std::abs(b));
  return std::abs(a - b) / s;
}

}  // namespace

class AdcVsDequant : public ::testing::TestWithParam<QuantBits> {};

TEST_P(AdcVsDequant, MatchesDecodedDot) {
  const QuantBits bits = GetParam();
  const size_t dim = 256;
  Rotator R(dim, 0xBEEF);
  std::mt19937_64 rng(0x42);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim), q(dim);
  for (size_t i = 0; i < dim; ++i) {
    x[i] = dist(rng);
    q[i] = dist(rng);
  }

  std::vector<uint8_t> payload(PayloadSize(dim, bits));
  Quantize(R, bits, x.data(), payload.data());

  // Decode and compute "ground truth" in the rotated/dequantized space.
  std::vector<float> x_hat(dim);
  Dequantize(R, bits, payload.data(), x_hat.data());
  const double dot_truth = DotF(q.data(), x_hat.data(), dim);
  const double norm2_truth = NormSq(x_hat.data(), dim);

  // ADC stats: must agree closely (both operate on the same rotated levels).
  std::vector<float> q_rot(R.padded_dim());
  RotateQuery(R, q.data(), q_rot.data());
  AdcStats stats = AdcScore(R, bits, q_rot.data(), payload.data());

  EXPECT_LT(RelErr(stats.dot, dot_truth), 5e-3)
      << "bits=" << static_cast<int>(bits)
      << " adc=" << stats.dot << " truth=" << dot_truth;
  EXPECT_LT(RelErr(stats.decoded_norm2, norm2_truth), 5e-3)
      << "bits=" << static_cast<int>(bits)
      << " adc_n2=" << stats.decoded_norm2 << " truth_n2=" << norm2_truth;
}

INSTANTIATE_TEST_SUITE_P(AllBitWidths, AdcVsDequant,
                         ::testing::Values(QuantBits::B1, QuantBits::B2,
                                           QuantBits::B4, QuantBits::B6,
                                           QuantBits::B8, QuantBits::B12));

TEST(AdcConsistency, ApproximatesTrueDot) {
  // ADC should be close to true <q, x> for high bit widths.
  const size_t dim = 512;
  Rotator R(dim, 0xC0DE);
  std::mt19937_64 rng(0x99);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(dim), q(dim);
  for (size_t i = 0; i < dim; ++i) {
    x[i] = dist(rng);
    q[i] = dist(rng);
  }
  const double truth = DotF(q.data(), x.data(), dim);

  std::vector<uint8_t> payload(PayloadSize(dim, QuantBits::B8));
  Quantize(R, QuantBits::B8, x.data(), payload.data());
  std::vector<float> q_rot(R.padded_dim());
  RotateQuery(R, q.data(), q_rot.data());
  AdcStats stats = AdcScore(R, QuantBits::B8, q_rot.data(), payload.data());

  EXPECT_LT(RelErr(stats.dot, truth), 5e-2)
      << "adc=" << stats.dot << " truth=" << truth;
}
