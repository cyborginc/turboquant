#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "rotation.h"
#include "turboquant/turboquant.h"

using turboquant::ApplySigns;
using turboquant::HadamardTransform;
using turboquant::Rotator;

namespace {
float L2Norm(const float* v, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(v[i]) * v[i];
  return static_cast<float>(std::sqrt(s));
}
}  // namespace

TEST(Rotation, HadamardPreservesNorm) {
  for (size_t n : {1ul, 2ul, 4ul, 8ul, 16ul, 64ul, 256ul, 1024ul}) {
    std::mt19937_64 rng(0xABCDull + n);
    std::vector<float> v(n);
    std::normal_distribution<float> dist(0, 1);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    const float before = L2Norm(v.data(), n);
    HadamardTransform(v.data(), n);
    const float after = L2Norm(v.data(), n);
    EXPECT_NEAR(before, after, 1e-3f * before + 1e-5f) << "n=" << n;
  }
}

TEST(Rotation, HadamardSelfInverse) {
  for (size_t n : {2ul, 4ul, 8ul, 16ul, 64ul, 1024ul}) {
    std::mt19937_64 rng(0xDEAD + n);
    std::vector<float> v(n), original(n);
    std::normal_distribution<float> dist(0, 1);
    for (size_t i = 0; i < n; ++i) {
      v[i] = dist(rng);
      original[i] = v[i];
    }
    HadamardTransform(v.data(), n);
    HadamardTransform(v.data(), n);
    for (size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(v[i], original[i], 1e-3f) << "n=" << n << " i=" << i;
    }
  }
}

TEST(Rotation, RotatorPreservesInnerProduct) {
  // Orthogonal transform preserves <x, y>.
  for (size_t dim : {3ul, 7ul, 32ul, 128ul, 129ul, 500ul}) {
    std::mt19937_64 rng(0x7777 + dim);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> x(dim), y(dim);
    for (size_t i = 0; i < dim; ++i) {
      x[i] = dist(rng);
      y[i] = dist(rng);
    }
    Rotator R(dim, 42);
    std::vector<float> xr(R.padded_dim()), yr(R.padded_dim());
    R.Apply(x.data(), xr.data());
    R.Apply(y.data(), yr.data());

    double a = 0, b = 0;
    for (size_t i = 0; i < dim; ++i) a += static_cast<double>(x[i]) * y[i];
    for (size_t i = 0; i < R.padded_dim(); ++i)
      b += static_cast<double>(xr[i]) * yr[i];
    EXPECT_NEAR(a, b, 1e-3 * std::abs(a) + 1e-3) << "dim=" << dim;
  }
}

TEST(Rotation, RotatorInverseRoundtrip) {
  for (size_t dim : {5ul, 64ul, 100ul, 257ul}) {
    std::mt19937_64 rng(0x123 + dim);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> x(dim);
    for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

    Rotator R(dim, 7);
    std::vector<float> rotated(R.padded_dim());
    R.Apply(x.data(), rotated.data());
    std::vector<float> back(dim);
    R.ApplyInverse(rotated.data(), back.data());
    for (size_t i = 0; i < dim; ++i) {
      EXPECT_NEAR(x[i], back[i], 1e-3f) << "i=" << i;
    }
  }
}
