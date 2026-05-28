#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "rotation.h"
#include "rotator_padded.h"
#include "turboquant/turboquant.h"

using turboquant::ApplySigns;
using turboquant::HadamardTransform;
using turboquant::internal::RotatorPadded;

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
    RotatorPadded R(dim, 42);
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

// Mixed-radix orthogonal rotation for dim = 3 * 2^k. Must preserve norms
// and inner products to within FP rounding (no padding).
#include "rotator_mixed.h"
using turboquant::internal::RotatorMixed3;

TEST(RotatorMixed3, NormPreservation) {
  for (size_t dim : {12ul, 48ul, 192ul, 768ul, 1536ul}) {
    ASSERT_TRUE(RotatorMixed3::DimSupported(dim)) << "dim=" << dim;
    RotatorMixed3 R(dim, 0xC0FFEE);
    std::mt19937_64 rng(0x42 + dim);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> x(dim), y(dim);
    for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);
    R.Apply(x.data(), y.data());
    const float nx = L2Norm(x.data(), dim);
    const float ny = L2Norm(y.data(), dim);
    EXPECT_NEAR(nx, ny, 1e-3f * nx + 1e-5f) << "dim=" << dim;
  }
}

TEST(RotatorMixed3, InnerProductPreservation) {
  for (size_t dim : {12ul, 48ul, 192ul, 768ul, 1536ul}) {
    RotatorMixed3 R(dim, 0xDEADBEEF + dim);
    std::mt19937_64 rng(0x7777 + dim);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> a(dim), b(dim), ra(dim), rb(dim);
    for (size_t i = 0; i < dim; ++i) {
      a[i] = dist(rng);
      b[i] = dist(rng);
    }
    R.Apply(a.data(), ra.data());
    R.Apply(b.data(), rb.data());
    double ip_orig = 0, ip_rot = 0;
    for (size_t i = 0; i < dim; ++i) {
      ip_orig += static_cast<double>(a[i]) * b[i];
      ip_rot += static_cast<double>(ra[i]) * rb[i];
    }
    EXPECT_NEAR(ip_orig, ip_rot, 1e-3 * std::abs(ip_orig) + 1e-3)
        << "dim=" << dim;
  }
}

TEST(RotatorMixed3, InverseRoundtrip) {
  for (size_t dim : {12ul, 192ul, 768ul, 1536ul}) {
    RotatorMixed3 R(dim, 0x123 + dim);
    std::mt19937_64 rng(0x999 + dim);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> x(dim);
    for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);
    std::vector<float> y(dim);
    R.Apply(x.data(), y.data());
    std::vector<float> back(dim);
    R.ApplyInverse(y.data(), back.data());
    for (size_t i = 0; i < dim; ++i) {
      EXPECT_NEAR(x[i], back[i], 1e-3f * std::abs(x[i]) + 1e-3f)
          << "dim=" << dim << " i=" << i;
    }
  }
}

TEST(RotatorMixed3, RejectsUnsupportedDim) {
  EXPECT_FALSE(RotatorMixed3::DimSupported(100));
  EXPECT_FALSE(RotatorMixed3::DimSupported(1024));  // pow2, not 3*pow2
  EXPECT_FALSE(RotatorMixed3::DimSupported(1000));
  EXPECT_TRUE(RotatorMixed3::DimSupported(3));
  EXPECT_TRUE(RotatorMixed3::DimSupported(6));
  EXPECT_TRUE(RotatorMixed3::DimSupported(768));
  EXPECT_TRUE(RotatorMixed3::DimSupported(3072));
}

TEST(Rotation, RotatorInverseRoundtrip) {
  for (size_t dim : {5ul, 64ul, 100ul, 257ul}) {
    std::mt19937_64 rng(0x123 + dim);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> x(dim);
    for (size_t i = 0; i < dim; ++i) x[i] = dist(rng);

    RotatorPadded R(dim, 7);
    std::vector<float> rotated(R.padded_dim());
    R.Apply(x.data(), rotated.data());
    std::vector<float> back(dim);
    R.ApplyInverse(rotated.data(), back.data());
    for (size_t i = 0; i < dim; ++i) {
      EXPECT_NEAR(x[i], back[i], 1e-3f) << "i=" << i;
    }
  }
}
