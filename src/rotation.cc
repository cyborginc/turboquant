// clang-format off
// foreach_target.h must precede highway.h and both must lead the file: this
// translation unit re-includes itself once per SIMD target. Do not reorder.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rotation.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
// clang-format on

#include <cmath>
#include <cstddef>

#include "rotation.h"

HWY_BEFORE_NAMESPACE();
namespace turboquant {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Butterfly stride `h` on contiguous chunks of size 2*h: pair (a, b) -> (a+b,
// a-b).
void ButterflyStride(float* data, size_t n, size_t h) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t base = 0; base < n; base += 2 * h) {
    float* a = data + base;
    float* b = data + base + h;
    if (h >= lanes) {
      size_t i = 0;
      for (; i + lanes <= h; i += lanes) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, a + i);
        hn::StoreU(hn::Sub(va, vb), d, b + i);
      }
      for (; i < h; ++i) {
        const float x = a[i], y = b[i];
        a[i] = x + y;
        b[i] = x - y;
      }
    } else {
      // h < lanes: per-element. Keeps things simple for small head stages.
      for (size_t i = 0; i < h; ++i) {
        const float x = a[i], y = b[i];
        a[i] = x + y;
        b[i] = x - y;
      }
    }
  }
}

void ScaleInPlace(float* data, size_t n, float s) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto vs = hn::Set(d, s);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    hn::StoreU(hn::Mul(hn::LoadU(d, data + i), vs), d, data + i);
  }
  for (; i < n; ++i) data[i] *= s;
}

void HadamardTransformImpl(float* data, size_t n) {
  // Iterative WHT: log2(n) stages, each doing pair butterflies at stride h.
  for (size_t h = 1; h < n; h <<= 1) {
    ButterflyStride(data, n, h);
  }
  ScaleInPlace(data, n, 1.0f / std::sqrt(static_cast<float>(n)));
}

void ApplySignsImpl(float* data, const float* signs, size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto vd = hn::LoadU(d, data + i);
    auto vs = hn::LoadU(d, signs + i);
    hn::StoreU(hn::Mul(vd, vs), d, data + i);
  }
  for (; i < n; ++i) data[i] *= signs[i];
}

// --- Fast variants -------------------------------------------------------

void ApplySignsAndScaleImpl(float* data, const float* signs, size_t n,
                            float scale) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto vscale = hn::Set(d, scale);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto vd = hn::LoadU(d, data + i);
    auto vs = hn::LoadU(d, signs + i);
    hn::StoreU(hn::Mul(hn::Mul(vd, vs), vscale), d, data + i);
  }
  for (; i < n; ++i) data[i] = data[i] * signs[i] * scale;
}

// In-register fused stage(s). For every lane count we can do h=1 with
// DupEven/DupOdd + OddEven. On 4-lane targets we additionally fold h=2 via
// the half-half permutes. After this pass the next memory butterfly starts at
// h = (returned `h_done` * 2). Returns the largest in-register h actually
// applied.
size_t FusedInRegisterPass(float* data, size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  if (lanes < 2 || n < lanes) return 0;

  if (lanes == 4) {
    // Fuse h=1 and h=2 in one register.
    for (size_t i = 0; i + lanes <= n; i += lanes) {
      auto v = hn::LoadU(d, data + i);
      // h=1: pair adjacent lanes.
      auto e = hn::DupEven(v);
      auto o = hn::DupOdd(v);
      v = hn::OddEven(hn::Sub(e, o), hn::Add(e, o));
      // h=2: pair the two halves of the (now post-h=1) vector.
      auto lo = hn::ConcatLowerLower(d, v, v);
      auto hi = hn::ConcatUpperUpper(d, v, v);
      v = hn::ConcatLowerLower(d, hn::Sub(lo, hi), hn::Add(lo, hi));
      hn::StoreU(v, d, data + i);
    }
    return 2;
  }

  // Lanes > 4: do h=1 only. h=2 within a vector spans non-half regions and is
  // not portable enough to be worth a target-specific path here.
  for (size_t i = 0; i + lanes <= n; i += lanes) {
    auto v = hn::LoadU(d, data + i);
    auto e = hn::DupEven(v);
    auto o = hn::DupOdd(v);
    v = hn::OddEven(hn::Sub(e, o), hn::Add(e, o));
    hn::StoreU(v, d, data + i);
  }
  return 1;
}

void FastHadamardTransformUnscaledImpl(float* data, size_t n) {
  if (n <= 1) return;
  const size_t h_done = FusedInRegisterPass(data, n);
  for (size_t h = (h_done == 0 ? 1 : h_done * 2); h < n; h <<= 1) {
    ButterflyStride(data, n, h);
  }
  // No scaling here — caller pairs this with ApplySignsAndScale.
}

// Mirror of FusedInRegisterPass that additionally multiplies by `signs` during
// the first SIMD load. Folds the diagonal D matrix into the WHT's first stage,
// removing the standalone ApplySigns pass.
size_t FusedInRegisterPassWithSigns(float* data, const float* signs, size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  if (lanes < 2 || n < lanes) return 0;

  if (lanes == 4) {
    for (size_t i = 0; i + lanes <= n; i += lanes) {
      auto v = hn::LoadU(d, data + i);
      auto s = hn::LoadU(d, signs + i);
      v = hn::Mul(v, s);
      // h=1
      auto e = hn::DupEven(v);
      auto o = hn::DupOdd(v);
      v = hn::OddEven(hn::Sub(e, o), hn::Add(e, o));
      // h=2
      auto lo = hn::ConcatLowerLower(d, v, v);
      auto hi = hn::ConcatUpperUpper(d, v, v);
      v = hn::ConcatLowerLower(d, hn::Sub(lo, hi), hn::Add(lo, hi));
      hn::StoreU(v, d, data + i);
    }
    return 2;
  }

  for (size_t i = 0; i + lanes <= n; i += lanes) {
    auto v = hn::LoadU(d, data + i);
    auto s = hn::LoadU(d, signs + i);
    v = hn::Mul(v, s);
    auto e = hn::DupEven(v);
    auto o = hn::DupOdd(v);
    v = hn::OddEven(hn::Sub(e, o), hn::Add(e, o));
    hn::StoreU(v, d, data + i);
  }
  return 1;
}

// Householder reflection H_3 over the three rows of a 3 × N matrix, SIMD over
// the column dimension. H_3 = I - (2/3) * ones * ones^T; per column the
// formula is `entry -= (2/3) * (a + b + c)`, applied to all three rows.
void Householder3InPlaceImpl(float* data, size_t n_block) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto two_thirds = hn::Set(d, 2.0f / 3.0f);
  float* row0 = data;
  float* row1 = data + n_block;
  float* row2 = data + 2 * n_block;
  size_t j = 0;
  for (; j + lanes <= n_block; j += lanes) {
    const auto va = hn::LoadU(d, row0 + j);
    const auto vb = hn::LoadU(d, row1 + j);
    const auto vc = hn::LoadU(d, row2 + j);
    const auto vs = hn::Mul(two_thirds, hn::Add(hn::Add(va, vb), vc));
    hn::StoreU(hn::Sub(va, vs), d, row0 + j);
    hn::StoreU(hn::Sub(vb, vs), d, row1 + j);
    hn::StoreU(hn::Sub(vc, vs), d, row2 + j);
  }
  for (; j < n_block; ++j) {
    const float a = row0[j], b = row1[j], c = row2[j];
    const float s = (2.0f / 3.0f) * (a + b + c);
    row0[j] = a - s;
    row1[j] = b - s;
    row2[j] = c - s;
  }
}

void ForwardRotateImpl(float* data, const float* signs, size_t n) {
  if (n == 0) return;
  if (n == 1) {
    data[0] *= signs[0];
    return;
  }
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);

  size_t h_done = 0;
  if (n >= lanes && lanes >= 2) {
    h_done = FusedInRegisterPassWithSigns(data, signs, n);
  } else {
    // n smaller than SIMD width: scalar sign pass, then standard butterflies.
    for (size_t i = 0; i < n; ++i) data[i] *= signs[i];
  }
  for (size_t h = (h_done == 0 ? 1 : h_done * 2); h < n; h <<= 1) {
    ButterflyStride(data, n, h);
  }
  ScaleInPlace(data, n, 1.0f / std::sqrt(static_cast<float>(n)));
}

}  // namespace HWY_NAMESPACE
}  // namespace turboquant
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace turboquant {

HWY_EXPORT(HadamardTransformImpl);
HWY_EXPORT(ApplySignsImpl);
HWY_EXPORT(FastHadamardTransformUnscaledImpl);
HWY_EXPORT(ApplySignsAndScaleImpl);
HWY_EXPORT(ForwardRotateImpl);
HWY_EXPORT(Householder3InPlaceImpl);

void HadamardTransform(float* data, size_t n) {
  HWY_DYNAMIC_DISPATCH(HadamardTransformImpl)(data, n);
}

void ApplySigns(float* data, const float* signs, size_t n) {
  HWY_DYNAMIC_DISPATCH(ApplySignsImpl)(data, signs, n);
}

void FastHadamardTransformUnscaled(float* data, size_t n) {
  HWY_DYNAMIC_DISPATCH(FastHadamardTransformUnscaledImpl)(data, n);
}

void ApplySignsAndScale(float* data, const float* signs, size_t n,
                        float scale) {
  HWY_DYNAMIC_DISPATCH(ApplySignsAndScaleImpl)(data, signs, n, scale);
}

void ForwardRotate(float* data, const float* signs, size_t n) {
  HWY_DYNAMIC_DISPATCH(ForwardRotateImpl)(data, signs, n);
}

void Householder3InPlace(float* data, size_t n_block) {
  HWY_DYNAMIC_DISPATCH(Householder3InPlaceImpl)(data, n_block);
}

}  // namespace turboquant
#endif  // HWY_ONCE
