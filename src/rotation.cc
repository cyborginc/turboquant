#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rotation.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cmath>
#include <cstddef>

#include "rotation.h"

HWY_BEFORE_NAMESPACE();
namespace turboquant {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Butterfly stride `h` on contiguous chunks of size 2*h: pair (a, b) -> (a+b, a-b).
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

}  // namespace HWY_NAMESPACE
}  // namespace turboquant
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace turboquant {

HWY_EXPORT(HadamardTransformImpl);
HWY_EXPORT(ApplySignsImpl);

void HadamardTransform(float* data, size_t n) {
  HWY_DYNAMIC_DISPATCH(HadamardTransformImpl)(data, n);
}

void ApplySigns(float* data, const float* signs, size_t n) {
  HWY_DYNAMIC_DISPATCH(ApplySignsImpl)(data, signs, n);
}

}  // namespace turboquant
#endif  // HWY_ONCE
