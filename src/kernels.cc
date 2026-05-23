#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kernels.h"

HWY_BEFORE_NAMESPACE();
namespace turboquant {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

float MaxAbsImpl(const float* data, size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  auto vmax = hn::Zero(d);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto v = hn::Abs(hn::LoadU(d, data + i));
    vmax = hn::Max(vmax, v);
  }
  float m = hn::GetLane(hn::MaxOfLanes(d, vmax));
  for (; i < n; ++i) {
    const float a = std::fabs(data[i]);
    if (a > m) m = a;
  }
  return m;
}

void QuantizeAffineImpl(const float* data, size_t n, float scale, int zero_point,
                        int max_code, uint16_t* codes_out) {
  const float inv_scale = 1.0f / scale;
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<uint16_t, decltype(df)> du16;
  const size_t lanes = hn::Lanes(df);
  const auto v_invs = hn::Set(df, inv_scale);
  const auto v_zp = hn::Set(di32, zero_point);
  const auto v_lo = hn::Zero(di32);
  const auto v_hi = hn::Set(di32, max_code);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto v = hn::LoadU(df, data + i);
    auto v_scaled = hn::Mul(v, v_invs);
    auto v_int = hn::ConvertTo(di32, hn::Round(v_scaled));
    auto v_code = hn::Add(v_int, v_zp);
    v_code = hn::Min(hn::Max(v_code, v_lo), v_hi);
    auto v_u16 = hn::DemoteTo(du16, v_code);
    hn::StoreU(v_u16, du16, codes_out + i);
  }
  for (; i < n; ++i) {
    int level = static_cast<int>(std::lrintf(data[i] * inv_scale));
    int code = level + zero_point;
    if (code < 0) code = 0;
    if (code > max_code) code = max_code;
    codes_out[i] = static_cast<uint16_t>(code);
  }
}

void QuantizeBinaryImpl(const float* data, size_t n, uint16_t* codes_out) {
  for (size_t i = 0; i < n; ++i) {
    codes_out[i] = (data[i] >= 0.0f) ? 1 : 0;
  }
}

void DequantizeAffineImpl(const uint16_t* codes, size_t n, float scale,
                          int zero_point, float* data_out) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<uint16_t, decltype(df)> du16;
  const size_t lanes = hn::Lanes(df);
  const auto v_scale = hn::Set(df, scale);
  const auto v_zp = hn::Set(di32, zero_point);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto vu16 = hn::LoadU(du16, codes + i);
    auto vi32 = hn::Sub(hn::PromoteTo(di32, vu16), v_zp);
    auto vf = hn::Mul(hn::ConvertTo(df, vi32), v_scale);
    hn::StoreU(vf, df, data_out + i);
  }
  for (; i < n; ++i) {
    data_out[i] = scale * (static_cast<int>(codes[i]) - zero_point);
  }
}

void DequantizeBinaryImpl(const uint16_t* codes, size_t n, float scale,
                          float* data_out) {
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = scale * (2.0f * static_cast<float>(codes[i]) - 1.0f);
  }
}

// Encode by counting how many boundaries each value exceeds. Outer loop over
// boundaries keeps the per-lane code accumulator register-resident; inner loop
// is SIMD over the n input values.
void EncodeBetaCodebookImpl(const float* values, size_t n,
                            const float* boundaries, size_t num_boundaries,
                            uint16_t* codes_out) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<uint16_t, decltype(df)> du16;
  const size_t lanes = hn::Lanes(df);
  const auto v_one = hn::Set(di32, 1);
  const auto v_zero_i32 = hn::Zero(di32);

  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    auto v_codes = hn::Zero(di32);
    const auto vx = hn::LoadU(df, values + i);
    for (size_t k = 0; k < num_boundaries; ++k) {
      const auto vb = hn::Set(df, boundaries[k]);
      const auto m32 = hn::RebindMask(di32, hn::Gt(vx, vb));
      v_codes = hn::Add(v_codes, hn::IfThenElse(m32, v_one, v_zero_i32));
    }
    hn::StoreU(hn::DemoteTo(du16, v_codes), du16, codes_out + i);
  }
  for (; i < n; ++i) {
    int c = 0;
    for (size_t k = 0; k < num_boundaries; ++k) {
      if (values[i] > boundaries[k]) ++c;
    }
    codes_out[i] = static_cast<uint16_t>(c);
  }
}

void DecodeBetaCodebookImpl(const uint16_t* codes, size_t n,
                            const float* centroids, float scale,
                            float* data_out) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<uint16_t, decltype(df)> du16;
  const size_t lanes = hn::Lanes(df);
  const auto v_scale = hn::Set(df, scale);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto vu16 = hn::LoadU(du16, codes + i);
    const auto vi32 = hn::PromoteTo(di32, vu16);
    const auto v = hn::GatherIndex(df, centroids, vi32);
    hn::StoreU(hn::Mul(v, v_scale), df, data_out + i);
  }
  for (; i < n; ++i) {
    data_out[i] = scale * centroids[codes[i]];
  }
}

float CentroidInnerProductImpl(const float* values, const uint16_t* codes,
                               size_t n, const float* centroids) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<uint16_t, decltype(df)> du16;
  const size_t lanes = hn::Lanes(df);
  auto vacc = hn::Zero(df);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto vu16 = hn::LoadU(du16, codes + i);
    const auto vi32 = hn::PromoteTo(di32, vu16);
    const auto vc = hn::GatherIndex(df, centroids, vi32);
    const auto vv = hn::LoadU(df, values + i);
    vacc = hn::MulAdd(vv, vc, vacc);
  }
  float acc = hn::ReduceSum(df, vacc);
  for (; i < n; ++i) {
    acc += values[i] * centroids[codes[i]];
  }
  return acc;
}

}  // namespace HWY_NAMESPACE
}  // namespace turboquant
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace turboquant {

HWY_EXPORT(MaxAbsImpl);
HWY_EXPORT(QuantizeAffineImpl);
HWY_EXPORT(QuantizeBinaryImpl);
HWY_EXPORT(DequantizeAffineImpl);
HWY_EXPORT(DequantizeBinaryImpl);
HWY_EXPORT(EncodeBetaCodebookImpl);
HWY_EXPORT(DecodeBetaCodebookImpl);
HWY_EXPORT(CentroidInnerProductImpl);

float MaxAbs(const float* data, size_t n) {
  return HWY_DYNAMIC_DISPATCH(MaxAbsImpl)(data, n);
}
void QuantizeAffine(const float* data, size_t n, float scale, int zero_point,
                    int max_code, uint16_t* codes_out) {
  HWY_DYNAMIC_DISPATCH(QuantizeAffineImpl)
  (data, n, scale, zero_point, max_code, codes_out);
}
void QuantizeBinary(const float* data, size_t n, uint16_t* codes_out) {
  HWY_DYNAMIC_DISPATCH(QuantizeBinaryImpl)(data, n, codes_out);
}
void DequantizeAffine(const uint16_t* codes, size_t n, float scale,
                      int zero_point, float* data_out) {
  HWY_DYNAMIC_DISPATCH(DequantizeAffineImpl)
  (codes, n, scale, zero_point, data_out);
}
void DequantizeBinary(const uint16_t* codes, size_t n, float scale,
                      float* data_out) {
  HWY_DYNAMIC_DISPATCH(DequantizeBinaryImpl)(codes, n, scale, data_out);
}
void EncodeBetaCodebook(const float* values, size_t n, const float* boundaries,
                        size_t num_boundaries, uint16_t* codes_out) {
  HWY_DYNAMIC_DISPATCH(EncodeBetaCodebookImpl)(values, n, boundaries,
                                               num_boundaries, codes_out);
}
void DecodeBetaCodebook(const uint16_t* codes, size_t n, const float* centroids,
                        float scale, float* data_out) {
  HWY_DYNAMIC_DISPATCH(DecodeBetaCodebookImpl)(codes, n, centroids, scale,
                                               data_out);
}
float CentroidInnerProduct(const float* values, const uint16_t* codes, size_t n,
                           const float* centroids) {
  return HWY_DYNAMIC_DISPATCH(CentroidInnerProductImpl)(values, codes, n,
                                                        centroids);
}

}  // namespace turboquant
#endif  // HWY_ONCE
