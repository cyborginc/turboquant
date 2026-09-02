// clang-format off
// foreach_target.h must precede highway.h and both must lead the file: this
// translation unit re-includes itself once per SIMD target. Do not reorder.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
// clang-format on

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

void QuantizeAffineImpl(const float* data, size_t n, float scale,
                        int zero_point, int max_code, uint16_t* codes_out) {
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

// Branch-free symmetric binary-search encode. For a Beta codebook of
// 2^Bits levels (symmetric around 0), we run a binary search on the
// positive-half boundaries (length 2^(Bits-1), padded with +inf), then
// fold the sign back in. Per-coord work: Bits-1 dependent compares.
//
// pos_bounds_pad is L1-resident (≤ 2 KB at Bits=12). The compiler unrolls
// the template loop fully because the step values are constexpr.
template <int Bits>
HWY_INLINE int EncodeBetaSymBranchFree(float x, const float* pos_bounds_pad) {
  static_assert(Bits >= 1 && Bits <= 12, "Beta encode supports b1..b12");
  constexpr int kHalf = 1 << (Bits - 1);
  const float abs_x = std::fabs(x);
  int sub_code = 0;
  // Fixed-depth binary search. Tree probe at position (sub_code + step - 1).
  for (int step = kHalf >> 1; step > 0; step >>= 1) {
    const float bnd = pos_bounds_pad[sub_code + step - 1];
    sub_code += (abs_x > bnd) ? step : 0;
  }
  // sub_code is in [0, kHalf). Fold sign back: positive x maps to upper half,
  // negative x mirrors into the lower half.
  return (x >= 0.0f) ? (kHalf + sub_code) : (kHalf - 1 - sub_code);
}

template <int Bits>
void EncodeBetaCodebookN(const float* values, size_t n,
                         const float* pos_bounds_pad, uint16_t* codes_out) {
  // Scalar per coord. The compiler pipelines many coords' independent binary
  // searches through OoO scheduling; SIMD-batched binary search would require
  // gather (scarce on NEON) so we leave that on the table.
  for (size_t i = 0; i < n; ++i) {
    codes_out[i] = static_cast<uint16_t>(
        EncodeBetaSymBranchFree<Bits>(values[i], pos_bounds_pad));
  }
}

void EncodeBetaCodebookImpl(const float* values, size_t n,
                            const float* pos_bounds_pad, QuantBits bits,
                            uint16_t* codes_out) {
  switch (bits) {
    case QuantBits::B1:
      EncodeBetaCodebookN<1>(values, n, pos_bounds_pad, codes_out);
      return;
    case QuantBits::B2:
      EncodeBetaCodebookN<2>(values, n, pos_bounds_pad, codes_out);
      return;
    case QuantBits::B3:
      EncodeBetaCodebookN<3>(values, n, pos_bounds_pad, codes_out);
      return;
    case QuantBits::B4:
      EncodeBetaCodebookN<4>(values, n, pos_bounds_pad, codes_out);
      return;
    case QuantBits::B6:
      EncodeBetaCodebookN<6>(values, n, pos_bounds_pad, codes_out);
      return;
    case QuantBits::B8:
      EncodeBetaCodebookN<8>(values, n, pos_bounds_pad, codes_out);
      return;
    case QuantBits::B12:
      EncodeBetaCodebookN<12>(values, n, pos_bounds_pad, codes_out);
      return;
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
void EncodeBetaCodebook(const float* values, size_t n,
                        const float* pos_bounds_pad, QuantBits bits,
                        uint16_t* codes_out) {
  HWY_DYNAMIC_DISPATCH(EncodeBetaCodebookImpl)
  (values, n, pos_bounds_pad, bits, codes_out);
}
void DecodeBetaCodebook(const uint16_t* codes, size_t n, const float* centroids,
                        float scale, float* data_out) {
  HWY_DYNAMIC_DISPATCH(DecodeBetaCodebookImpl)
  (codes, n, centroids, scale, data_out);
}
// Deliberately scalar and deliberately not dynamically dispatched.
//
// This value becomes the 4-byte scale header of every Beta-path payload, so it
// is part of a persisted format. A SIMD reduction accumulates into one partial
// sum per lane and folds them at the end, which means the summation order — and
// therefore the last-ULP result — depends on the vector width chosen at
// runtime. Measured on an AMD EPYC 9V74: SSE2, SSSE3, SSE4, AVX2 and AVX3_ZEN4
// each produced a different scale, so the same vector encoded to different
// bytes depending on which host wrote it.
//
// Accumulating in double in index order is width-independent by construction
// and gives the same answer on every target and compiler. The affine path is
// unaffected because its scale comes from MaxAbs, and max is associative.
//
// Cost is one scalar pass over n floats on the encode path only, alongside the
// scalar double norm loop the callers already run; decode never calls this.
float CentroidInnerProduct(const float* values, const uint16_t* codes, size_t n,
                           const float* centroids) {
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(values[i]) *
           static_cast<double>(centroids[codes[i]]);
  }
  return static_cast<float>(acc);
}

}  // namespace turboquant
#endif  // HWY_ONCE
