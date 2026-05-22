#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kernels.h"
#include "packing.h"

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
  // Scalar loop: autovectorizes well and avoids fragile float->u16 SIMD paths.
  for (size_t i = 0; i < n; ++i) {
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
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = scale * (static_cast<int>(codes[i]) - zero_point);
  }
}

void DequantizeBinaryImpl(const uint16_t* codes, size_t n, float scale,
                          float* data_out) {
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = scale * (2.0f * static_cast<float>(codes[i]) - 1.0f);
  }
}

// -- ADC inner loops --------------------------------------------------------

// 8-bit fast path: direct byte read, sign-correct relative to zero_point=128.
AdcUnscaled AdcScore8(const uint8_t* packed, size_t n, const float* q) {
  const hn::ScalableTag<float> df;
  const size_t lanes = hn::Lanes(df);
  auto vdot = hn::Zero(df);
  auto vnrm = hn::Zero(df);
  const auto v128 = hn::Set(df, 128.0f);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    HWY_ALIGN float buf[hn::MaxLanes(df)];
    for (size_t k = 0; k < lanes; ++k) {
      buf[k] = static_cast<float>(packed[i + k]);
    }
    auto vc = hn::Load(df, buf);
    auto vlvl = hn::Sub(vc, v128);
    auto vq = hn::LoadU(df, q + i);
    vdot = hn::MulAdd(vq, vlvl, vdot);
    vnrm = hn::MulAdd(vlvl, vlvl, vnrm);
  }
  float dot = hn::ReduceSum(df, vdot);
  float nrm = hn::ReduceSum(df, vnrm);
  for (; i < n; ++i) {
    const float lvl = static_cast<float>(packed[i]) - 128.0f;
    dot += q[i] * lvl;
    nrm += lvl * lvl;
  }
  return {dot, nrm};
}

// 1-bit fast path: each level is +/-1.
AdcUnscaled AdcScore1(const uint8_t* packed, size_t n, const float* q) {
  // sum_q_pos minus sum_q_neg = sum q*level, and norm2 = n (each level squared is 1).
  // Vectorize by gathering signs from the bitmap.
  const hn::ScalableTag<float> df;
  const size_t lanes = hn::Lanes(df);
  auto vacc = hn::Zero(df);
  const auto vpos = hn::Set(df, 1.0f);
  const auto vneg = hn::Set(df, -1.0f);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    HWY_ALIGN float lvls[hn::MaxLanes(df)];
    for (size_t k = 0; k < lanes; ++k) {
      const size_t idx = i + k;
      const uint8_t b = packed[idx >> 3];
      lvls[k] = (b >> (idx & 7)) & 1u ? 1.0f : -1.0f;
    }
    (void)vpos;
    (void)vneg;
    auto vl = hn::Load(df, lvls);
    auto vq = hn::LoadU(df, q + i);
    vacc = hn::MulAdd(vq, vl, vacc);
  }
  float dot = hn::ReduceSum(df, vacc);
  for (; i < n; ++i) {
    const uint8_t b = packed[i >> 3];
    const float lvl = (b >> (i & 7)) & 1u ? 1.0f : -1.0f;
    dot += q[i] * lvl;
  }
  return {dot, static_cast<float>(n)};
}

// 4-bit fast path: 2 codes per byte, zero_point=8.
AdcUnscaled AdcScore4(const uint8_t* packed, size_t n, const float* q) {
  const hn::ScalableTag<float> df;
  const size_t lanes = hn::Lanes(df);
  auto vdot = hn::Zero(df);
  auto vnrm = hn::Zero(df);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    HWY_ALIGN float lvls[hn::MaxLanes(df)];
    for (size_t k = 0; k < lanes; ++k) {
      const size_t idx = i + k;
      const uint8_t b = packed[idx >> 1];
      const uint8_t nibble = (idx & 1) ? (b >> 4) : (b & 0xF);
      lvls[k] = static_cast<float>(static_cast<int>(nibble) - 8);
    }
    auto vl = hn::Load(df, lvls);
    auto vq = hn::LoadU(df, q + i);
    vdot = hn::MulAdd(vq, vl, vdot);
    vnrm = hn::MulAdd(vl, vl, vnrm);
  }
  float dot = hn::ReduceSum(df, vdot);
  float nrm = hn::ReduceSum(df, vnrm);
  for (; i < n; ++i) {
    const uint8_t b = packed[i >> 1];
    const uint8_t nibble = (i & 1) ? (b >> 4) : (b & 0xF);
    const float lvl = static_cast<float>(static_cast<int>(nibble) - 8);
    dot += q[i] * lvl;
    nrm += lvl * lvl;
  }
  return {dot, nrm};
}

// Generic path: unpack via bitstream helpers and accumulate.
AdcUnscaled AdcScoreGeneric(const uint8_t* packed, size_t n, QuantBits bits,
                            const float* q) {
  // Unpack into a stack/heap buffer of int16 codes.
  // Cap stack use; spill to heap for large n.
  constexpr size_t kStack = 4096;
  uint16_t stack_codes[kStack];
  uint16_t* codes = stack_codes;
  std::unique_ptr<uint16_t[]> heap;
  if (n > kStack) {
    heap.reset(new uint16_t[n]);
    codes = heap.get();
  }
  UnpackCodes(packed, n, bits, codes);

  const int zp = 1 << (static_cast<int>(bits) - 1);
  const hn::ScalableTag<float> df;
  const size_t lanes = hn::Lanes(df);
  const auto vzp = hn::Set(df, static_cast<float>(zp));
  auto vdot = hn::Zero(df);
  auto vnrm = hn::Zero(df);
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    HWY_ALIGN float buf[hn::MaxLanes(df)];
    for (size_t k = 0; k < lanes; ++k) buf[k] = static_cast<float>(codes[i + k]);
    auto vc = hn::Load(df, buf);
    auto vl = hn::Sub(vc, vzp);
    auto vq = hn::LoadU(df, q + i);
    vdot = hn::MulAdd(vq, vl, vdot);
    vnrm = hn::MulAdd(vl, vl, vnrm);
  }
  float dot = hn::ReduceSum(df, vdot);
  float nrm = hn::ReduceSum(df, vnrm);
  for (; i < n; ++i) {
    const float lvl = static_cast<float>(static_cast<int>(codes[i]) - zp);
    dot += q[i] * lvl;
    nrm += lvl * lvl;
  }
  return {dot, nrm};
}

AdcUnscaled AdcUnscaledScoreImpl(const uint8_t* packed_codes, size_t padded_dim,
                                 QuantBits bits, const float* q_rot) {
  switch (bits) {
    case QuantBits::B8:
      return AdcScore8(packed_codes, padded_dim, q_rot);
    case QuantBits::B4:
      return AdcScore4(packed_codes, padded_dim, q_rot);
    case QuantBits::B1:
      return AdcScore1(packed_codes, padded_dim, q_rot);
    case QuantBits::B2:
    case QuantBits::B6:
    case QuantBits::B12:
      return AdcScoreGeneric(packed_codes, padded_dim, bits, q_rot);
  }
  return {0.0f, 0.0f};
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
HWY_EXPORT(AdcUnscaledScoreImpl);

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
AdcUnscaled AdcUnscaledScore(const uint8_t* packed_codes, size_t padded_dim,
                             QuantBits bits, const float* q_rot) {
  return HWY_DYNAMIC_DISPATCH(AdcUnscaledScoreImpl)(packed_codes, padded_dim,
                                                    bits, q_rot);
}

}  // namespace turboquant
#endif  // HWY_ONCE
