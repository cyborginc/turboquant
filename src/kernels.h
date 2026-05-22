#ifndef TURBOQUANT_SRC_KERNELS_H_
#define TURBOQUANT_SRC_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "turboquant/turboquant.h"

namespace turboquant {

// Max absolute value over a contiguous float array.
float MaxAbs(const float* data, size_t n);

// codes_out[i] = clamp(round(data[i]/scale), -zp, zp-1) + zp, for non-1-bit modes.
// `zero_point = 1 << (bits-1)`, `max_code = (1 << bits) - 1`.
void QuantizeAffine(const float* data, size_t n, float scale, int zero_point,
                    int max_code, uint16_t* codes_out);

// 1-bit symmetric: code = (data[i] >= 0) ? 1 : 0.
void QuantizeBinary(const float* data, size_t n, uint16_t* codes_out);

// data_out[i] = scale * (codes[i] - zero_point), for non-1-bit modes.
void DequantizeAffine(const uint16_t* codes, size_t n, float scale,
                      int zero_point, float* data_out);

// 1-bit symmetric: data_out[i] = scale * (2*codes[i] - 1).
void DequantizeBinary(const uint16_t* codes, size_t n, float scale,
                      float* data_out);

// Unpacks the payload's packed codes and computes (sum q*level, sum level*level)
// without the per-vector scale. Caller multiplies by scale/scale^2.
//
// `bits` selects the unpack/level mapping. `padded_dim` is the count of codes.
// `q_rot` is the rotated query of length `padded_dim`.
struct AdcUnscaled {
  float dot;
  float norm2;
};
AdcUnscaled AdcUnscaledScore(const uint8_t* packed_codes, size_t padded_dim,
                             QuantBits bits, const float* q_rot);

}  // namespace turboquant

#endif  // TURBOQUANT_SRC_KERNELS_H_
