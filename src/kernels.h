#ifndef TURBOQUANT_SRC_KERNELS_H_
#define TURBOQUANT_SRC_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "turboquant/turboquant.h"

namespace turboquant {

// Max absolute value over a contiguous float array.
float MaxAbs(const float* data, size_t n);

// codes_out[i] = clamp(round(data[i]/scale), -zp, zp-1) + zp, for non-1-bit
// modes. `zero_point = 1 << (bits-1)`, `max_code = (1 << bits) - 1`.
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

// Beta-codebook encode. For each input value, code = |{k : value >
// boundaries[k]}|. Uses a templated branch-free binary search over the
// positive-half boundaries (the codebook is symmetric around zero), reducing
// the per-coord work from O(2^bits) compares to O(bits).
//
// `pos_bounds_pad` is the codebook's positive_boundaries_padded() (length
// 2^(bits-1), padded with +inf at the tail). `bits` selects the templated
// search depth at runtime.
void EncodeBetaCodebook(const float* values, size_t n,
                        const float* pos_bounds_pad, QuantBits bits,
                        uint16_t* codes_out);

// data_out[i] = scale * centroids[codes[i]], single SIMD pass with table
// gather.
void DecodeBetaCodebook(const uint16_t* codes, size_t n, const float* centroids,
                        float scale, float* data_out);

// inner_product(values, centroids[codes[*]]) — single pass over n.
float CentroidInnerProduct(const float* values, const uint16_t* codes, size_t n,
                           const float* centroids);

}  // namespace turboquant

#endif  // TURBOQUANT_SRC_KERNELS_H_
