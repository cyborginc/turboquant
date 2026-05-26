#ifndef TURBOQUANT_SRC_INTERNAL_H_
#define TURBOQUANT_SRC_INTERNAL_H_

// Test / benchmark hooks. Not part of the shipped public API.
//
// The public Quantize/Dequantize auto-route between an affine min/max scheme
// and a Lloyd-Max Beta-codebook scheme based on bit width. These helpers let
// the bench and unit tests exercise either path explicitly so we can verify
// the routing remains optimal and detect regressions.

#include <cstddef>
#include <cstdint>

#include "turboquant/turboquant.h"

namespace turboquant {
namespace internal {

// Affine min/max path: scale = max_abs/max_pos, codes uniformly spaced.
void QuantizeAffine(const Rotator& rot, QuantBits bits, const float* x,
                    uint8_t* payload_out);
void DequantizeAffine(const Rotator& rot, QuantBits bits,
                      const uint8_t* payload, float* x_out);

// Lloyd-Max Beta-codebook path: codebook tuned to the Beta((dim-1)/2,
// (dim-1)/2) distribution of rotated unit-vector coordinates.
void QuantizeBeta(const Rotator& rot, QuantBits bits, const float* x,
                  uint8_t* payload_out);
void DequantizeBeta(const Rotator& rot, QuantBits bits, const uint8_t* payload,
                    float* x_out);

}  // namespace internal
}  // namespace turboquant

#include "rotator_mixed.h"  // RotatorMixed3 + QuantizeMixed3 / DequantizeMixed3.

#endif  // TURBOQUANT_SRC_INTERNAL_H_
