#ifndef TURBOQUANT_SRC_INTERNAL_H_
#define TURBOQUANT_SRC_INTERNAL_H_

// Test / benchmark hooks. Not part of the shipped public API.
//
// The public Quantizer auto-routes between four code paths (padded-affine,
// padded-beta, mixed3-affine, mixed3-beta) based on the (dim, bits) it was
// constructed with. These helpers let the bench and unit tests exercise
// each path explicitly so we can verify the routing remains optimal and
// detect regressions.

#include <cstddef>
#include <cstdint>

#include "rotator_mixed.h"  // RotatorMixed3 + QuantizeMixed3 / DequantizeMixed3 / Affine variants.
#include "rotator_padded.h"  // RotatorPadded
#include "turboquant/turboquant.h"

namespace turboquant {
namespace internal {

// Affine min/max path against a padded WHT rotation.
void QuantizeAffine(const RotatorPadded& rot, QuantBits bits, const float* x,
                    uint8_t* payload_out);
void DequantizeAffine(const RotatorPadded& rot, QuantBits bits,
                      const uint8_t* payload, float* x_out);

// Lloyd-Max Beta-codebook path against a padded WHT rotation.
void QuantizeBeta(const RotatorPadded& rot, QuantBits bits, const float* x,
                  uint8_t* payload_out);
void DequantizeBeta(const RotatorPadded& rot, QuantBits bits,
                    const uint8_t* payload, float* x_out);

// Payload size for the padded path. Auto-routing PayloadBytes does NOT
// use this; bench/tests do.
size_t PayloadSizePadded(size_t dim, QuantBits bits);

}  // namespace internal
}  // namespace turboquant

#endif  // TURBOQUANT_SRC_INTERNAL_H_
