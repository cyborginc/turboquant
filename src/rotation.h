#ifndef TURBOQUANT_SRC_ROTATION_H_
#define TURBOQUANT_SRC_ROTATION_H_

#include <cstddef>

namespace turboquant {

// In-place normalized Walsh-Hadamard transform on `data` of length `n`,
// where n is a power of two. After the call, ||data||_2 is unchanged.
void HadamardTransform(float* data, size_t n);

// Multiply elementwise by `signs` (entries are +1 or -1).
void ApplySigns(float* data, const float* signs, size_t n);

// Same WHT as HadamardTransform but with the small-stride stages (h=1 always;
// h=2 additionally on lane-4 targets) fused in-register via SIMD shuffles.
// Does NOT apply the 1/sqrt(n) normalization — callers that pair this with
// ApplySignsAndScale (which does both the sign flip and the normalization in
// one pass) avoid two extra passes over `data`.
void FastHadamardTransformUnscaled(float* data, size_t n);

// data[i] = scale * data[i] * signs[i], single SIMD pass.
void ApplySignsAndScale(float* data, const float* signs, size_t n, float scale);

}  // namespace turboquant

#endif  // TURBOQUANT_SRC_ROTATION_H_
