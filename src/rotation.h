#ifndef TURBOQUANT_SRC_ROTATION_H_
#define TURBOQUANT_SRC_ROTATION_H_

#include <cstddef>

namespace turboquant {

// In-place normalized Walsh-Hadamard transform on `data` of length `n`,
// where n is a power of two. After the call, ||data||_2 is unchanged.
void HadamardTransform(float* data, size_t n);

// Multiply elementwise by `signs` (entries are +1 or -1).
void ApplySigns(float* data, const float* signs, size_t n);

}  // namespace turboquant

#endif  // TURBOQUANT_SRC_ROTATION_H_
