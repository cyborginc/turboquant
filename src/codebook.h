#ifndef TURBOQUANT_SRC_CODEBOOK_H_
#define TURBOQUANT_SRC_CODEBOOK_H_

#include <cstddef>
#include <vector>

#include "turboquant/turboquant.h"

namespace turboquant {

// Lloyd-Max scalar quantizer for Beta((n-1)/2, (n-1)/2) on [-1, 1].
//
// After unit-normalizing a vector and applying an orthogonal rotation, each
// coordinate of the rotated vector follows this distribution exactly. The
// resulting codebook places quantization centroids at the conditional means
// of the distribution's quantiles — information-theoretically optimal for
// scalar quantization at the given bit width.
//
// Construction is one-time and depends only on (bits, padded_dim). Cache it.
class BetaCodebook {
 public:
  BetaCodebook(QuantBits bits, size_t padded_dim);

  size_t num_levels() const { return centroids_.size(); }
  // Sorted ascending; length num_levels - 1.
  const float* boundaries() const { return boundaries_.data(); }
  size_t num_boundaries() const { return boundaries_.size(); }
  // Sorted ascending; length num_levels.
  const float* centroids() const { return centroids_.data(); }

 private:
  std::vector<float> boundaries_;
  std::vector<float> centroids_;
};

}  // namespace turboquant

#endif  // TURBOQUANT_SRC_CODEBOOK_H_
