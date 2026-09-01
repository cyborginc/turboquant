#ifndef TURBOQUANT_SRC_ROTATOR_PADDED_H_
#define TURBOQUANT_SRC_ROTATOR_PADDED_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "codebook.h"
#include "turboquant/turboquant.h"

namespace turboquant {

namespace internal {

// Walsh-Hadamard rotation padded to next_pow2(dim). Each rotated unit vector
// lives in R^padded_dim and follows Beta((padded_dim - 1)/2, ...).
class RotatorPadded {
 public:
  RotatorPadded(size_t dim, uint64_t seed);
  ~RotatorPadded();
  RotatorPadded(RotatorPadded&&) noexcept;
  RotatorPadded& operator=(RotatorPadded&&) noexcept;
  RotatorPadded(const RotatorPadded&) = delete;
  RotatorPadded& operator=(const RotatorPadded&) = delete;

  size_t dim() const { return dim_; }
  size_t padded_dim() const { return padded_dim_; }
  const float* signs() const { return signs_.data(); }
  const BetaCodebook* beta_codebook(QuantBits bits) const;

  // Apply H * D * pad(x). `x` has length dim(); `out` has length padded_dim().
  void Apply(const float* x, float* out) const;
  // Apply the inverse rotation D * H * y_padded. `y_padded` (length padded_dim)
  // is overwritten; `out_dim` receives the first dim() entries.
  void ApplyInverse(float* y_padded, float* out_dim) const;

 private:
  size_t dim_;
  size_t padded_dim_;
  std::vector<float> signs_;
  BetaCodebookCache beta_codebooks_;
};

}  // namespace internal
}  // namespace turboquant

#endif  // TURBOQUANT_SRC_ROTATOR_PADDED_H_
