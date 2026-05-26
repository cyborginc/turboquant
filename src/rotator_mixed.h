#ifndef TURBOQUANT_SRC_ROTATOR_MIXED_H_
#define TURBOQUANT_SRC_ROTATOR_MIXED_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "turboquant/turboquant.h"

namespace turboquant {

class BetaCodebook;  // src/codebook.h

namespace internal {

// Mixed-radix orthogonal rotation for dim = 3 * 2^k. Built so we can encode
// without zero-padding to the next power of two, saving 33% storage at common
// embedding dims (768, 1536, 3072).
//
// Forward:  y = (1/sqrt(N)) * H_3 * WHT_N(per row) * D * x
// where N = dim/3, D is a random ±1 diagonal, WHT_N is applied to each of the
// 3 length-N rows of the reshaped 3 × N input, and H_3 is the Householder
// reflection I - (2/3) * 1 * 1^T over the 3 row-axis.
//
// The rotation is exactly orthogonal on R^dim (no padding), so the rotated
// unit vector follows Beta((dim-1)/2, (dim-1)/2) on [-1, 1] and the Lloyd-Max
// codebook is computed against that distribution directly.
class RotatorMixed3 {
 public:
  RotatorMixed3(size_t dim, uint64_t seed);
  ~RotatorMixed3();
  RotatorMixed3(RotatorMixed3&&) noexcept;
  RotatorMixed3& operator=(RotatorMixed3&&) noexcept;
  RotatorMixed3(const RotatorMixed3&) = delete;
  RotatorMixed3& operator=(const RotatorMixed3&) = delete;

  static bool DimSupported(size_t dim);  // dim = 3 * 2^k for some k >= 0.

  size_t dim() const { return dim_; }
  const float* signs() const { return signs_.data(); }
  // Codebook computed for the true (unpadded) dim. nullptr for unsupported bits.
  const BetaCodebook* beta_codebook(QuantBits bits) const;

  // Forward rotation: out[i] for i in [0, dim).
  void Apply(const float* x, float* out) const;
  // Inverse rotation. y_padded is overwritten (scratch); out_dim receives the
  // first dim entries (which are all entries — no padding).
  void ApplyInverse(float* y, float* out) const;

 private:
  size_t dim_;
  size_t n_block_;  // dim_ / 3
  std::vector<float> signs_;
  std::vector<std::unique_ptr<BetaCodebook>> beta_codebooks_;
};

// Payload layout: 4-byte scale + dim*bits/8 packed-code bytes. Note this is
// strictly smaller than PayloadSize(dim, bits) when dim is not a power of 2.
size_t PayloadSizeMixed3(size_t dim, QuantBits bits);

// Mirror of the public Quantize/Dequantize but using the mixed-radix rotation
// and dim-sized (not padded_dim) packed codes.
//
// "Beta" uses the Lloyd-Max codebook tuned to Beta((dim-1)/2, (dim-1)/2);
// "Affine" uses uniform-step quantization with scale = max_abs/max_pos and is
// what we'd want at b8/b12 where the codebook advantage vanishes.
void QuantizeMixed3(const RotatorMixed3& rot, QuantBits bits, const float* x,
                    uint8_t* payload_out);
void DequantizeMixed3(const RotatorMixed3& rot, QuantBits bits,
                      const uint8_t* payload, float* x_out);

void QuantizeMixed3Affine(const RotatorMixed3& rot, QuantBits bits,
                          const float* x, uint8_t* payload_out);
void DequantizeMixed3Affine(const RotatorMixed3& rot, QuantBits bits,
                            const uint8_t* payload, float* x_out);

}  // namespace internal
}  // namespace turboquant

#endif  // TURBOQUANT_SRC_ROTATOR_MIXED_H_
