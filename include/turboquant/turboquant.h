#ifndef TURBOQUANT_TURBOQUANT_H_
#define TURBOQUANT_TURBOQUANT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace turboquant {

enum class QuantBits : uint8_t {
  B1 = 1,
  B2 = 2,
  B4 = 4,
  B6 = 6,
  B8 = 8,
  B12 = 12,
};

// Smallest power-of-two >= dim (and >= 1).
size_t PaddedDim(size_t dim);

// Bytes of packed codes for `padded_dim` codes at the given bit width.
size_t PackedBytes(size_t padded_dim, QuantBits bits);

// Total per-vector payload size: 4-byte scale (LE float32) + N packed-code bytes.
size_t PayloadSize(size_t dim, QuantBits bits);

// TurboQuant rotation: x -> H * D * pad(x), where:
//   - pad zero-extends to next_pow2(dim);
//   - D is a deterministic diagonal of random {-1,+1} entries seeded by `seed`;
//   - H is the normalized Walsh-Hadamard transform (orthogonal, ||Hv|| = ||v||).
class Rotator {
 public:
  Rotator(size_t dim, uint64_t seed);

  size_t dim() const { return dim_; }
  size_t padded_dim() const { return padded_dim_; }
  const float* signs() const { return signs_.data(); }

  // Apply H * D * pad(x). `x` has length dim(); `out` has length padded_dim().
  void Apply(const float* x, float* out) const;

  // Apply the inverse rotation D * H * y_padded. `y_padded` has length padded_dim()
  // and is overwritten (acts as scratch). `out_dim` receives the leading dim()
  // entries of D * H * y_padded.
  void ApplyInverse(float* y_padded, float* out_dim) const;

 private:
  size_t dim_;
  size_t padded_dim_;
  std::vector<float> signs_;  // length padded_dim_, entries are +1 / -1
};

// Encode `x` (length rot.dim()) into `payload_out` (PayloadSize(rot.dim(), bits) bytes).
// Internally rotates x with `rot` then quantizes to `bits`.
void Quantize(const Rotator& rot, QuantBits bits, const float* x,
              uint8_t* payload_out);

// Decode `payload` back to the original-dimension vector `x_out` (length
// rot.dim()). Applies the inverse rotation using:
//   - an unrolled u64-load bitstream unpack for B6/B12;
//   - a Walsh-Hadamard kernel that fuses the small-stride stages in-register;
//   - a single SIMD pass that combines the post-WHT sign flip and the
//     1/sqrt(padded_dim) normalization.
void Dequantize(const Rotator& rot, QuantBits bits, const uint8_t* payload,
                float* x_out);

}  // namespace turboquant

#endif  // TURBOQUANT_TURBOQUANT_H_
