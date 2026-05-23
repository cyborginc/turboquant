#ifndef TURBOQUANT_TURBOQUANT_H_
#define TURBOQUANT_TURBOQUANT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace turboquant {

class BetaCodebook;  // Private; defined in src/codebook.h.

enum class QuantBits : uint8_t {
  B1 = 1,
  B2 = 2,
  B3 = 3,
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
  ~Rotator();  // Out-of-line: BetaCodebook is incomplete here.
  Rotator(Rotator&&) noexcept;
  Rotator& operator=(Rotator&&) noexcept;
  Rotator(const Rotator&) = delete;
  Rotator& operator=(const Rotator&) = delete;

  size_t dim() const { return dim_; }
  size_t padded_dim() const { return padded_dim_; }
  const float* signs() const { return signs_.data(); }

  // Apply H * D * pad(x). `x` has length dim(); `out` has length padded_dim().
  void Apply(const float* x, float* out) const;

  // Apply the inverse rotation D * H * y_padded. `y_padded` has length padded_dim()
  // and is overwritten (acts as scratch). `out_dim` receives the leading dim()
  // entries of D * H * y_padded.
  void ApplyInverse(float* y_padded, float* out_dim) const;

  // Codebook used by the Beta-path quant/dequant. nullptr if the bit width
  // isn't supported by that path (currently B12 is skipped — too many levels
  // to be useful at this scope).
  const BetaCodebook* beta_codebook(QuantBits bits) const;

 private:
  size_t dim_;
  size_t padded_dim_;
  std::vector<float> signs_;  // length padded_dim_, entries are +1 / -1
  // One codebook per supported bit width. Indexed by Bits(QuantBits).
  std::vector<std::unique_ptr<BetaCodebook>> beta_codebooks_;
};

// Encode `x` (length rot.dim()) into `payload_out` (PayloadSize(rot.dim(), bits)
// bytes). Internally picks the best-known quantization scheme for the chosen
// bit width: a Lloyd-Max codebook over the Beta distribution induced by the
// rotation for low bits (B1/B2/B3/B4/B6, where it dramatically improves recall),
// or affine min/max quantization for high bits (B8/B12, where the two schemes
// are equivalent and affine is cheaper to encode).
void Quantize(const Rotator& rot, QuantBits bits, const float* x,
              uint8_t* payload_out);

// Decode `payload` back to the original-dimension vector `x_out` (length
// rot.dim()). Routing matches Quantize.
void Dequantize(const Rotator& rot, QuantBits bits, const uint8_t* payload,
                float* x_out);

}  // namespace turboquant

#endif  // TURBOQUANT_TURBOQUANT_H_
