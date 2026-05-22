#ifndef TURBOQUANT_TURBOQUANT_H_
#define TURBOQUANT_TURBOQUANT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
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

struct AdcStats {
  float dot;
  float decoded_norm2;
};

// Smallest power-of-two >= dim (and >= 1).
size_t PaddedDim(size_t dim);

// Bytes of packed codes for `padded_dim` codes at the given bit width.
size_t PackedBytes(size_t padded_dim, QuantBits bits);

// Total per-vector payload size: 4-byte scale + 12-byte reserved + N packed-code bytes.
size_t PayloadSize(size_t dim, QuantBits bits);

// TurboQuant rotation: x -> H * D * pad(x), where:
//   - pad zero-extends to next_pow2(dim);
//   - D is a deterministic diagonal of random {-1,+1} entries seeded by `seed`;
//   - H is the normalized Walsh-Hadamard transform (orthogonal, ||Hv|| = ||v||).
//
// Use the same Rotator for both vectors and queries so they live in the same rotated space.
class Rotator {
 public:
  Rotator(size_t dim, uint64_t seed);

  size_t dim() const { return dim_; }
  size_t padded_dim() const { return padded_dim_; }
  const float* signs() const { return signs_.data(); }

  // Apply H * D * pad(x). `x` has length dim(); `out` has length padded_dim().
  void Apply(const float* x, float* out) const;

  // Apply transpose: pad(x') = D * H^T * y. Since H is symmetric and orthogonal,
  // and D is its own inverse, the inverse rotation is D * H * y. `y` has length
  // padded_dim(); `out` has length dim() (the leading dim() entries of D*H*y).
  void ApplyInverse(const float* y, float* out) const;

 private:
  size_t dim_;
  size_t padded_dim_;
  std::vector<float> signs_;  // length padded_dim_, entries are +1 / -1
};

// Encode `x` (length rot.dim()) into `payload_out` (PayloadSize(rot.dim(), bits) bytes).
// Internally rotates x with `rot` then quantizes to `bits`.
void Quantize(const Rotator& rot, QuantBits bits, const float* x,
              uint8_t* payload_out);

// Decode `payload` back to the original-dimension vector `x_out`
// (length rot.dim()). This applies the inverse rotation.
void Dequantize(const Rotator& rot, QuantBits bits, const uint8_t* payload,
                float* x_out);

// Pre-rotate a query for ADC scoring. `q` has length rot.dim();
// `q_rot_out` has length rot.padded_dim().
void RotateQuery(const Rotator& rot, const float* q, float* q_rot_out);

// Approximate dot product and decoded squared L2 norm of the quantized vector
// against `q_rot` (already produced by RotateQuery). The rotation is orthogonal,
// so dot(query, x) = dot(query_rot, x_rot).
AdcStats AdcScore(const Rotator& rot, QuantBits bits, const float* q_rot,
                  const uint8_t* payload);

}  // namespace turboquant

#endif  // TURBOQUANT_TURBOQUANT_H_
