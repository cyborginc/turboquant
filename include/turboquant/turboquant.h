#ifndef TURBOQUANT_TURBOQUANT_H_
#define TURBOQUANT_TURBOQUANT_H_

#include <cstddef>
#include <cstdint>
#include <memory>

namespace turboquant {

enum class QuantBits : uint8_t {
  B1 = 1,
  B2 = 2,
  B3 = 3,
  B4 = 4,
  B6 = 6,
  B8 = 8,
  B12 = 12,
};

// Single quantizer instance, bound to a specific (dim, bits, seed) at
// construction. Picks both the rotation scheme (mixed-radix for dim = 3*2^k,
// padded Walsh-Hadamard otherwise) and the quantization scheme (Lloyd-Max
// Beta codebook at b1/b2/b3/b4/b6, affine min/max at b8/b12) automatically.
// The user never has to know which path was picked.
//
// Construction builds the Lloyd-Max codebook for the chosen bit width
// (~10-200 ms at d=768 depending on configuration). After construction the
// encode/decode paths are zero-allocation per call.
//
// Thread-safety: Quantize/Dequantize are safe to call concurrently from
// multiple threads against the same Quantizer instance (per-thread scratch
// buffers are used internally).
class Quantizer {
 public:
  // Number of bytes one encoded payload occupies for the given (dim, bits).
  // Computable without constructing.
  static size_t PayloadBytes(size_t dim, QuantBits bits);

  // `seed = 0` selects the built-in canonical seed, which is what encoders
  // and decoders that use the default agree on. Pass a non-zero value only
  // if you need a custom rotation (e.g., to encode the same data under
  // multiple independent rotations).
  Quantizer(size_t dim, QuantBits bits, uint64_t seed = 0);
  ~Quantizer();
  Quantizer(Quantizer&&) noexcept;
  Quantizer& operator=(Quantizer&&) noexcept;
  Quantizer(const Quantizer&) = delete;
  Quantizer& operator=(const Quantizer&) = delete;

  size_t dim() const;
  QuantBits bits() const;
  size_t payload_bytes() const;  // == PayloadBytes(dim(), bits())

  // Encode `n` contiguous row-major vectors of dim floats into `n` contiguous
  // payloads. n = 1 is the single-vector case. Zero allocation; caller owns
  // both buffers.
  //   x          : n * dim() floats
  //   payloads   : n * payload_bytes() uint8s
  void Quantize(const float* x, size_t n, uint8_t* payloads_out) const;

  // Decode `n` payloads into `n` contiguous row-major vectors.
  void Dequantize(const uint8_t* payloads, size_t n, float* x_out) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace turboquant

#endif  // TURBOQUANT_TURBOQUANT_H_
