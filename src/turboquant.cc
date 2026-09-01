#include "turboquant/turboquant.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "codebook.h"
#include "internal.h"
#include "kernels.h"
#include "packing.h"
#include "rotation.h"
#include "rotator_padded.h"
#include "rotator_mixed.h"

namespace turboquant {

namespace {

constexpr size_t kHeaderBytes = 4;  // 4-byte scale (LE float32).

size_t NextPow2(size_t x) {
  if (x <= 1) return 1;
  size_t v = 1;
  while (v < x) v <<= 1;
  return v;
}

uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int BitsInt(QuantBits b) { return static_cast<int>(b); }

bool UseBetaPath(QuantBits bits) {
  switch (bits) {
    case QuantBits::B1:
    case QuantBits::B2:
    case QuantBits::B3:
    case QuantBits::B4:
    case QuantBits::B6:
      return true;
    case QuantBits::B8:
    case QuantBits::B12:
      return false;
  }
  return false;
}

thread_local std::vector<float> tls_rotated;
thread_local std::vector<uint16_t> tls_codes;

}  // namespace

// ---------------------------------------------------------------------------
// internal::RotatorPadded
// ---------------------------------------------------------------------------
namespace internal {

RotatorPadded::RotatorPadded(size_t dim, uint64_t seed)
    : dim_(dim),
      padded_dim_(NextPow2(dim)),
      signs_(padded_dim_, 1.0f),
      // Codebooks are built on first use, not here: a Quantizer reads at most
      // one bit width, and the affine widths read none.
      beta_codebooks_(padded_dim_) {
  uint64_t s = seed ? seed : 0xD1B54A32D192ED03ULL;
  for (size_t i = 0; i < padded_dim_; ++i) {
    const uint64_t r = SplitMix64(s);
    signs_[i] = (r & 1ULL) ? 1.0f : -1.0f;
  }
}

RotatorPadded::~RotatorPadded() = default;
RotatorPadded::RotatorPadded(RotatorPadded&&) noexcept = default;
RotatorPadded& RotatorPadded::operator=(RotatorPadded&&) noexcept = default;

const BetaCodebook* RotatorPadded::beta_codebook(QuantBits bits) const {
  return beta_codebooks_.Get(bits);
}

void RotatorPadded::Apply(const float* x, float* out) const {
  std::memcpy(out, x, dim_ * sizeof(float));
  if (padded_dim_ > dim_) {
    std::memset(out + dim_, 0, (padded_dim_ - dim_) * sizeof(float));
  }
  ForwardRotate(out, signs_.data(), padded_dim_);
}

void RotatorPadded::ApplyInverse(float* y_padded, float* out_dim) const {
  FastHadamardTransformUnscaled(y_padded, padded_dim_);
  const float inv_sqrt_pd = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
  ApplySignsAndScale(y_padded, signs_.data(), padded_dim_, inv_sqrt_pd);
  std::memcpy(out_dim, y_padded, dim_ * sizeof(float));
}

// ---------------------------------------------------------------------------
// internal::Quantize{Affine,Beta} + payload size for the padded path.
// ---------------------------------------------------------------------------

size_t PayloadSizePadded(size_t dim, QuantBits bits) {
  const size_t total_bits =
      NextPow2(dim) * static_cast<size_t>(BitsInt(bits));
  return kHeaderBytes + (total_bits + 7) / 8;
}

void QuantizeAffine(const RotatorPadded& rot, QuantBits bits, const float* x,
                    uint8_t* payload_out) {
  const size_t pd = rot.padded_dim();
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  if (tls_codes.size() < pd) tls_codes.resize(pd);
  float* rotated = tls_rotated.data();
  uint16_t* codes = tls_codes.data();
  rot.Apply(x, rotated);

  const int b = BitsInt(bits);

  float scale;
  if (b == 1) {
    const float m = ::turboquant::MaxAbs(rotated, pd);
    scale = m > 0.0f ? m : 1.0f;
    ::turboquant::QuantizeBinary(rotated, pd, codes);
  } else {
    const int max_pos = (1 << (b - 1)) - 1;
    const int zp = 1 << (b - 1);
    const int max_code = (1 << b) - 1;
    const float m = ::turboquant::MaxAbs(rotated, pd);
    scale = m > 0.0f ? m / static_cast<float>(max_pos) : 1.0f;
    ::turboquant::QuantizeAffine(rotated, pd, scale, zp, max_code, codes);
  }
  std::memcpy(payload_out, &scale, sizeof(float));
  PackCodes(codes, pd, bits, payload_out + kHeaderBytes);
}

void QuantizeBeta(const RotatorPadded& rot, QuantBits bits, const float* x,
                  uint8_t* payload_out) {
  const BetaCodebook* cb = rot.beta_codebook(bits);
  if (!cb) return;
  const size_t pd = rot.padded_dim();
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  if (tls_codes.size() < pd) tls_codes.resize(pd);
  float* rotated = tls_rotated.data();
  uint16_t* codes = tls_codes.data();

  double sumsq = 0.0;
  const size_t d = rot.dim();
  for (size_t i = 0; i < d; ++i) sumsq += static_cast<double>(x[i]) * x[i];
  const float norm = static_cast<float>(std::sqrt(sumsq));
  const float inv_norm = norm > 1e-20f ? 1.0f / norm : 0.0f;

  for (size_t i = 0; i < d; ++i) rotated[i] = x[i] * inv_norm;
  if (pd > d) std::memset(rotated + d, 0, (pd - d) * sizeof(float));
  ForwardRotate(rotated, rot.signs(), pd);

  EncodeBetaCodebook(rotated, pd, cb->positive_boundaries_padded(), bits, codes);

  const float inner =
      CentroidInnerProduct(rotated, codes, pd, cb->centroids());
  const float scale_eff =
      std::abs(inner) > 1e-20f ? norm / inner : norm;

  std::memcpy(payload_out, &scale_eff, sizeof(float));
  PackCodes(codes, pd, bits, payload_out + kHeaderBytes);
}

void DequantizeAffine(const RotatorPadded& rot, QuantBits bits,
                      const uint8_t* payload, float* x_out) {
  const size_t pd = rot.padded_dim();
  float scale;
  std::memcpy(&scale, payload, sizeof(float));

  if (tls_codes.size() < pd) tls_codes.resize(pd);
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  uint16_t* codes = tls_codes.data();
  float* rotated = tls_rotated.data();
  UnpackCodes(payload + kHeaderBytes, pd, bits, codes);

  const int b = BitsInt(bits);
  if (b == 1) {
    ::turboquant::DequantizeBinary(codes, pd, scale, rotated);
  } else {
    const int zp = 1 << (b - 1);
    ::turboquant::DequantizeAffine(codes, pd, scale, zp, rotated);
  }
  rot.ApplyInverse(rotated, x_out);
}

void DequantizeBeta(const RotatorPadded& rot, QuantBits bits,
                    const uint8_t* payload, float* x_out) {
  const BetaCodebook* cb = rot.beta_codebook(bits);
  if (!cb) return;
  const size_t pd = rot.padded_dim();
  float scale;
  std::memcpy(&scale, payload, sizeof(float));

  if (tls_codes.size() < pd) tls_codes.resize(pd);
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  uint16_t* codes = tls_codes.data();
  float* rotated = tls_rotated.data();
  UnpackCodes(payload + kHeaderBytes, pd, bits, codes);

  DecodeBetaCodebook(codes, pd, cb->centroids(), scale, rotated);
  rot.ApplyInverse(rotated, x_out);
}

}  // namespace internal

// ---------------------------------------------------------------------------
// Public Quantizer
// ---------------------------------------------------------------------------

struct Quantizer::Impl {
  size_t dim;
  QuantBits bits;
  size_t payload_bytes;
  bool use_mixed3;
  bool use_beta;
  std::unique_ptr<internal::RotatorPadded> padded;
  std::unique_ptr<internal::RotatorMixed3> mixed3;
};

size_t Quantizer::PayloadBytes(size_t dim, QuantBits bits) {
  if (internal::RotatorMixed3::DimSupported(dim)) {
    return internal::PayloadSizeMixed3(dim, bits);
  }
  return internal::PayloadSizePadded(dim, bits);
}

Quantizer::Quantizer(size_t dim, QuantBits bits, uint64_t seed)
    : impl_(std::make_unique<Impl>()) {
  impl_->dim = dim;
  impl_->bits = bits;
  impl_->payload_bytes = PayloadBytes(dim, bits);
  impl_->use_mixed3 = internal::RotatorMixed3::DimSupported(dim);
  impl_->use_beta = UseBetaPath(bits);
  if (impl_->use_mixed3) {
    impl_->mixed3 = std::make_unique<internal::RotatorMixed3>(dim, seed);
  } else {
    impl_->padded = std::make_unique<internal::RotatorPadded>(dim, seed);
  }
}

Quantizer::~Quantizer() = default;
Quantizer::Quantizer(Quantizer&&) noexcept = default;
Quantizer& Quantizer::operator=(Quantizer&&) noexcept = default;

size_t Quantizer::dim() const { return impl_->dim; }
QuantBits Quantizer::bits() const { return impl_->bits; }
size_t Quantizer::payload_bytes() const { return impl_->payload_bytes; }

void Quantizer::Quantize(const float* x, size_t n,
                         uint8_t* payloads_out) const {
  const size_t d = impl_->dim;
  const size_t ps = impl_->payload_bytes;
  if (impl_->use_mixed3) {
    if (impl_->use_beta) {
      for (size_t i = 0; i < n; ++i) {
        internal::QuantizeMixed3(*impl_->mixed3, impl_->bits, x + i * d,
                                 payloads_out + i * ps);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        internal::QuantizeMixed3Affine(*impl_->mixed3, impl_->bits, x + i * d,
                                       payloads_out + i * ps);
      }
    }
  } else {
    if (impl_->use_beta) {
      for (size_t i = 0; i < n; ++i) {
        internal::QuantizeBeta(*impl_->padded, impl_->bits, x + i * d,
                               payloads_out + i * ps);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        internal::QuantizeAffine(*impl_->padded, impl_->bits, x + i * d,
                                 payloads_out + i * ps);
      }
    }
  }
}

void Quantizer::Dequantize(const uint8_t* payloads, size_t n,
                           float* x_out) const {
  const size_t d = impl_->dim;
  const size_t ps = impl_->payload_bytes;
  if (impl_->use_mixed3) {
    if (impl_->use_beta) {
      for (size_t i = 0; i < n; ++i) {
        internal::DequantizeMixed3(*impl_->mixed3, impl_->bits,
                                   payloads + i * ps, x_out + i * d);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        internal::DequantizeMixed3Affine(*impl_->mixed3, impl_->bits,
                                         payloads + i * ps, x_out + i * d);
      }
    }
  } else {
    if (impl_->use_beta) {
      for (size_t i = 0; i < n; ++i) {
        internal::DequantizeBeta(*impl_->padded, impl_->bits,
                                 payloads + i * ps, x_out + i * d);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        internal::DequantizeAffine(*impl_->padded, impl_->bits,
                                   payloads + i * ps, x_out + i * d);
      }
    }
  }
}

}  // namespace turboquant
