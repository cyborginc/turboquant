#include "turboquant/turboquant.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "codebook.h"
#include "kernels.h"
#include "packing.h"
#include "rotation.h"

namespace turboquant {

namespace {

constexpr size_t kHeaderBytes = 4;  // 4-byte scale (LE float32).

size_t NextPow2(size_t x) {
  if (x <= 1) return 1;
  size_t v = 1;
  while (v < x) v <<= 1;
  return v;
}

// Tiny splitmix64 to derive per-position signs deterministically from a seed.
uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int Bits(QuantBits b) { return static_cast<int>(b); }

thread_local std::vector<float> tls_rotated;
thread_local std::vector<uint16_t> tls_codes;

}  // namespace

size_t PaddedDim(size_t dim) { return NextPow2(dim); }

size_t PackedBytes(size_t padded_dim, QuantBits bits) {
  const size_t total_bits = padded_dim * static_cast<size_t>(Bits(bits));
  return (total_bits + 7) / 8;
}

size_t PayloadSize(size_t dim, QuantBits bits) {
  return kHeaderBytes + PackedBytes(PaddedDim(dim), bits);
}

Rotator::Rotator(size_t dim, uint64_t seed)
    : dim_(dim), padded_dim_(NextPow2(dim)), signs_(padded_dim_, 1.0f) {
  uint64_t s = seed ? seed : 0xD1B54A32D192ED03ULL;
  for (size_t i = 0; i < padded_dim_; ++i) {
    const uint64_t r = SplitMix64(s);
    signs_[i] = (r & 1ULL) ? 1.0f : -1.0f;
  }
  // Build Beta codebooks for all supported bit widths (B12 skipped — 4096
  // levels makes both the Lloyd-Max construction and the per-coord encode
  // expensive enough not to be useful at this scope).
  beta_codebooks_.resize(13);  // indexed by Bits(QuantBits); only some slots filled.
  for (QuantBits b : {QuantBits::B1, QuantBits::B2, QuantBits::B4,
                      QuantBits::B6, QuantBits::B8}) {
    const int bi = Bits(b);
    beta_codebooks_[bi] = std::make_unique<BetaCodebook>(b, padded_dim_);
  }
}

Rotator::~Rotator() = default;
Rotator::Rotator(Rotator&&) noexcept = default;
Rotator& Rotator::operator=(Rotator&&) noexcept = default;

const BetaCodebook* Rotator::beta_codebook(QuantBits bits) const {
  const int bi = Bits(bits);
  if (bi < 0 || static_cast<size_t>(bi) >= beta_codebooks_.size()) return nullptr;
  return beta_codebooks_[bi].get();
}

void Rotator::Apply(const float* x, float* out) const {
  std::memcpy(out, x, dim_ * sizeof(float));
  if (padded_dim_ > dim_) {
    std::memset(out + dim_, 0, (padded_dim_ - dim_) * sizeof(float));
  }
  ForwardRotate(out, signs_.data(), padded_dim_);
}

void Rotator::ApplyInverse(float* y_padded, float* out_dim) const {
  // H is symmetric & orthogonal in this normalization; D is its own inverse.
  // y = H D x  =>  x = D H y. We run the unscaled WHT and then a single SIMD
  // pass that combines the sign flip with the 1/sqrt(padded_dim) normalization
  // (saving one pass over y_padded compared to running each in turn).
  FastHadamardTransformUnscaled(y_padded, padded_dim_);
  const float inv_sqrt_pd = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
  ApplySignsAndScale(y_padded, signs_.data(), padded_dim_, inv_sqrt_pd);
  std::memcpy(out_dim, y_padded, dim_ * sizeof(float));
}

void Quantize(const Rotator& rot, QuantBits bits, const float* x,
              uint8_t* payload_out) {
  const size_t pd = rot.padded_dim();
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  if (tls_codes.size() < pd) tls_codes.resize(pd);
  float* rotated = tls_rotated.data();
  uint16_t* codes = tls_codes.data();
  rot.Apply(x, rotated);

  const int b = Bits(bits);

  float scale;
  if (b == 1) {
    const float m = MaxAbs(rotated, pd);
    scale = m > 0.0f ? m : 1.0f;
    QuantizeBinary(rotated, pd, codes);
  } else {
    const int max_pos = (1 << (b - 1)) - 1;
    const int zp = 1 << (b - 1);
    const int max_code = (1 << b) - 1;
    const float m = MaxAbs(rotated, pd);
    scale = m > 0.0f ? m / static_cast<float>(max_pos) : 1.0f;
    QuantizeAffine(rotated, pd, scale, zp, max_code, codes);
  }

  // Header: just the scale (LE float32).
  std::memcpy(payload_out, &scale, sizeof(float));

  // Packed codes.
  PackCodes(codes, pd, bits, payload_out + kHeaderBytes);
}

void QuantizeBeta(const Rotator& rot, QuantBits bits, const float* x,
                  uint8_t* payload_out) {
  const BetaCodebook* cb = rot.beta_codebook(bits);
  if (!cb) return;  // unsupported (B12)
  const size_t pd = rot.padded_dim();
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  if (tls_codes.size() < pd) tls_codes.resize(pd);
  float* rotated = tls_rotated.data();
  uint16_t* codes = tls_codes.data();

  // Compute ||v|| from the original (un-padded) input.
  double sumsq = 0.0;
  const size_t d = rot.dim();
  for (size_t i = 0; i < d; ++i) sumsq += static_cast<double>(x[i]) * x[i];
  const float norm = static_cast<float>(std::sqrt(sumsq));
  const float inv_norm = norm > 1e-20f ? 1.0f / norm : 0.0f;

  // Normalize into rotated[] (with zero padding), then apply forward rotation.
  for (size_t i = 0; i < d; ++i) rotated[i] = x[i] * inv_norm;
  if (pd > d) std::memset(rotated + d, 0, (pd - d) * sizeof(float));
  // Apply signs + WHT + 1/sqrt(pd) — same path as Rotator::Apply, but on the
  // already-padded buffer (skip the memcpy into rotated[] that Apply does).
  ForwardRotate(rotated, rot.signs(), pd);

  // Encode each rotated coord to its Lloyd-Max code.
  EncodeBetaCodebook(rotated, pd, cb->boundaries(), cb->num_boundaries(), codes);

  // scale = ||v|| / <u_rot, x_hat>. Falls back to ||v|| if the inner product
  // degenerates (e.g., zero vector).
  const float inner =
      CentroidInnerProduct(rotated, codes, pd, cb->centroids());
  const float scale_eff =
      std::abs(inner) > 1e-20f ? norm / inner : norm;

  std::memcpy(payload_out, &scale_eff, sizeof(float));
  PackCodes(codes, pd, bits, payload_out + kHeaderBytes);
}

void DequantizeBeta(const Rotator& rot, QuantBits bits, const uint8_t* payload,
                    float* x_out) {
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

  // rotated[i] = scale * centroid[codes[i]] — single SIMD pass with gather.
  DecodeBetaCodebook(codes, pd, cb->centroids(), scale, rotated);

  // Inverse rotation (same as the affine path).
  rot.ApplyInverse(rotated, x_out);
}

void Dequantize(const Rotator& rot, QuantBits bits, const uint8_t* payload,
                float* x_out) {
  const size_t pd = rot.padded_dim();
  float scale;
  std::memcpy(&scale, payload, sizeof(float));

  if (tls_codes.size() < pd) tls_codes.resize(pd);
  if (tls_rotated.size() < pd) tls_rotated.resize(pd);
  uint16_t* codes = tls_codes.data();
  float* rotated = tls_rotated.data();
  UnpackCodes(payload + kHeaderBytes, pd, bits, codes);

  const int b = Bits(bits);
  if (b == 1) {
    DequantizeBinary(codes, pd, scale, rotated);
  } else {
    const int zp = 1 << (b - 1);
    DequantizeAffine(codes, pd, scale, zp, rotated);
  }
  rot.ApplyInverse(rotated, x_out);
}

}  // namespace turboquant
