#include "turboquant/turboquant.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

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
}

void Rotator::Apply(const float* x, float* out) const {
  std::memcpy(out, x, dim_ * sizeof(float));
  if (padded_dim_ > dim_) {
    std::memset(out + dim_, 0, (padded_dim_ - dim_) * sizeof(float));
  }
  ApplySigns(out, signs_.data(), padded_dim_);
  HadamardTransform(out, padded_dim_);
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
