#include "rotator_mixed.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include "codebook.h"
#include "kernels.h"
#include "packing.h"
#include "rotation.h"

namespace turboquant {
namespace internal {

namespace {

constexpr size_t kHeaderBytes = 4;

bool IsPowerOfTwo(size_t x) { return x > 0 && (x & (x - 1)) == 0; }

uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int BitsInt(QuantBits b) { return static_cast<int>(b); }

thread_local std::vector<float> tls_rotated;
thread_local std::vector<uint16_t> tls_codes;

}  // namespace

bool RotatorMixed3::DimSupported(size_t dim) {
  return dim > 0 && (dim % 3) == 0 && IsPowerOfTwo(dim / 3);
}

RotatorMixed3::RotatorMixed3(size_t dim, uint64_t seed)
    : dim_(dim),
      n_block_(dim / 3),
      signs_(dim, 1.0f),
      // Codebooks are computed against the true (unpadded) dim — coordinates
      // of a rotated unit vector in R^dim follow Beta((dim-1)/2, (dim-1)/2) —
      // and are built on first use, not here: a Quantizer reads at most one
      // bit width, and the affine widths read none.
      beta_codebooks_(dim) {
  assert(DimSupported(dim) && "RotatorMixed3 requires dim = 3 * 2^k");

  uint64_t s = seed ? seed : 0xD1B54A32D192ED03ULL;
  for (size_t i = 0; i < dim_; ++i) {
    const uint64_t r = SplitMix64(s);
    signs_[i] = (r & 1ULL) ? 1.0f : -1.0f;
  }
}

RotatorMixed3::~RotatorMixed3() = default;
RotatorMixed3::RotatorMixed3(RotatorMixed3&&) noexcept = default;
RotatorMixed3& RotatorMixed3::operator=(RotatorMixed3&&) noexcept = default;

const BetaCodebook* RotatorMixed3::beta_codebook(QuantBits bits) const {
  return beta_codebooks_.Get(bits);
}

void RotatorMixed3::Apply(const float* x, float* out) const {
  // Forward: out := (1/sqrt(N)) * H_3 * WHT_N(per row) * D * x.
  // Step 1+2: D * x with the WHT's first SIMD load fused (per row).
  std::memcpy(out, x, dim_ * sizeof(float));
  // ForwardRotate applies signs + WHT + 1/sqrt(N) normalization per length-N
  // row. We invoke it on each of the 3 rows separately.
  for (size_t r = 0; r < 3; ++r) {
    ForwardRotate(out + r * n_block_, signs_.data() + r * n_block_, n_block_);
  }
  // Step 3: H_3 across columns.
  Householder3InPlace(out, n_block_);
}

void RotatorMixed3::ApplyInverse(float* y, float* out) const {
  // Inverse of (1/sqrt(N)) * H_3 * WHT_N * D is D * WHT_N * H_3 * sqrt(N).
  // Since each piece is orthogonal and self-inverse (for WHT we rely on
  // WHT^2 = scaled identity), we run the steps in reverse.
  Householder3InPlace(y, n_block_);
  // WHT_N inverse per row, with the sign-flip + 1/sqrt(N) folded — exactly
  // what the existing fast inverse pipeline does (unscaled WHT then a single
  // pass that multiplies by signs and the 1/sqrt(N) factor).
  for (size_t r = 0; r < 3; ++r) {
    FastHadamardTransformUnscaled(y + r * n_block_, n_block_);
  }
  const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n_block_));
  ApplySignsAndScale(y, signs_.data(), dim_, inv_sqrt_n);
  std::memcpy(out, y, dim_ * sizeof(float));
}

size_t PayloadSizeMixed3(size_t dim, QuantBits bits) {
  const size_t total_bits = dim * static_cast<size_t>(BitsInt(bits));
  return kHeaderBytes + (total_bits + 7) / 8;
}

void QuantizeMixed3(const RotatorMixed3& rot, QuantBits bits, const float* x,
                    uint8_t* payload_out) {
  const BetaCodebook* cb = rot.beta_codebook(bits);
  if (!cb) return;
  const size_t d = rot.dim();
  if (tls_rotated.size() < d) tls_rotated.resize(d);
  if (tls_codes.size() < d) tls_codes.resize(d);
  float* rotated = tls_rotated.data();
  uint16_t* codes = tls_codes.data();

  // Unit-normalize the input.
  double sumsq = 0.0;
  for (size_t i = 0; i < d; ++i) sumsq += static_cast<double>(x[i]) * x[i];
  const float norm = static_cast<float>(std::sqrt(sumsq));
  const float inv_norm = norm > 1e-20f ? 1.0f / norm : 0.0f;
  for (size_t i = 0; i < d; ++i) rotated[i] = x[i] * inv_norm;

  // Apply the mixed-radix forward rotation in-place. The Rotator's Apply is
  // not in-place, so feed it via a small detour: copy-in then transform.
  // Cheaper: hand-inline what Apply does on the already-populated buffer.
  for (size_t r = 0; r < 3; ++r) {
    const size_t N = d / 3;
    ForwardRotate(rotated + r * N, rot.signs() + r * N, N);
  }
  Householder3InPlace(rotated, d / 3);

  // Encode via branch-free symmetric binary search.
  EncodeBetaCodebook(rotated, d, cb->positive_boundaries_padded(), bits, codes);

  // scale = ||v|| / <u_rot, x_hat>.
  const float inner = CentroidInnerProduct(rotated, codes, d, cb->centroids());
  const float scale_eff = std::abs(inner) > 1e-20f ? norm / inner : norm;

  std::memcpy(payload_out, &scale_eff, sizeof(float));
  PackCodes(codes, d, bits, payload_out + kHeaderBytes);
}

void QuantizeMixed3Affine(const RotatorMixed3& rot, QuantBits bits,
                          const float* x, uint8_t* payload_out) {
  const size_t d = rot.dim();
  if (tls_rotated.size() < d) tls_rotated.resize(d);
  if (tls_codes.size() < d) tls_codes.resize(d);
  float* rotated = tls_rotated.data();
  uint16_t* codes = tls_codes.data();

  // Apply the mixed-radix forward rotation directly on the input.
  rot.Apply(x, rotated);

  const int b = BitsInt(bits);
  float scale;
  if (b == 1) {
    const float m = MaxAbs(rotated, d);
    scale = m > 0.0f ? m : 1.0f;
    QuantizeBinary(rotated, d, codes);
  } else {
    const int max_pos = (1 << (b - 1)) - 1;
    const int zp = 1 << (b - 1);
    const int max_code = (1 << b) - 1;
    const float m = MaxAbs(rotated, d);
    scale = m > 0.0f ? m / static_cast<float>(max_pos) : 1.0f;
    QuantizeAffine(rotated, d, scale, zp, max_code, codes);
  }
  std::memcpy(payload_out, &scale, sizeof(float));
  PackCodes(codes, d, bits, payload_out + kHeaderBytes);
}

void DequantizeMixed3Affine(const RotatorMixed3& rot, QuantBits bits,
                            const uint8_t* payload, float* x_out) {
  const size_t d = rot.dim();
  float scale;
  std::memcpy(&scale, payload, sizeof(float));

  if (tls_codes.size() < d) tls_codes.resize(d);
  if (tls_rotated.size() < d) tls_rotated.resize(d);
  uint16_t* codes = tls_codes.data();
  float* rotated = tls_rotated.data();
  UnpackCodes(payload + kHeaderBytes, d, bits, codes);

  const int b = BitsInt(bits);
  if (b == 1) {
    DequantizeBinary(codes, d, scale, rotated);
  } else {
    const int zp = 1 << (b - 1);
    DequantizeAffine(codes, d, scale, zp, rotated);
  }
  rot.ApplyInverse(rotated, x_out);
}

void DequantizeMixed3(const RotatorMixed3& rot, QuantBits bits,
                      const uint8_t* payload, float* x_out) {
  const BetaCodebook* cb = rot.beta_codebook(bits);
  if (!cb) return;
  const size_t d = rot.dim();
  float scale;
  std::memcpy(&scale, payload, sizeof(float));

  if (tls_codes.size() < d) tls_codes.resize(d);
  if (tls_rotated.size() < d) tls_rotated.resize(d);
  uint16_t* codes = tls_codes.data();
  float* rotated = tls_rotated.data();
  UnpackCodes(payload + kHeaderBytes, d, bits, codes);

  // rotated[i] = scale * centroid[codes[i]]
  DecodeBetaCodebook(codes, d, cb->centroids(), scale, rotated);

  rot.ApplyInverse(rotated, x_out);
}

}  // namespace internal
}  // namespace turboquant
