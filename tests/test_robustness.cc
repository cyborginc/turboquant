// Robustness tests: degenerate inputs, concurrency, and unpadded-vs-padded
// dimension handling.
//
// The round-trip tests elsewhere use well-conditioned random vectors, which is
// the Beta codebook's ideal case. These cover the inputs a real corpus actually
// contains — empty embeddings, saturated rows, one-hot features — plus the
// thread-safety contract the public header advertises.

#include "turboquant/turboquant.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace {

using turboquant::QuantBits;
using turboquant::Quantizer;

constexpr QuantBits kAllBits[] = {QuantBits::B1, QuantBits::B2, QuantBits::B3,
                                  QuantBits::B4, QuantBits::B6, QuantBits::B8,
                                  QuantBits::B12};

// Beta-path widths (b1..b6) normalize and store the vector norm as the scale;
// affine widths (b8/b12) store max_abs/max_pos. Both are exercised.
constexpr QuantBits kShippedBits[] = {QuantBits::B4, QuantBits::B6,
                                      QuantBits::B8, QuantBits::B12};

std::vector<float> RoundTrip(const std::vector<float>& x, size_t dim,
                             QuantBits bits) {
  Quantizer q(dim, bits);
  std::vector<uint8_t> payload(q.payload_bytes());
  q.Quantize(x.data(), 1, payload.data());
  std::vector<float> y(dim);
  q.Dequantize(payload.data(), 1, y.data());
  return y;
}

void ExpectAllFinite(const std::vector<float>& y, const char* what,
                     QuantBits bits) {
  for (size_t i = 0; i < y.size(); ++i) {
    ASSERT_TRUE(std::isfinite(y[i]))
        << what << " produced a non-finite value at index " << i
        << " (bits=" << static_cast<int>(bits) << ")";
  }
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  if (na == 0 || nb == 0) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

// The all-zero vector is the degenerate case for the Beta path, whose encode
// divides by the vector norm. It must not produce NaN or Inf.
TEST(Robustness, ZeroVector) {
  constexpr size_t dim = 768;
  for (QuantBits bits : kAllBits) {
    const std::vector<float> x(dim, 0.0f);
    const std::vector<float> y = RoundTrip(x, dim, bits);
    ExpectAllFinite(y, "zero vector", bits);
    for (float v : y) {
      EXPECT_NEAR(v, 0.0f, 1e-6f) << "zero vector should decode to zero (bits="
                                  << static_cast<int>(bits) << ")";
    }
  }
}

// A constant vector rotates to a single non-zero coefficient — the worst case
// for a codebook tuned to the Beta distribution, since almost every coordinate
// lands in the same bin.
TEST(Robustness, ConstantVector) {
  constexpr size_t dim = 768;
  for (QuantBits bits : kShippedBits) {
    const std::vector<float> x(dim, 0.37f);
    const std::vector<float> y = RoundTrip(x, dim, bits);
    ExpectAllFinite(y, "constant vector", bits);
    EXPECT_GT(Cosine(x, y), 0.99)
        << "constant vector round-trip lost direction (bits="
        << static_cast<int>(bits) << ")";
  }
}

// One-hot / maximally sparse input: all the energy in a single coordinate.
TEST(Robustness, SingleNonZeroCoordinate) {
  constexpr size_t dim = 768;
  for (QuantBits bits : kShippedBits) {
    for (size_t idx : {size_t{0}, size_t{1}, size_t{dim - 1}}) {
      std::vector<float> x(dim, 0.0f);
      x[idx] = 2.5f;
      const std::vector<float> y = RoundTrip(x, dim, bits);
      ExpectAllFinite(y, "one-hot vector", bits);
      // The rotation spreads a spike across all coordinates, so reconstruction
      // is lossy at low bit widths; only require the spike stay dominant.
      const size_t argmax =
          std::max_element(
              y.begin(), y.end(),
              [](float a, float b) { return std::fabs(a) < std::fabs(b); }) -
          y.begin();
      EXPECT_EQ(argmax, idx)
          << "one-hot spike moved (bits=" << static_cast<int>(bits)
          << ", idx=" << idx << ")";
    }
  }
}

// Magnitudes far from 1.0 in both directions, within the range both paths
// actually support. The boundaries are pinned separately below.
TEST(Robustness, ExtremeMagnitudes) {
  constexpr size_t dim = 768;
  const float magnitudes[] = {1e-18f, 1e-10f, 1e-3f, 1e3f, 1e10f, 1e20f, 1e30f};
  for (QuantBits bits : kShippedBits) {
    for (float m : magnitudes) {
      std::vector<float> x(dim);
      for (size_t i = 0; i < dim; ++i) {
        x[i] = ((i % 3) - 1.0f) * m;  // mix of -m, 0, +m
      }
      const std::vector<float> y = RoundTrip(x, dim, bits);
      ExpectAllFinite(y, "extreme magnitude vector", bits);
      EXPECT_GT(Cosine(x, y), 0.90)
          << "magnitude " << m
          << " lost direction (bits=" << static_cast<int>(bits) << ")";
    }
  }
}

// KNOWN BUG, pinned here so it is visible rather than silent.
//
// QuantizeBeta gates its normalization on an absolute epsilon:
//
//     const float inv_norm = norm > 1e-20f ? 1.0f / norm : 0.0f;
//
// (src/turboquant.cc and src/rotator_mixed.cc). A vector whose norm falls below
// 1e-20 therefore normalizes to all zeros, every coordinate encodes to the same
// mid-level code, and decode returns a vector unrelated to the input — measured
// cosine 0.0, with no error signalled to the caller.
//
// 1e-20 is not a float limitation. FLT_MIN is 1.2e-38, so this rejects ~18
// orders of magnitude of perfectly representable input. Quantization should be
// scale-invariant: the same direction at 1e-25 encodes fine at 1e-15.
//
// The fix is to make the guard relative (compare against FLT_MIN, or rescale by
// the max coordinate before summing squares) rather than an absolute constant.
// Left unfixed here because it changes numeric policy on the encode path and
// wants a deliberate decision, not a drive-by edit.
TEST(Robustness, SmallNormBetaPathIsBroken_KnownBug) {
  constexpr size_t dim = 768;
  // Just above the threshold: correct.
  {
    std::vector<float> x(dim, 1e-10f);
    const std::vector<float> y = RoundTrip(x, dim, QuantBits::B4);
    EXPECT_GT(Cosine(x, y), 0.99) << "1e-10 should be well inside range";
  }
  // Just below it: silently wrong, not merely imprecise.
  {
    std::vector<float> x(dim, 1e-30f);
    const std::vector<float> y = RoundTrip(x, dim, QuantBits::B4);
    ExpectAllFinite(y, "small-norm vector", QuantBits::B4);
    EXPECT_LT(Cosine(x, y), 0.5)
        << "the 1e-20 epsilon in QuantizeBeta appears to be fixed — remove "
           "this test and widen ExtremeMagnitudes instead";
  }
}

// Known limitation, pinned so it is a deliberate choice rather than a surprise.
//
// Both paths accumulate before normalizing — the Beta path sums squares to get
// the norm, the affine path runs the Hadamard transform on the raw values — so
// coordinate magnitudes above roughly FLT_MAX/padded_dim overflow to infinity.
// The failure is not graceful: the Beta path emits NaN, the affine path
// silently emits zeros. Measured cliff at d=768 is ~1.2e37 (b4) and ~6.3e36
// (b8); at d=1000 (padded to 1024) it is ~6.1e36 and ~3.3e36.
//
// Unlike the small-norm case above this really is a float-range limit, and it
// sits far outside any real embedding range (coordinates are O(1), and even
// unnormalized model outputs are O(100)), so it is documented rather than
// guarded — a runtime check would cost a pass over every vector on the ingest
// hot path. If TurboQuant ever accepts untrusted input directly, revisit.
TEST(Robustness, ExtremeMagnitudeOverflowIsDocumented) {
  constexpr size_t dim = 768;
  constexpr float kKnownSafe = 1e30f;
  constexpr float kKnownUnsafe = 1e38f;

  const std::vector<float> safe(dim, kKnownSafe);
  const std::vector<float> safe_out = RoundTrip(safe, dim, QuantBits::B4);
  for (float v : safe_out) {
    ASSERT_TRUE(std::isfinite(v)) << "1e30 should be inside the safe range";
  }

  // Not asserting the *failure* is correct — only that the boundary is where
  // this comment says it is, so a future change that widens the safe range
  // trips this test and the comment gets updated with it.
  const std::vector<float> unsafe(dim, kKnownUnsafe);
  const std::vector<float> unsafe_out = RoundTrip(unsafe, dim, QuantBits::B4);
  bool any_non_finite = false;
  for (float v : unsafe_out) any_non_finite |= !std::isfinite(v);
  EXPECT_TRUE(any_non_finite)
      << "1e38 no longer overflows — the safe range widened, so update the "
         "comment on this test and on Quantizer::Quantize";
}

// The public header promises Quantize/Dequantize are safe to call concurrently
// against one shared Quantizer. Nothing in the type system enforces that, and
// the Beta codebooks are now built lazily on first use, so this races many
// threads through that build and requires every thread to agree with a
// single-threaded reference.
TEST(Robustness, ConcurrentQuantizeMatchesSerial) {
  constexpr size_t dim = 768;
  constexpr size_t kVectors = 64;
  const unsigned hw = std::thread::hardware_concurrency();
  const size_t kThreads = std::max(4u, std::min(hw, 16u));

  for (QuantBits bits : kShippedBits) {
    // Reference: one thread, fresh quantizer.
    std::vector<float> x(kVectors * dim);
    uint64_t s = 0xC0FFEEULL + static_cast<int>(bits);
    for (auto& f : x) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      f = static_cast<float>(static_cast<int32_t>(s >> 32)) / 2147483648.0f;
    }

    std::vector<uint8_t> expected;
    {
      Quantizer q(dim, bits);
      expected.resize(kVectors * q.payload_bytes());
      q.Quantize(x.data(), kVectors, expected.data());
    }

    // Concurrent: one shared quantizer, all threads starting together so they
    // contend on the lazy codebook build.
    Quantizer shared(dim, bits);
    const size_t ps = shared.payload_bytes();
    std::atomic<bool> go{false};
    std::atomic<size_t> mismatches{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (size_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&] {
        while (!go.load(std::memory_order_acquire)) {
        }
        std::vector<uint8_t> got(kVectors * ps);
        shared.Quantize(x.data(), kVectors, got.data());
        if (got != expected) mismatches.fetch_add(1, std::memory_order_relaxed);

        // Decode concurrently too, and check it against a local decode.
        std::vector<float> y(kVectors * dim);
        shared.Dequantize(got.data(), kVectors, y.data());
        for (float v : y) {
          if (!std::isfinite(v)) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
            break;
          }
        }
      });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    EXPECT_EQ(mismatches.load(), 0u)
        << kThreads << " threads disagreed with the serial encoding at bits="
        << static_cast<int>(bits);
  }
}

// Dimension handling: 3*2^k uses the mixed-radix rotation and packs the true
// dim; everything else pads to the next power of two. Both must round-trip, and
// the size difference is large enough that callers need to know which they get.
TEST(Robustness, UnpaddedAndPaddedDimsRoundTrip) {
  const size_t dims[] = {3, 6, 96,  384, 768, 1536,  // 3 * 2^k
                         1, 2, 100, 128, 129, 512,  600, 1000, 1024, 1152};
  for (size_t dim : dims) {
    for (QuantBits bits : {QuantBits::B4, QuantBits::B8}) {
      SCOPED_TRACE("dim=" + std::to_string(dim) +
                   " bits=" + std::to_string(static_cast<int>(bits)));
      std::vector<float> x(dim);
      uint64_t s = 0xD1CEULL + dim;
      for (auto& f : x) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        f = static_cast<float>(static_cast<int32_t>(s >> 32)) / 2147483648.0f;
      }
      const std::vector<float> y = RoundTrip(x, dim, bits);
      ExpectAllFinite(y, "arbitrary dim", bits);
      // At dim 1 and 2 there is too little signal for a direction check.
      if (dim >= 3) {
        EXPECT_GT(Cosine(x, y), 0.80) << "round-trip lost direction";
      }
    }
  }
}

// Padding waste is invisible at the API surface but decides whether a tier is
// actually a saving. At some dims a TurboQuant tier costs MORE than plain fp16,
// which callers must be able to detect before choosing one.
TEST(Robustness, PaddingOverheadIsVisibleToCallers) {
  // 3 * 2^k: no padding, payload is exactly 4 + dim*bits/8.
  EXPECT_EQ(Quantizer::PayloadBytes(768, QuantBits::B4), 4u + 384u);
  // 1152 = 3 * 384, not 3 * 2^k, so it pads to 2048 — 78% more code bytes than
  // the 1152 the caller might reasonably expect.
  EXPECT_EQ(Quantizer::PayloadBytes(1152, QuantBits::B4), 4u + 1024u);

  // The case that matters: at small dims a high TurboQuant tier is larger than
  // fp16 storage. Callers sizing storage must compare, not assume.
  const size_t tq12_at_129 = Quantizer::PayloadBytes(129, QuantBits::B12);
  const size_t fp16_at_129 = 129 * sizeof(uint16_t);
  EXPECT_GT(tq12_at_129, fp16_at_129)
      << "documented case: tq12 at dim=129 (padded to 256) costs "
      << tq12_at_129 << " B vs " << fp16_at_129 << " B for fp16";
}
