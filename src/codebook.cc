#include "codebook.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>

namespace turboquant {
namespace {

// Unnormalized Beta(a, a) PDF on [-1, 1]: f̃(x) = (1 - x^2)^(a-1).
// The normalization constant cancels in every ratio of integrals we compute
// here, so we don't bother forming it. log1p keeps the exponent finite at high
// a when x is near 0.
double BetaPDFUnnorm(double x, double a_minus_1) {
  if (x <= -1.0 || x >= 1.0) return 0.0;
  const double log_f = a_minus_1 * std::log1p(-x * x);
  return std::exp(log_f);
}

template <class F>
double SimpsonStep(double lo, double hi, double f_lo, double f_hi,
                   double f_mid) {
  return (hi - lo) / 6.0 * (f_lo + 4.0 * f_mid + f_hi);
}

template <class F>
double AdaptiveSimpsonRec(const F& g, double lo, double hi, double f_lo,
                          double f_hi, double f_mid, double whole, double tol,
                          int depth) {
  const double mid = 0.5 * (lo + hi);
  const double m1 = 0.5 * (lo + mid);
  const double m2 = 0.5 * (mid + hi);
  const double f_m1 = g(m1);
  const double f_m2 = g(m2);
  const double left = SimpsonStep<F>(lo, mid, f_lo, f_mid, f_m1);
  const double right = SimpsonStep<F>(mid, hi, f_mid, f_hi, f_m2);
  const double refined = left + right;
  if (depth <= 0 || std::abs(refined - whole) < 15.0 * tol) {
    return refined + (refined - whole) / 15.0;
  }
  return AdaptiveSimpsonRec(g, lo, mid, f_lo, f_mid, f_m1, left, 0.5 * tol,
                            depth - 1) +
         AdaptiveSimpsonRec(g, mid, hi, f_mid, f_hi, f_m2, right, 0.5 * tol,
                            depth - 1);
}

template <class F>
double AdaptiveSimpson(const F& g, double lo, double hi, double tol) {
  const double mid = 0.5 * (lo + hi);
  const double f_lo = g(lo);
  const double f_hi = g(hi);
  const double f_mid = g(mid);
  const double whole = SimpsonStep<F>(lo, hi, f_lo, f_hi, f_mid);
  return AdaptiveSimpsonRec(g, lo, hi, f_lo, f_hi, f_mid, whole, tol, 50);
}

}  // namespace

BetaCodebook::BetaCodebook(QuantBits bits, size_t padded_dim) {
  const size_t num_levels = 1ull << static_cast<int>(bits);
  // Beta(a, a) on [-1, 1] where a = (padded_dim - 1) / 2 is the half-degrees-
  // of-freedom of a unit vector in R^padded_dim projected onto one coordinate.
  //
  // The density is bounded only for padded_dim >= 3 (a >= 1). At padded_dim 1
  // the distribution is degenerate (the coordinate is always +/-1) and at 2 it
  // is arcsine — both diverge at the endpoints, which the adaptive integrator
  // below cannot resolve: its tolerance test never passes, so it recurses to
  // the depth limit on every subinterval and takes ~2^50 evaluations. Neither
  // dim carries a meaningful distribution to fit, so clamp to the smallest
  // well-posed case. This leaves every padded_dim >= 3 codebook unchanged.
  const double a =
      0.5 * (static_cast<double>(std::max<size_t>(padded_dim, 3)) - 1.0);
  const double a_m1 = a - 1.0;

  auto pdf = [a_m1](double x) { return BetaPDFUnnorm(x, a_m1); };
  auto xpdf = [a_m1](double x) { return x * BetaPDFUnnorm(x, a_m1); };

  // Initial centroids spread within ±3 std. Var(Beta(a,a) on [-1,1]) = 1/(2a+1).
  const double std_dev = 1.0 / std::sqrt(2.0 * a + 1.0);
  const double spread = std::min(0.9, 3.0 * std_dev);

  std::vector<double> centroids(num_levels);
  if (num_levels == 1) {
    centroids[0] = 0.0;
  } else {
    for (size_t i = 0; i < num_levels; ++i) {
      const double t =
          static_cast<double>(i) / static_cast<double>(num_levels - 1);
      centroids[i] = -spread + 2.0 * spread * t;
    }
  }

  constexpr int kMaxIter = 200;
  constexpr double kTol = 1e-10;
  std::vector<double> new_centroids(num_levels);
  std::vector<double> edges(num_levels + 1);
  for (int iter = 0; iter < kMaxIter; ++iter) {
    edges[0] = -1.0;
    for (size_t i = 0; i + 1 < num_levels; ++i) {
      edges[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
    }
    edges[num_levels] = 1.0;

    for (size_t i = 0; i < num_levels; ++i) {
      const double lo = edges[i];
      const double hi = edges[i + 1];
      const double prob = AdaptiveSimpson(pdf, lo, hi, 1e-14);
      if (prob < 1e-30) {
        new_centroids[i] = centroids[i];
      } else {
        const double mom = AdaptiveSimpson(xpdf, lo, hi, 1e-14);
        new_centroids[i] = mom / prob;
      }
    }

    double max_change = 0.0;
    for (size_t i = 0; i < num_levels; ++i) {
      max_change =
          std::max(max_change, std::abs(new_centroids[i] - centroids[i]));
    }
    std::swap(centroids, new_centroids);
    if (max_change < kTol) break;
  }

  // Beta(a, a) is symmetric around 0; Lloyd-Max iteration should preserve this
  // but floating-point error can let small asymmetries accumulate. Force
  // exact symmetry — important for the branch-free encode below, which relies
  // on it.
  if (num_levels >= 2) {
    for (size_t i = 0; i < num_levels / 2; ++i) {
      const double mag =
          0.5 * (centroids[num_levels - 1 - i] - centroids[i]);
      centroids[i] = -mag;
      centroids[num_levels - 1 - i] = mag;
    }
  }

  // Final boundaries are the midpoints between consecutive centroids.
  boundaries_.resize(num_levels > 0 ? num_levels - 1 : 0);
  for (size_t i = 0; i + 1 < num_levels; ++i) {
    boundaries_[i] = 0.5f * static_cast<float>(centroids[i] + centroids[i + 1]);
  }
  centroids_.resize(num_levels);
  for (size_t i = 0; i < num_levels; ++i) {
    centroids_[i] = static_cast<float>(centroids[i]);
  }

  // Positive-half boundaries, padded to a power of two with +inf at the tail
  // so a branch-free binary search can compare against any index in [0, half)
  // without bounds checks.
  //
  // For num_levels = 2^bits, the middle boundary at index num_levels/2 - 1 is
  // exactly 0 (by symmetry) and is implicit in the encode — we work with
  // |x| against the strictly-positive boundaries
  //   boundaries[num_levels/2 ... num_levels - 2]
  // which has length num_levels/2 - 1. Pad to num_levels/2 with +inf.
  if (num_levels >= 2) {
    const size_t half = num_levels / 2;
    pos_bounds_pad_.assign(half, std::numeric_limits<float>::infinity());
    for (size_t i = 0; i + 1 < half; ++i) {
      pos_bounds_pad_[i] = boundaries_[half + i];
    }
  } else {
    pos_bounds_pad_.assign(1, std::numeric_limits<float>::infinity());
  }
}

// ---------------------------------------------------------------------------
// BetaCodebookCache
// ---------------------------------------------------------------------------

// Slots are indexed by the raw bit value, so the array spans [0, 12] inclusive.
// Index 0 is unused (no 0-bit width) but keeps the indexing arithmetic trivial.
struct BetaCodebookCache::State {
  explicit State(size_t d) : dim(d) {
    for (auto& slot : slots) slot.store(nullptr, std::memory_order_relaxed);
  }

  size_t dim;
  std::mutex mu;
  std::array<std::atomic<const BetaCodebook*>, 13> slots;
  // Owns what `slots` points at. unique_ptr elements mean a vector reallocation
  // never invalidates a published pointer.
  std::vector<std::unique_ptr<BetaCodebook>> owned;
};

BetaCodebookCache::BetaCodebookCache(size_t codebook_dim)
    : state_(std::make_unique<State>(codebook_dim)) {}

BetaCodebookCache::~BetaCodebookCache() = default;
BetaCodebookCache::BetaCodebookCache(BetaCodebookCache&&) noexcept = default;
BetaCodebookCache& BetaCodebookCache::operator=(BetaCodebookCache&&) noexcept =
    default;

const BetaCodebook* BetaCodebookCache::Get(QuantBits bits) const {
  const int bi = static_cast<int>(bits);
  if (bi <= 0 || static_cast<size_t>(bi) >= state_->slots.size()) {
    return nullptr;
  }
  // Fast path: already built. Acquire pairs with the release store below, so
  // the fully-constructed codebook is visible to this thread.
  if (const BetaCodebook* cb =
          state_->slots[bi].load(std::memory_order_acquire)) {
    return cb;
  }
  std::lock_guard<std::mutex> lock(state_->mu);
  // Re-check: another thread may have built this width while we waited.
  if (const BetaCodebook* cb =
          state_->slots[bi].load(std::memory_order_relaxed)) {
    return cb;
  }
  state_->owned.push_back(std::make_unique<BetaCodebook>(bits, state_->dim));
  const BetaCodebook* cb = state_->owned.back().get();
  state_->slots[bi].store(cb, std::memory_order_release);
  return cb;
}

}  // namespace turboquant
