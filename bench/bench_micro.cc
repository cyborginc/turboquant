#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "turboquant/turboquant.h"

namespace {

using turboquant::AdcScore;
using turboquant::Dequantize;
using turboquant::PayloadSize;
using turboquant::QuantBits;
using turboquant::Quantize;
using turboquant::RotateQuery;
using turboquant::Rotator;

std::vector<float> RandomVec(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

constexpr size_t kDim = 128;

void BM_Quant(benchmark::State& state, QuantBits bits) {
  Rotator R(kDim, 0xCAFE);
  auto x = RandomVec(kDim, 1);
  std::vector<uint8_t> payload(PayloadSize(kDim, bits));
  for (auto _ : state) {
    Quantize(R, bits, x.data(), payload.data());
    benchmark::DoNotOptimize(payload);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * kDim * sizeof(float));
}

void BM_Dequant(benchmark::State& state, QuantBits bits) {
  Rotator R(kDim, 0xCAFE);
  auto x = RandomVec(kDim, 1);
  std::vector<uint8_t> payload(PayloadSize(kDim, bits));
  Quantize(R, bits, x.data(), payload.data());
  std::vector<float> out(kDim);
  for (auto _ : state) {
    Dequantize(R, bits, payload.data(), out.data());
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * kDim * sizeof(float));
}

// One quantized vector, one rotated query, many score calls (the hot path).
void BM_Adc(benchmark::State& state, QuantBits bits) {
  Rotator R(kDim, 0xCAFE);
  auto x = RandomVec(kDim, 1);
  auto q = RandomVec(kDim, 2);
  std::vector<uint8_t> payload(PayloadSize(kDim, bits));
  Quantize(R, bits, x.data(), payload.data());
  std::vector<float> q_rot(R.padded_dim());
  RotateQuery(R, q.data(), q_rot.data());
  for (auto _ : state) {
    auto s = AdcScore(R, bits, q_rot.data(), payload.data());
    benchmark::DoNotOptimize(s);
  }
  state.SetItemsProcessed(state.iterations());
}

// Sweep: score one query against `kBatch` quantized vectors.
constexpr size_t kBatch = 1000;

void BM_AdcBatch(benchmark::State& state, QuantBits bits) {
  Rotator R(kDim, 0xCAFE);
  auto q = RandomVec(kDim, 2);
  std::vector<float> q_rot(R.padded_dim());
  RotateQuery(R, q.data(), q_rot.data());

  const size_t ps = PayloadSize(kDim, bits);
  std::vector<uint8_t> payloads(ps * kBatch);
  std::mt19937_64 rng(7);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> x(kDim);
  for (size_t b = 0; b < kBatch; ++b) {
    for (size_t i = 0; i < kDim; ++i) x[i] = dist(rng);
    Quantize(R, bits, x.data(), payloads.data() + b * ps);
  }

  for (auto _ : state) {
    float total = 0;
    for (size_t b = 0; b < kBatch; ++b) {
      auto s = AdcScore(R, bits, q_rot.data(), payloads.data() + b * ps);
      total += s.dot;
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * kBatch);
}

// Float32 baseline dot product for comparison.
void BM_DotFloat(benchmark::State& state) {
  std::vector<float> a = RandomVec(kDim, 1);
  std::vector<float> b = RandomVec(kDim, 2);
  for (auto _ : state) {
    float s = 0;
    for (size_t i = 0; i < kDim; ++i) s += a[i] * b[i];
    benchmark::DoNotOptimize(s);
  }
  state.SetItemsProcessed(state.iterations());
}

#define REGISTER_ALL(name, fn)                                                    \
  BENCHMARK_CAPTURE(fn, name##_b1, QuantBits::B1);                                \
  BENCHMARK_CAPTURE(fn, name##_b2, QuantBits::B2);                                \
  BENCHMARK_CAPTURE(fn, name##_b4, QuantBits::B4);                                \
  BENCHMARK_CAPTURE(fn, name##_b6, QuantBits::B6);                                \
  BENCHMARK_CAPTURE(fn, name##_b8, QuantBits::B8);                                \
  BENCHMARK_CAPTURE(fn, name##_b12, QuantBits::B12);

REGISTER_ALL(Quant, BM_Quant)
REGISTER_ALL(Dequant, BM_Dequant)
REGISTER_ALL(Adc, BM_Adc)
REGISTER_ALL(AdcBatch1k, BM_AdcBatch)
BENCHMARK(BM_DotFloat);

}  // namespace

BENCHMARK_MAIN();
