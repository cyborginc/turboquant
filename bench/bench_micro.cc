#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "turboquant/turboquant.h"

namespace {

using turboquant::Dequantize;
using turboquant::PayloadSize;
using turboquant::QuantBits;
using turboquant::Quantize;
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

#define REGISTER_ALL(name, fn)                                                    \
  BENCHMARK_CAPTURE(fn, name##_b1, QuantBits::B1);                                \
  BENCHMARK_CAPTURE(fn, name##_b2, QuantBits::B2);                                \
  BENCHMARK_CAPTURE(fn, name##_b4, QuantBits::B4);                                \
  BENCHMARK_CAPTURE(fn, name##_b6, QuantBits::B6);                                \
  BENCHMARK_CAPTURE(fn, name##_b8, QuantBits::B8);                                \
  BENCHMARK_CAPTURE(fn, name##_b12, QuantBits::B12);

REGISTER_ALL(Quant, BM_Quant)
REGISTER_ALL(Dequant, BM_Dequant)

}  // namespace

BENCHMARK_MAIN();
