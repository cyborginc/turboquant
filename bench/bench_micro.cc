#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "turboquant/turboquant.h"

namespace {

using turboquant::QuantBits;
using turboquant::Quantizer;

std::vector<float> RandomVec(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0, 1);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

constexpr size_t kDim = 128;

void BM_Quant(benchmark::State& state, QuantBits bits) {
  Quantizer q(kDim, bits, /*seed=*/0xCAFE);
  auto x = RandomVec(kDim, 1);
  std::vector<uint8_t> payload(q.payload_bytes());
  for (auto _ : state) {
    q.Quantize(x.data(), 1, payload.data());
    benchmark::DoNotOptimize(payload);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * kDim * sizeof(float));
}

void BM_Dequant(benchmark::State& state, QuantBits bits) {
  Quantizer q(kDim, bits, /*seed=*/0xCAFE);
  auto x = RandomVec(kDim, 1);
  std::vector<uint8_t> payload(q.payload_bytes());
  q.Quantize(x.data(), 1, payload.data());
  std::vector<float> out(kDim);
  for (auto _ : state) {
    q.Dequantize(payload.data(), 1, out.data());
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * kDim * sizeof(float));
}

#define REGISTER_ALL(name, fn)                                                    \
  BENCHMARK_CAPTURE(fn, name##_b1, QuantBits::B1);                                \
  BENCHMARK_CAPTURE(fn, name##_b2, QuantBits::B2);                                \
  BENCHMARK_CAPTURE(fn, name##_b3, QuantBits::B3);                                \
  BENCHMARK_CAPTURE(fn, name##_b4, QuantBits::B4);                                \
  BENCHMARK_CAPTURE(fn, name##_b6, QuantBits::B6);                                \
  BENCHMARK_CAPTURE(fn, name##_b8, QuantBits::B8);                                \
  BENCHMARK_CAPTURE(fn, name##_b12, QuantBits::B12);

REGISTER_ALL(Quant, BM_Quant)
REGISTER_ALL(Dequant, BM_Dequant)

}  // namespace

BENCHMARK_MAIN();
