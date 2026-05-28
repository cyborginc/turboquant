# turboquant

A C++17 library implementing the TurboQuant raw-vector quantization scheme.
The public surface is a single class — `turboquant::Quantizer` — which picks
both the rotation scheme and the quantization scheme appropriate for the
requested `(dim, bits)` at construction time:

- `Quantizer(dim, bits, seed=0)` — builds the rotation and (for low bit
  widths) a Lloyd-Max codebook tuned to the rotated coordinate distribution.
- `q.Quantize(x, n, payloads)` — rotate + quantize `n` contiguous row-major
  vectors into `n` contiguous packed payloads. `n = 1` is the single-vector
  case. Zero allocation on the hot path.
- `q.Dequantize(payloads, n, x_out)` — recover float approximations of the
  original vectors.
- `Quantizer::PayloadBytes(dim, bits)` — the size of one payload; computable
  without constructing.

`turboquant.h` is the only public header. The hot paths use
[Google Highway](https://github.com/google/highway) with dynamic dispatch, so
a single binary picks the best SIMD target at runtime (SSE4 / AVX2 /
AVX-512 / NEON / SVE / ...).

`Quantize`/`Dequantize` are safe to call concurrently from multiple threads
against the same `Quantizer` (per-thread scratch buffers are used
internally).

## Layout

```
include/turboquant/turboquant.h   Public API (Quantizer)
src/turboquant.cc                 API glue + routing
src/rotator_padded.h              Padded WHT rotation (general dim)
src/rotator_mixed.{h,cc}          Mixed-radix rotation for dim = 3 * 2^k
src/rotation.{h,cc}               WHT + sign-flip kernels (SIMD)
src/codebook.{h,cc}               Lloyd-Max Beta codebook
src/kernels.{h,cc}                Quant/dequant inner loops (SIMD)
src/packing.{h,cc}                Bit packing per the spec
tests/                            GoogleTest unit tests
bench/bench_micro.cc              google-benchmark microbenchmarks
bench/bench_dataset.cc            HDF5 dataset recall + speed bench
CMakeLists.txt
```

## Routing

`Quantizer` picks two things automatically; the caller never has to know
which path was taken.

**Rotation.** For `dim = 3 * 2^k` (covers common embedding sizes 768, 1536,
3072), it uses a *mixed-radix* orthogonal rotation that operates on the
unpadded vector — the payload is ~25% smaller than the padded path. For all
other `dim`, it zero-pads to the next power of two and applies a
sign-flip + Walsh-Hadamard rotation:

```
y = H * D * pad(x)
```

`D` is a deterministic ±1 diagonal seeded from `seed`, and `H` is the
normalized Walsh-Hadamard transform. The transform preserves L2 norm and
inner products, so the rotated coordinates can be quantized independently
with much better behavior than the raw axes.

**Quantization.** For `bits ∈ {1, 2, 3, 4, 6}` it uses a *Beta* codebook:
the rotated unit vector follows `Beta((d-1)/2, (d-1)/2)` on `[-1, 1]`, so a
Lloyd-Max codebook computed against that distribution is near-optimal. For
`bits ∈ {8, 12}` the codebook advantage vanishes and it falls back to
uniform-step affine quantization (`code = round(x/scale) + 2^(b-1)`,
`scale = max_abs / (2^(b-1) - 1)`). The 1-bit affine path is the symmetric
`code = (x >= 0)` case with `scale = max_abs`.

Codebook construction takes ~10-200 ms at d=768 depending on bit width.
After construction the encode/decode paths are zero-allocation per call.

## Payload format

```
offset  size    field
0       4       scale (IEEE-754 float32, little-endian)
4       N       packed codes, N = ceil(code_dim * bits / 8)
```

`code_dim` is `dim` on the mixed-radix path and `next_pow2(dim)` on the
padded path. Codes are a little-endian bitstream: code `i` occupies bits
`[i*bits, (i+1)*bits)`, with bit 0 of byte 0 the lowest-order bit.
`Quantizer::PayloadBytes(dim, bits)` returns the total byte count.

Supported bit widths: `B1, B2, B3, B4, B6, B8, B12`.

## Build

Requires CMake >= 3.16 and a C++17 compiler. Dependencies:

- Google Highway (required). Linked from the system if found; otherwise
  fetched via `FetchContent` at configure time.
- GoogleTest (optional, for unit tests).
- google-benchmark (optional, for microbenchmarks).
- HDF5 + BLAS (optional, for the dataset bench; on macOS BLAS is provided
  by the system Accelerate framework).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build           # unit tests
./build/turboquant_bench         # microbench
./build/turboquant_dataset_bench datasets/glove-25-angular.hdf5 10 1000
```

On Ubuntu 24.04 a one-line install of the deps is:

```
sudo apt-get install libhwy-dev libgtest-dev libbenchmark-dev libhdf5-dev libopenblas-dev
```

CMake options:
- `-DTURBOQUANT_BUILD_TESTS=OFF` skip unit tests
- `-DTURBOQUANT_BUILD_BENCH=OFF` skip microbench
- `-DTURBOQUANT_BUILD_DATASET_BENCH=OFF` skip HDF5 bench

## Quick example

```cpp
#include "turboquant/turboquant.h"

constexpr size_t kDim = 768;
turboquant::Quantizer q(kDim, turboquant::QuantBits::B4);

// Encode a batch of n vectors into a contiguous payload buffer.
std::vector<uint8_t> payloads(q.payload_bytes() * n);
q.Quantize(x.data(), n, payloads.data());

// Decode them back to float.
std::vector<float> recon(kDim * n);
q.Dequantize(payloads.data(), n, recon.data());
```

## Dataset benchmark

`turboquant_dataset_bench <file.hdf5> [k=10] [max_test=1000]` loads an
[ann-benchmarks](https://ann-benchmarks.com/)-style file (datasets `/train`,
`/test`, `/neighbors`) and reports, per bit width and per encode variant:

- **Ingest throughput** vs. an `fp32` memcpy and an `fp16` cast baseline.
- **Read throughput** (full-pass dequantize) vs. the same baselines.
- **Distance throughput** for the *dequantize + sgemm* path vs. a direct
  `sgemm` on the original floats — and `recall@K` of the resulting top-K
  against the float32 brute-force ground truth.

The bench exposes each rotation/codebook variant separately
(`affine`, `beta`, `mix3 bet`, `mix3 aff`) so the head-to-head between them
is visible. The public `Quantizer` always selects one of these
automatically.

The metric is inferred from a `distance` HDF5 attribute or, failing that,
from the filename (`*-angular.hdf5` → cosine, anything else → L2).

Note: when linked against OpenBLAS, `sgemm` is multi-threaded; the
dequantize step is single-threaded. Cap BLAS to one thread with
`OPENBLAS_NUM_THREADS=1` for an apples-to-apples single-core comparison.

## Sample run

`wiki_all_1M_cosine.hdf5` — d=768, 1M base × 1k queries, recall@100, on an
Apple-silicon laptop (Accelerate BLAS, multi-threaded; quant/dequant
single-threaded). Bit widths abridged to `b2 / b4 / b8`; full sweep in
`docs/bench_mixed_radix.txt`.

```
[Ingest: quantize train set]               MB/s_quantized
fp32 memcpy                                       74790
fp16 cast                                         36262
b4  affine                                          439
b4  beta                                            223
b4  mix3 (beta, unpadded)                           226   (payload 75% of padded)

[Distance: 1000 q x 1M base, recall@100]   ms_total   recall@K   rel_vs_f32
f32 (BLAS)                                  515         1.0000      1.00x
fp16+BLAS                                   859         0.9996      1.67x
b2  beta+BLAS                              1630         0.8292      3.17x
b2  mix3 beta+BLAS                         1256         0.8102      2.44x
b4  beta+BLAS                              1612         0.9500      3.13x
b4  mix3 beta+BLAS                         1254         0.9415      2.44x
b8  affine+BLAS                            1495         0.9942      2.91x
b8  mix3 aff+BLAS                          1283         0.9952      2.49x
```

### What to read into these numbers

- **The Beta codebook dominates at low bit widths.** At b2 on a cosine
  dataset, affine quantization keeps only ~21% recall@100; the Beta
  codebook keeps ~83% at the same 2 bits/dim. By b8 the two paths converge
  (both ≥ 99%), which is why `Quantizer` switches to plain affine there —
  no point paying for a codebook that doesn't help.
- **The mixed-radix rotation is the better default at d=768.** It avoids
  padding 768 → 1024, so payloads are ~25% smaller *and* dequantize is
  faster (less data through the WHT). Recall is within rounding of the
  padded path at every bit width.
- **Quality vs. size:** 4-bit retains ≥ 94% recall@100 at 8× compression
  vs. fp32 (or 4× vs. fp16); 8-bit retains ≥ 99% at 4× / 2×.
- **Dequant+BLAS is several times slower than `f32 (BLAS)`** because the
  per-batch dequant step adds ~0.8 s on 1M × 768 before the matmul even
  starts. The trade is memory footprint: the quantized base is 4×–32×
  smaller than the float32 base, so the win shows up when the float base
  doesn't fit in RAM/cache, or in streaming / per-shard scoring.
