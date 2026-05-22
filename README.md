# turboquant

A C++17 library implementing the TurboQuant raw-vector quantization scheme.
Three operations are exposed:

- `Quantize(rot, bits, x, payload)` — rotate + quantize a float vector to a
  packed per-vector payload.
- `Dequantize(rot, bits, payload, x)` — recover an approximation of the
  original float vector.
- `AdcScore(rot, bits, q_rot, payload)` — compute approximate dot product and
  decoded squared L2 norm directly from packed codes, without materializing
  floats.

`turboquant.h` is the only public header. Internally the hot paths use
[Google Highway](https://github.com/google/highway) with dynamic dispatch, so a
single binary picks the best SIMD target at runtime (SSE4 / AVX2 / AVX-512 /
NEON / SVE / ...).

## Layout

```
include/turboquant/turboquant.h   Public API
src/turboquant.cc                 API glue, payload write
src/rotation.{h,cc}               Walsh-Hadamard + sign-flip rotation (SIMD)
src/kernels.{h,cc}                Quant/dequant/ADC inner loops (SIMD)
src/packing.{h,cc}                Bit packing per the spec
tests/                            GoogleTest unit tests
bench/bench_micro.cc              google-benchmark microbenchmarks
bench/bench_dataset.cc            HDF5 dataset recall + speed bench
CMakeLists.txt
```

## TurboQuant rotation

Each `Rotator(dim, seed)` defines an orthogonal transform

```
y = H * D * pad(x)
```

where `pad` zero-extends `x` to the next power-of-two `padded_dim`, `D` is a
deterministic diagonal of `{-1, +1}` values seeded from `seed`, and `H` is the
normalized Walsh-Hadamard transform. The transform preserves L2 norm and inner
products (`<x, y> = <H D pad(x), H D pad(y)>`), so the rotated coordinates can
be quantized independently with much better behavior than the raw axes.

`Quantize` and `RotateQuery` apply the forward rotation. `Dequantize` applies
the inverse (`D * H * y`) and crops to the original `dim`. `AdcScore` operates
directly on the rotated codes, so the caller must pre-rotate the query once via
`RotateQuery`.

## Payload format

Per the spec:

```
offset  size    field
0       4       scale (IEEE-754 float32, little-endian)
4       12      reserved/padding (zero)
16      N       packed codes, N = ceil(padded_dim * bits / 8)
```

Codes are a little-endian bitstream: code `i` occupies bits `[i*bits, (i+1)*bits)`
with bit 0 of byte 0 being the lowest-order bit. `PayloadSize(dim, bits)`
returns the total byte count.

Supported bit widths: 1, 2, 4, 6, 8, 12. All non-1-bit modes use the affine
scheme `code = round(rotated_x / scale) + (1 << (bits-1))`, with `scale =
max_abs(rotated_x) / ((1 << (bits-1)) - 1)`. The 1-bit mode uses the symmetric
`code = (rotated_x >= 0) ? 1 : 0`, with `scale = max_abs(rotated_x)`.

## Build

Requires CMake >= 3.16 and a C++17 compiler. Dependencies:

- Google Highway (required). Linked from the system if found; otherwise fetched
  via `FetchContent` at configure time.
- GoogleTest (optional, for unit tests).
- google-benchmark (optional, for microbenchmarks).
- HDF5 + BLAS (optional, for the dataset bench).

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

constexpr size_t kDim = 128;
turboquant::Rotator rot(kDim, /*seed=*/0xCAFE);

// One-time per vector: pack into a contiguous payload.
std::vector<uint8_t> payload(turboquant::PayloadSize(kDim, turboquant::QuantBits::B8));
turboquant::Quantize(rot, turboquant::QuantBits::B8, x.data(), payload.data());

// One-time per query: rotate into the same space the codes live in.
std::vector<float> q_rot(rot.padded_dim());
turboquant::RotateQuery(rot, q.data(), q_rot.data());

// Hot loop: score the query against many quantized payloads.
auto stats = turboquant::AdcScore(rot, turboquant::QuantBits::B8, q_rot.data(),
                                  payload.data());
// stats.dot           ~= <q, x>
// stats.decoded_norm2 ~= <x_hat, x_hat>
// l2_squared          = ||q||^2 + stats.decoded_norm2 - 2 * stats.dot
// cosine              = stats.dot * (1 / ||q||) * rsqrt(stats.decoded_norm2)
```

## Dataset benchmark

`turboquant_dataset_bench <file.hdf5> [k=10] [max_test=1000]` loads an
[ann-benchmarks](https://ann-benchmarks.com/)-style file (datasets `/train`,
`/test`, `/neighbors`) and reports:

- Ingest throughput (Quantize vs. float32 memcpy baseline).
- Read throughput (Dequantize vs. float32 memcpy baseline).
- Distance throughput for three paths, per bit width:
  - **f32 (BLAS)**: one big `sgemm(test, train^T)` on the original floats.
  - **ADC**: per-pair `AdcScore` against the packed codes.
  - **deq+BLAS**: dequantize the whole base to float32 once, then `sgemm`.
- Recall@K vs. the float32 brute-force ground truth, per bit width.

The bench infers the metric from a `distance` HDF5 attribute or, failing that,
from the filename (`*-angular.hdf5` → cosine, anything else → L2).

Note: OpenBLAS sgemm is multi-threaded; the ADC path is single-threaded. Cap
BLAS to one thread with `OPENBLAS_NUM_THREADS=1` for an apples-to-apples
single-core comparison.

## Sample runs

Hardware: 4-core x86-64 with AVX2 (Ubuntu 24.04, gcc 13). OpenBLAS uses all
cores; the ADC path is single-threaded.

### glove-25-angular — 200 queries × 1.18M base vectors, recall@10

```
path               ms_total       Mops/s   recall@10  rel_vs_f32
f32 (BLAS)           134.76      1756.43     1.0000        1.00x
b1  (ADC)          39127.20         6.05     0.0120      290.34x
b1  (deq+BLAS)       427.33       553.91     0.0120        3.17x  (257ms deq + 171ms gemm)
b2  (ADC)          15801.47        14.98     0.0675      117.25x
b2  (deq+BLAS)       373.97       632.94     0.0675        2.78x  (251ms deq + 123ms gemm)
b4  (ADC)          11558.38        20.48     0.6305       85.77x
b4  (deq+BLAS)       365.46       647.68     0.6305        2.71x  (243ms deq + 122ms gemm)
b6  (ADC)          26613.67         8.89     0.8955      197.48x
b6  (deq+BLAS)       452.84       522.71     0.8955        3.36x  (301ms deq + 152ms gemm)
b8  (ADC)           9229.37        25.65     0.9705       68.49x
b8  (deq+BLAS)       357.58       661.96     0.9705        2.65x  (242ms deq + 116ms gemm)
b12 (ADC)          29622.42         7.99     0.9975      219.81x
b12 (deq+BLAS)       446.24       530.43     0.9975        3.31x  (324ms deq + 122ms gemm)
```

### sift-128-euclidean — 100 queries × 1.00M base vectors, recall@10

```
path               ms_total       Mops/s   recall@10  rel_vs_f32
f32 (BLAS)           139.66       716.00     1.0000        1.00x
b1  (ADC)          61980.22         1.61     0.0030      443.78x
b1  (deq+BLAS)       820.27       121.91     0.0030        5.87x  (693ms deq + 127ms gemm)
b2  (ADC)          17174.55         5.82     0.0830      122.97x
b2  (deq+BLAS)       875.29       114.25     0.0830        6.27x  (747ms deq + 128ms gemm)
b4  (ADC)          12702.95         7.87     0.7960       90.95x
b4  (deq+BLAS)       853.11       117.22     0.7960        6.11x  (692ms deq + 161ms gemm)
b6  (ADC)          38728.50         2.58     0.9420      277.30x
b6  (deq+BLAS)      1111.15        90.00     0.9420        7.96x  (970ms deq + 141ms gemm)
b8  (ADC)          11067.09         9.04     0.9840       79.24x
b8  (deq+BLAS)       843.69       118.53     0.9840        6.04x  (708ms deq + 136ms gemm)
b12 (ADC)          46565.39         2.15     0.9990      333.41x
b12 (deq+BLAS)      1173.32        85.23     0.9990        8.40x  (1045ms deq + 129ms gemm)
```

### What to read into these numbers

- **Recall is identical between ADC and deq+BLAS at every bit width**, which
  is the correctness signal: both paths compute the same dot product (one in
  the rotated space, one after materializing floats) so the quantization
  error — not the kernel — is what loses recall.
- **Quality vs. size tradeoff** is the headline: 4-bit retains 63% (glove-25)
  / 80% (sift-128) of recall@10 at 8× compression, 6-bit ≥ 89/94%, 8-bit
  ≥ 97/98% at 4× compression, 12-bit ≥ 99.7% at ~2.7× compression.
- **`deq+BLAS` is several times slower than `f32 (BLAS)`** because the
  per-batch dequant step adds ~0.7s on SIFT-1M before the matmul even
  starts. The trade is memory footprint: the quantized base is 4×–32×
  smaller than the float32 base.
- **Current ADC is much slower than `deq+BLAS`** in this brute-force
  scoring setup. The reason is that the Highway dynamic-dispatch
  thunk fires on every call and each call only does
  `padded_dim` lanes of work (32 lanes on glove, 128 on SIFT), so per-call
  overhead dominates. The natural fix is a batched ADC API (one rotated query
  × many packed payloads per call) which would amortize the dispatch and let
  the inner loop run a tight SIMD reduction — that wasn't part of this
  iteration but is the obvious next step if ADC ends up on a hot path.
- **The point of ADC isn't to beat sgemm at brute-force**; it's that the
  base never has to be materialized as floats. ADC is the natural kernel for
  cases where the float base doesn't fit in RAM/cache, for streaming /
  per-shard scoring, or for rerank-on-quantized inside an ANN index.
