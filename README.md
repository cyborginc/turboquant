# turboquant

A C++17 library implementing the TurboQuant raw-vector quantization scheme.
The public surface is a single class (`turboquant::Quantizer`) that picks
the rotation and quantization paths appropriate for the requested
`(dim, bits)` at construction time.

`turboquant.h` is the only public header. Hot paths use
[Google Highway](https://github.com/google/highway) with dynamic dispatch, so
a single binary picks the best SIMD target at runtime (SSE4 / AVX2 /
AVX-512 / NEON / SVE / ...).

## Quick example

```cpp
#include "turboquant/turboquant.h"

turboquant::Quantizer q(/*dim=*/768, turboquant::QuantBits::B4);

std::vector<uint8_t> payloads(q.payload_bytes() * n);
q.Quantize(x.data(), n, payloads.data());

std::vector<float> recon(768 * n);
q.Dequantize(payloads.data(), n, recon.data());
```

`Quantize`/`Dequantize` are batched (`n = 1` is the single-vector case),
zero-allocation on the hot path, and safe to call concurrently against the
same `Quantizer` from multiple threads.

## Benchmarks — wiki-all-1M, d=768, 1M base × 1k queries, recall@100

Apple-silicon laptop, Accelerate BLAS (multi-threaded); quant/dequant
single-threaded. Numbers are the auto-routed `Quantizer` path
(mixed-radix + Beta at b1–b6, mixed-radix + affine at b8/b12).
`distance` = full dequantize + sgemm. Full sweep with each
rotation/codebook variant separately in `docs/bench_mixed_radix.txt`.

### Cosine (wiki_all_1M_cosine)

```
              recall@100                          distance    (ms, lower=better)
fp32  ████████████████████ 1.0000              ▏       515    (1.00x baseline)
fp16  ████████████████████ 0.9996              ███▎    859    (1.67x)
b12   ████████████████████ 0.9993              █████▍ 1393    (2.71x)
b8    ████████████████████ 0.9952              █████  1283    (2.49x)
b6    ███████████████████▋ 0.9844              █████  1315    (2.55x)
b4    ██████████████████▊  0.9415              ████▉  1254    (2.44x)
b3    █████████████████▉   0.8939              █████  1309    (2.54x)
b2    ████████████████▏    0.8102              ████▉  1256    (2.44x)
b1    ████████████▉        0.6437              █████  1285    (2.50x)
```

### Euclidean (wiki_all_1M)

```
              recall@100                          distance    (ms, lower=better)
fp32  ████████████████████ 1.0000              ▏       500    (1.00x baseline)
fp16  ████████████████████ 0.9998              ██▍     614    (1.23x)
b12   ████████████████████ 0.9994              █████▎ 1339    (2.68x)
b8    ████████████████████ 0.9955              ████▉  1239    (2.48x)
b6    ███████████████████▋ 0.9821              █████▏ 1294    (2.59x)
b4    ██████████████████▉  0.9439              ████▉  1248    (2.49x)
b3    █████████████████▊   0.8871              █████▏ 1308    (2.62x)
b2    ███████████████▏     0.7595              ████▉  1251    (2.50x)
b1    ███████▎             0.3657              █████  1278    (2.55x)
```

### Footprint

```
              bytes/vec                        compression vs fp32
fp32  ████████████████████  3072               ▏              1.00x
fp16  ██████████            1536               ▏              2.00x
b12   ███████▌              1156               ▎              2.66x
b8    █████                  772               ▍              3.98x
b6    ███▊                   580               ▌              5.30x
b4    ██▌                    388               ▊              7.92x
b3    █▉                     292               █              10.52x
b2    █▎                     196               █▌             15.67x
b1    ▋                      100               ███            30.72x
```

### What to read into these numbers

- 1 bit is workable on cosine (64% recall@100) but collapses on Euclidean
  (37%) — the norm is gone after binarization, so L2 ranking loses what
  cosine ranking didn't need.
- 4 bits gets ≥ 94% recall@100 at ~8× compression vs fp32.
- 8 bits is within 0.5pp of fp32 recall at ~4× compression and is the
  drop-in fp16 replacement when you want half the bytes again.
- `dequant + sgemm` is slower than fp32 sgemm — the ~0.8 s per-batch
  dequant dominates. The win is footprint: useful when the float base
  doesn't fit in RAM/cache, in streaming / per-shard scoring, or as a
  rerank stage inside an ANN index.

## Auto-routing

`Quantizer` picks two things automatically:

- **Rotation.** For `dim = 3·2^k` (common embedding sizes 768, 1536, 3072) a
  mixed-radix orthogonal rotation operates on the unpadded vector — ~25%
  smaller payloads than the padded path. Otherwise: zero-pad to the next
  power of two, sign-flip, Walsh-Hadamard.
- **Quantization.** For `bits ∈ {1,2,3,4,6}` a Lloyd-Max codebook tuned to
  the rotated coordinate distribution (`Beta((d-1)/2, (d-1)/2)`). For
  `bits ∈ {8,12}` uniform-step affine, where the codebook no longer helps.

Supported bit widths: `B1, B2, B3, B4, B6, B8, B12`. Codebook construction
is ~10–200 ms at d=768; encode/decode are zero-allocation after that.

## Payload format

```
offset  size    field
0       4       scale (IEEE-754 float32, little-endian)
4       N       packed codes, N = ceil(code_dim * bits / 8)
```

`code_dim = dim` on the mixed-radix path, `next_pow2(dim)` on the padded
path. Codes are a little-endian bitstream: code `i` at bits
`[i*bits, (i+1)*bits)`. `Quantizer::PayloadBytes(dim, bits)` returns the
total.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler. Required: Google Highway
(linked from the system, otherwise fetched at configure time). Optional:
GoogleTest, google-benchmark, HDF5 + BLAS (for the dataset bench; macOS
uses Accelerate automatically).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
./build/turboquant_bench
./build/turboquant_dataset_bench datasets/glove-25-angular.hdf5 10 1000
```
