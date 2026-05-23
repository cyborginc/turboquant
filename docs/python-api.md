# Python API design for the bit-width parameter

> Context: turboquant supports a discrete set of quantization bit widths
> (1, 2, 3, 4, 6, 8, 12). Other values aren't on the Pareto frontier and aren't
> implemented. The C++ API uses a `QuantBits` enum which makes this natural.
> The Python bindings need to expose the same "discrete supported set" without
> feeling gappy to users who'd expect `bits=4` to just work.

## The genre convention

Other Python quantization libraries already handle this well, and users coming
from them won't be surprised:

| library | parameter | supported values |
|---|---|---|
| FAISS | `nbits=` (int) | mostly `{4, 8}`, errors otherwise |
| bitsandbytes | separate `quantize_4bit` / `quantize_8bit` functions | 4 or 8 |
| GPTQ | `bits=` (int) | `{2, 3, 4, 8}`, gaps at 5/6/7 |
| AutoAWQ | `w_bit=` (int) | `{4}` only |

Discrete supported sets with int parameters are the norm. We just need a clear
error path and IDE-friendly typing.

## Recommended solve

A two-layer approach: accept `int` at runtime with a clear error; use
`typing.Literal` in the stub for write-time IDE help.

### Runtime (PyBind11 wrapper)

```cpp
// turboquant_py.cc
namespace py = pybind11;

static const std::set<int> kSupportedBits = {1, 2, 3, 4, 6, 8, 12};

static turboquant::QuantBits ParseBits(int bits) {
  if (!kSupportedBits.count(bits)) {
    int nearest = *std::min_element(
        kSupportedBits.begin(), kSupportedBits.end(),
        [bits](int a, int b) {
          return std::abs(a - bits) < std::abs(b - bits);
        });
    throw py::value_error(
        "bits=" + std::to_string(bits) + " is not supported. "
        "Supported values: {1, 2, 3, 4, 6, 8, 12}. "
        "Closest available: " + std::to_string(nearest) + ".");
  }
  return static_cast<turboquant::QuantBits>(bits);
}

PYBIND11_MODULE(turboquant, m) {
  // Accept int directly; validate to the supported set with a helpful error.
  m.def("quantize",
        [](const turboquant::Rotator& rot, int bits, py::array_t<float> x) {
          auto qb = ParseBits(bits);
          // ... allocate payload, call turboquant::Quantize, return as bytes.
        },
        py::arg("rot"), py::arg("bits"), py::arg("x"));

  // Same for dequantize, payload_size, etc.

  // Also expose the enum for users who prefer constants.
  py::enum_<turboquant::QuantBits>(m, "QuantBits")
      .value("B1", turboquant::QuantBits::B1)
      .value("B2", turboquant::QuantBits::B2)
      .value("B3", turboquant::QuantBits::B3)
      .value("B4", turboquant::QuantBits::B4)
      .value("B6", turboquant::QuantBits::B6)
      .value("B8", turboquant::QuantBits::B8)
      .value("B12", turboquant::QuantBits::B12);
}
```

### Type stub (`turboquant.pyi`)

```python
from typing import Literal, Union
import numpy as np

# Literal narrows the accepted ints to the supported set at type-check time.
# Editors (pyright, mypy, VS Code Pylance) flag bits=3 wait, bits=5 etc. with
# a red squiggle BEFORE the call runs.
BitWidth = Literal[1, 2, 3, 4, 6, 8, 12]

class Rotator:
    def __init__(self, dim: int, seed: int = 0) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def padded_dim(self) -> int: ...

class QuantBits:
    B1: "QuantBits"
    B2: "QuantBits"
    B3: "QuantBits"
    B4: "QuantBits"
    B6: "QuantBits"
    B8: "QuantBits"
    B12: "QuantBits"

def payload_size(dim: int, bits: BitWidth) -> int: ...

def quantize(
    rot: Rotator,
    bits: BitWidth,
    x: np.ndarray,  # float32, shape (dim,)
) -> bytes: ...

def dequantize(
    rot: Rotator,
    bits: BitWidth,
    payload: bytes,
) -> np.ndarray: ...
```

### What this looks like to the user

```python
import turboquant as tq
import numpy as np

rot = tq.Rotator(dim=768, seed=42)
x = np.random.randn(768).astype(np.float32)

# Natural int — works.
payload = tq.quantize(rot, bits=4, x=x)

# Constant style — also works.
payload = tq.quantize(rot, bits=tq.QuantBits.B4, x=x)

# Mistyped value — IDE flags it as a Literal mismatch before runtime.
payload = tq.quantize(rot, bits=5, x=x)
# Pylance: Argument of type "Literal[5]" cannot be assigned to parameter "bits"
#         of type "Literal[1, 2, 3, 4, 6, 8, 12]"

# If they bypass the type checker, runtime gives a helpful error:
# ValueError: bits=5 is not supported. Supported values: {1, 2, 3, 4, 6, 8, 12}.
# Closest available: 4.
```

## Why not just support every bit width?

Considered and rejected. Filling in `b5, b7, b9, b10, b11, b13, b14, b15` would
require:

1. **Generic bitstream pack/unpack** (we have specialized u32/u64-load paths
   for the byte-aligned and B3/B6/B12 widths; arbitrary widths would need a
   generic path that runs ~50% slower).
2. **One Lloyd-Max codebook per `(bits, dim)`** — at b15 with 32,768 levels,
   construction is multiple seconds per Rotator.
3. **Decode tables grow** — at b15 the centroid table is 128 KB and stops
   fitting in L1, slowing Beta decode by ~3×.
4. **Test/bench bloat** — every new width adds ~10 rows to every benchmark
   sweep and ~5 new test cases.

And the recall payoff is small: looking at the data, every "missing" width
sits in a flat region of the recall curve (e.g., b7 between b6 at 0.98 and
b8 at 0.99 — no user can tell them apart). None is on the Pareto frontier.

The only "real gap" was b3 (between b2 at 0.83 cosine recall and b4 at 0.95),
and we landed it.

## Why hide the affine-vs-Beta choice?

The public `Quantize`/`Dequantize` auto-route between two internal schemes
based on the bit width: Lloyd-Max Beta-codebook for b1/b2/b3/b4/b6 (where it
dramatically improves recall), affine min/max for b8/b12 (where the schemes
are equivalent and affine is cheaper). The user never has to make this
algorithm choice — the bit width is the only knob.

This is by design. Surfacing the choice would let users pick "Beta at b8"
(strictly worse: same recall, 3× slower encode) or "affine at b1" (recall
collapses to 21% cosine on Wiki, 0.6% euclidean — a footgun). One knob,
no foot-gun configurations.

The internal `QuantizeAffine` / `QuantizeBeta` helpers exist in `src/internal.h`
for the test/bench harness — not exposed to Python.
