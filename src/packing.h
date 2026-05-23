#ifndef TURBOQUANT_SRC_PACKING_H_
#define TURBOQUANT_SRC_PACKING_H_

#include <cstddef>
#include <cstdint>

#include "turboquant/turboquant.h"

namespace turboquant {

// Pack `n` codes (each value fits in `bits` bits) into `out`.
// Layout is the little-endian bitstream described in the spec:
//   code i occupies bit positions [i*bits, (i+1)*bits) in the byte array,
//   with bit 0 of byte 0 being the lowest-order bit.
void PackCodes(const uint16_t* codes, size_t n, QuantBits bits, uint8_t* out);

// Inverse of PackCodes. B6 and B12 use an unrolled u64-load path (8 codes per
// iter for B6, 4 codes per iter for B12) with a scalar tail for the last
// group so we never read past the packed buffer.
void UnpackCodes(const uint8_t* in, size_t n, QuantBits bits, uint16_t* codes);

}  // namespace turboquant

#endif  // TURBOQUANT_SRC_PACKING_H_
