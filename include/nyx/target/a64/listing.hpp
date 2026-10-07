#pragma once

#include "nyx/ir/ssa/listing.hpp"

namespace nyx::a64 {

// X0-X30, SP, NZCV, TPIDR_EL0 and Q0-Q31 by name, with the storage AAPCS64
// lets a callee or the caller observe. Flags and X9-X17 are neither
// arguments, results, preserved nor the platform's, so a write to them is
// dead once control leaves. This rests on the declared calling convention, as
// the listing says.
ir::SsaListingTarget ListingTarget();

}  // namespace nyx::a64
