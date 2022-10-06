#pragma once

#include <array>
#include <complex>
#include <cstdint>

// Prefix "G" = Griday
using GInt = int64_t;
using GReal = float;
using GComplex = std::complex<GReal>;

using GIndex = std::int64_t;
using GIndex3 = std::array<GIndex, 3>;

namespace constant {

const GReal    PI  = 3.14159265358979323846264338328;
const GReal    E   = 2.71828182845904523536028747135;
const GComplex I   = {static_cast<GReal>(0.0),
                      static_cast<GReal>(1.0)};
const GComplex ONE = {static_cast<GReal>(1.0),
                      static_cast<GReal>(0.0)};

}
