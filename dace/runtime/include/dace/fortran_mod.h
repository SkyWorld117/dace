// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_FORTRAN_MOD_H
#define __DACE_FORTRAN_MOD_H

// Mixed-kind operands for Fortran MOD / MODULO.
//
// ``math.h`` defines ``cpp_mod`` (truncated remainder: C ``%`` / ``fmod``) and ``floor_mod``
// (floored remainder: Fortran MODULO) for ONE type.  A Fortran expression routinely mixes kinds
// -- ``i % n`` with a 64-bit loop index against a 32-bit bound, or ``x - MODULO(x, real(n))`` --
// and template argument deduction then finds no candidate ("no matching function for call to
// 'cpp_mod(int64_t&, int&)'").  These forward to the same-type form over the usual arithmetic
// conversions, which is what a Fortran compiler does with the operands.
//
// Kept out of ``math.h`` on purpose: that file carries an uncommitted local patch in this fork,
// and anything that touches it makes every downstream merge of this branch conflict.

template<typename T1, typename T2, std::enable_if_t<!std::is_same<T1, T2>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI auto cpp_mod(const T1& numerator, const T2& denominator)
    -> decltype(numerator + denominator) {
    using T = decltype(numerator + denominator);
    return cpp_mod<T>((T)numerator, (T)denominator);
}

template<typename T1, typename T2, std::enable_if_t<!std::is_same<T1, T2>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI auto floor_mod(const T1& numerator, const T2& denominator)
    -> decltype(numerator + denominator) {
    using T = decltype(numerator + denominator);
    return floor_mod<T>((T)numerator, (T)denominator);
}

#endif  // __DACE_FORTRAN_MOD_H
