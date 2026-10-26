// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_GLIBC_F32_H
#define __DACE_GLIBC_F32_H

// ---------------------------------------------------------------------------------------------
// Bit-exact device lowering of glibc's scalar FLOAT transcendentals.
//
// WHY THIS EXISTS.  The port's fidelity claim is "the dispatched answer equals the answer the
// stock (host) build produces".  That claim cannot hold for any transcendental while the device
// calls libdevice and the host calls glibc: the two disagree in the last bit on a large fraction
// of arguments (MEASURED 2026-10-21: tan 33.0%, exp 30.3%, cos 21.4%, sin 12.6%, ... 1-3 ulp),
// and glibc's float routines are NOT correctly rounded (sinf is 1.30% off by 1 ulp from the
// correctly-rounded result), so no correctly-rounded device routine can ever agree with them.
// The only way to agree is to reproduce glibc's own algorithm, in double, rounding once.
//
// `dace::math::sin(float)` already does this for `sinf` (see math.h).  This header adds the rest.
//
// THE SWITCH.  Everything here is behind DACE_GLIBC_F32 and is therefore OFF unless the bake
// passes `-DDACE_GLIBC_F32=1`.  Two reasons it is opt-in rather than the default:
//   1. It changes the numeric result of every dispatched kernel that uses a transcendental, so
//      turning it on re-bases every recorded md5 -- a decision, not a detail.
//   2. It only makes the port bit-identical when the HOST has a single reference too.  The stock
//      build is compiled with `-ftree-vectorize` (configure.wrf FCOPTIM), so its call sites bind a
//      MIX: scalar libm and libmvec `_ZGVbN4v_*` (MEASURED: 2,298 libmvec transcendental sites
//      across 104 objects; module_diffusion_em.o alone has 32 `_ZGVbN4vv_powf`, 7 `expf`, 6 `cosf`).
//      libmvec is a different algorithm, and the host disagrees WITH ITSELF (scalar sinf vs
//      `_ZGVbN4v_sinf` differ on 13.92% of args).  Reproducing that from the device is not
//      possible -- the device cannot know which call site the host would have taken.  So the
//      reference arm is built with `-fno-tree-vectorize` (see verify-d34/tree_scalar), giving ONE
//      host reference; with this header the two become bit-identical.
//
// Validated the same way as the sinf patch: a host transcription is checked against the real libm
// symbol over an exhaustive sweep of all 2^32 float inputs (0 differing required), and the device
// form is the same arithmetic with explicit `fma`.  See /scratch/zyi/wsmoke/glibcf32/<fn>/.
// ---------------------------------------------------------------------------------------------

#include "math.h"
#include "glibc_tanhf.h"

#if !defined(DACE_GLIBC_F32)
// Off: nothing is added and every `dace::math::<fn>` keeps its existing (libdevice) lowering.
#else

namespace dace
{
    namespace math
    {





















        static __constant__ unsigned long long dace_glibc_expf_tab[32] = {
            0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
            0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
            0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
            0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
            0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
            0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
            0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
            0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
            0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
            0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
            0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL
        };
        static __device__ const unsigned dace_tanf_two_over_pi[24] = {
            0x000000a2u, 0x0000a2f9u, 0x00a2f983u, 0xa2f9836eu,
            0xf9836e4eu, 0x836e4e44u, 0x6e4e4415u, 0x4e441529u,
            0x441529fcu, 0x1529fc27u, 0x29fc2757u, 0xfc2757d1u,
            0x2757d1f5u, 0x57d1f534u, 0xd1f534ddu, 0xf534ddc0u,
            0x34ddc0dbu, 0xddc0db62u, 0xc0db6295u, 0xdb629599u,
            0x6295993cu, 0x95993c43u, 0x993c4390u, 0x3c439041u,
        };

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

        // Bit pattern of a float, without pulling in a header that the device pass may not have.
        DACE_DFI unsigned dace_glibc_f32_bits(float f)
        {
#ifdef __CUDA_ARCH__
            return __float_as_uint(f);
#else
            unsigned u;
            __builtin_memcpy(&u, &f, 4);
            return u;
#endif
        }




        // -------------------------------------------------------------------------------------
        /* ------------------------------------------------------------------
         * Bit-exact device lowering of glibc 2.34's float `atanf`.
         *
         * glibc 2.34 exports atanf as a plain WEAK FUNC (no IFUNC) at
         * libm.so.6:0x3b400, 622 bytes.  Unlike sinf, it is SINGLE PRECISION
         * throughout (mulss/addss/subss/divss only; no cvtss2sd) and contains
         * NO fused multiply-add.  The routine below is that binary
         * instruction-for-instruction: ifunc-free generic SSE variant,
         * reduction (atanlo/atanhi @0x91b90/0x91ba0) + one float polynomial.
         *
         * Because nvcc contracts p*w+c into FFMA by default, every op is
         * written with __fmul_rn/__fadd_rn/__fsub_rn/__fdiv_rn, which are
         * never fused, preserving glibc's exact rounding.  Host C build is
         * -ffp-contract=off for the same reason.
         *
         * MEASURED: exhaustive over all 2^32 float bit patterns against the
         * real host libm atanf through a volatile function pointer:
         *   n_samples = 4294967296, n_differ = 0, max_ulp = 0.
         * ------------------------------------------------------------------ */
        DACE_DFI float dace_glibc_atanf(float a)
        {
            const unsigned eax = __float_as_uint(a);
            const unsigned ix  = eax & 0x7fffffffu;
            float xu = a;
            int id;

            if (ix > 0x4bffffffu)                       /* |x| >= 2^26 / inf / nan */
            {
                if (ix > 0x7f800000u)                   /* nan -> x + x */
                    return __fadd_rn(a, a);
                if (eax & 0x80000000u)                  /* -atanhi[3] - atanlo[3] */
                    return __fsub_rn(__uint_as_float(0xbfc90fdau),
                                     __uint_as_float(0x33a22168u));
                return __fadd_rn(__uint_as_float(0x33a22168u),
                                 __uint_as_float(0x3fc90fdau));
            }

            if (ix > 0x3edfffffu)                       /* |x| >= 0.4375 */
            {
                xu = __uint_as_float(ix);               /* |x| */
                float den = xu;
                if (ix > 0x3f97ffffu)                   /* |x| >= 1.1875 */
                {
                    if (ix > 0x401bffffu)               /* id = 3 */
                    {
                        id = 3;
                        xu = __fdiv_rn(-1.0f, den);
                    }
                    else                                /* id = 2 */
                    {
                        id = 2;
                        den = __fmul_rn(1.5f, den);
                        xu  = __fsub_rn(xu, 1.5f);
                        den = __fadd_rn(den, 1.0f);
                        xu  = __fdiv_rn(xu, den);
                    }
                }
                else if (ix > 0x3f2fffffu)              /* id = 1 */
                {
                    id = 1;
                    xu  = __fsub_rn(xu, 1.0f);
                    den = __fadd_rn(den, 1.0f);
                    xu  = __fdiv_rn(xu, den);
                }
                else                                    /* id = 0 */
                {
                    id = 0;
                    xu  = __fadd_rn(xu, xu);            /* 2|x| */
                    den = __fadd_rn(den, 2.0f);
                    xu  = __fsub_rn(xu, 1.0f);
                    xu  = __fdiv_rn(xu, den);
                }
            }
            else
            {
                if (ix <= 0x30ffffffu)                  /* |x| < 2^-29 -> x */
                    return a;
                id = -1;
            }

            const float z = __fmul_rn(xu, xu);
            const float w = __fmul_rn(z, z);

            float p = __uint_as_float(0x3c8569d7u);
            p = __fadd_rn(__fmul_rn(p, w), __uint_as_float(0x3d4bda59u));
            p = __fadd_rn(__fmul_rn(p, w), __uint_as_float(0x3d886b35u));
            p = __fadd_rn(__fmul_rn(p, w), __uint_as_float(0x3dba2e6eu));
            p = __fadd_rn(__fmul_rn(p, w), __uint_as_float(0x3e124925u));
            p = __fadd_rn(__fmul_rn(p, w), __uint_as_float(0x3eaaaaabu));
            const float s1 = __fmul_rn(z, p);

            float s2 = __uint_as_float(0xbd15a221u);
            s2 = __fsub_rn(__fmul_rn(s2, w), __uint_as_float(0x3d6ef16bu));
            s2 = __fsub_rn(__fmul_rn(s2, w), __uint_as_float(0x3d9d8795u));
            s2 = __fsub_rn(__fmul_rn(s2, w), __uint_as_float(0x3de38e38u));
            s2 = __fsub_rn(__fmul_rn(s2, w), __uint_as_float(0x3e4ccccdu));
            s2 = __fmul_rn(s2, w);

            float xp = __fmul_rn(xu, __fadd_rn(s1, s2));

            if (id == -1)
                return __fsub_rn(xu, xp);

            const float alo = __uint_as_float(id == 0 ? 0x31ac3769u
                                        : id == 1 ? 0x33222168u
                                        : id == 2 ? 0x33140fb4u
                                                  : 0x33a22168u);
            const float ahi = __uint_as_float(id == 0 ? 0x3eed6338u
                                        : id == 1 ? 0x3f490fdau
                                        : id == 2 ? 0x3f7b985eu
                                                  : 0x3fc90fdau);
            xp = __fsub_rn(xp, alo);
            xp = __fsub_rn(xp, xu);
            float res = __fsub_rn(ahi, xp);
            if (eax >> 31)
                res = __uint_as_float(__float_as_uint(res) ^ 0x80000000u);
            return res;
        }

        /* Non-template exact match wins overload resolution for float operands,
           so every generated device dace::math::atan(<float32>) uses the
           glibc-exact routine above. */
        DACE_DFI float atan(const float& a)
        {
            return dace_glibc_atanf(a);
        }

        /* Bit-exact device lowering of glibc 2.34 x86-64 float `cosf`.
         *
         * The host binds the multiarch IFUNC `cosf` to __cosf_fma
         * (libm.so.6 0x6e390; confirmed at runtime with dlsym/dladdr).  That routine
         * converts to double, range-reduces with round(x*2/pi), evaluates the shared
         * sincosf polynomial in DOUBLE and rounds once on return -- it is NOT correctly
         * rounded, so no CRT/libm cosf can match it.  This is a literal transcription of
         * that implementation (constants, branch thresholds and FMA fusion order), plus
         * its Payne-Hanek path for |x| >= 120 and its NaN/Inf convention.
         * VERIFIED exhaustively on the host: all 2^32 bit patterns, 0 differing. */
        DACE_DFI unsigned dace_cosf_bits(float f)
        {
#ifdef __CUDA_ARCH__
            return __float_as_uint(f);
#else
            unsigned u;
            __builtin_memcpy(&u, &f, 4);
            return u;
#endif
        }

        DACE_DFI float dace_cosf_from_bits(unsigned u)
        {
#ifdef __CUDA_ARCH__
            return __uint_as_float(u);
#else
            float f;
            __builtin_memcpy(&f, &u, 4);
            return f;
#endif
        }

        DACE_DFI float dace_glibc_cosf(float y)
        {
            /* sincosf table (libm.so.6 .rodata @0xa7ac0); table[1] negates c0..c4,
               s1/s2/s3 are shared.  HPI_INV = 2/pi * 2^24, HPI = pi/2. */
            const double HPI_INV = 0x1.45f306dc9c883p+23;   /* 10680707.430881744 */
            const double HPI     = 0x1.921fb54442d18p+0;    /* 1.5707963267948966 */
            const double KS      = 0x1.921fb54442d18p-62;   /* 3.4061215800865545e-19 */
            const double C0 =  1.0;
            const double C1 = -0x1.ffffffd0c621cp-2;
            const double S1 = -0x1.555545995a603p-3;
            const double C2 =  0x1.55553e1068f19p-5;
            const double S2 =  0x1.1107605230bc4p-7;
            const double C3 = -0x1.6c087e89a359dp-10;
            const double S3 = -0x1.994eb3774cf24p-13;
            const double C4 =  0x1.99343027bf8c3p-16;
            /* 2/pi bit stream (.rodata @0xa7a60). */
            const unsigned PH[24] = {
                0x000000a2u, 0x0000a2f9u, 0x00a2f983u, 0xa2f9836eu,
                0xf9836e4eu, 0x836e4e44u, 0x6e4e4415u, 0x4e441529u,
                0x441529fcu, 0x1529fc27u, 0x29fc2757u, 0xfc2757d1u,
                0x2757d1f5u, 0x57d1f534u, 0xd1f534ddu, 0xf534ddc0u,
                0x34ddc0dbu, 0xddc0db62u, 0xc0db6295u, 0xdb629599u,
                0x6295993cu, 0x95993c43u, 0x993c4390u, 0x3c439041u
            };

            unsigned u = dace_cosf_bits(y);
            unsigned b = (u >> 20) & 0x7ffu;
            double x = (double)y;

            if (b <= 0x3f3u)                        /* |y| < 0.6875 */
            {
                if (b <= 0x397u) return 1.0f;       /* |y| < 2^-12 */
                double z = x * x;
                double x4 = z * z;
                double e  = fma(C1, z, C0);
                double f_ = fma(C4, z, C3);
                double x6 = z * x4;
                double g  = fma(x4, C2, e);
                return (float)fma(f_, x6, g);
            }
            else if (b <= 0x42eu)                   /* 0.6875 <= |y| < 120 */
            {
                int t = (int)(x * HPI_INV);         /* cvttsd2si */
                int n = (t + 0x800000) >> 24;       /* round(x*2/pi) */
                double dn = (double)n;
                double r = fma(-dn, HPI, x);        /* vfnmadd: x - n*(pi/2) */
                double z = r * r;
                if ((n & 1) == 0)
                {
                    bool neg = (n & 2) != 0;
                    double c0 = neg ? -C0 : C0;
                    double c1 = neg ? -C1 : C1;
                    double c2 = neg ? -C2 : C2;
                    double c3 = neg ? -C3 : C3;
                    double c4 = neg ? -C4 : C4;
                    double x4 = z * z;
                    double e  = fma(c1, z, c0);
                    double f_ = fma(c4, z, c3);
                    double x6 = z * x4;
                    double g  = fma(x4, c2, e);
                    return (float)fma(f_, x6, g);
                }
                else
                {
                    int si = n & 3;                 /* sign[4] = {1,-1,-1,1} */
                    double rs = r * ((si == 0 || si == 3) ? 1.0 : -1.0);
                    double p  = fma(S3, z, S2);
                    double t3 = rs * z;
                    double t2 = z * t3;
                    double t1 = fma(t3, S1, rs);
                    return (float)fma(p, t2, t1);
                }
            }
            else if (b <= 0x7f7u)                   /* 120 <= |y| < 2^128 */
            {
                /* Payne-Hanek: the same 96-bit 2/pi window product as the host. */
                int sign = (int)(u >> 31);
                int esi  = (int)((u >> 26) & 0xfu);
                int sh   = (int)((u >> 23) & 0x7u);
                unsigned w = ((u & 0x7fffffu) | 0x800000u) << sh;
                unsigned long long t0 = PH[esi];
                unsigned long long t1 = PH[esi + 4];
                unsigned long long t2 = PH[esi + 8];
                unsigned lo = (unsigned)(t0 * (unsigned long long)w);
                unsigned long long rcx = ((t2 * (unsigned long long)w) >> 32)
                                       | ((unsigned long long)lo << 32);
                unsigned long long p = t1 * (unsigned long long)w + rcx;
                unsigned long long rr = p + 0x2000000000000000ull;
                unsigned long long top2 = rr >> 62;
                unsigned long long rem = p - (rr & 0xc000000000000000ull);
                double rd = (double)(long long)rem * KS;   /* signed vcvtsi2sd */
                double z = rd * rd;
                int edx = sign + (int)top2;
                if ((top2 & 1) == 0)
                {
                    bool neg = (edx & 2) != 0;
                    double c0 = neg ? -C0 : C0;
                    double c1 = neg ? -C1 : C1;
                    double c2 = neg ? -C2 : C2;
                    double c3 = neg ? -C3 : C3;
                    double c4 = neg ? -C4 : C4;
                    double x4 = z * z;
                    double e  = fma(c1, z, c0);
                    double f_ = fma(c4, z, c3);
                    double x6 = z * x4;
                    double g  = fma(x4, c2, e);
                    return (float)fma(f_, x6, g);
                }
                else
                {
                    int si = edx & 3;
                    double rs = rd * ((si == 0 || si == 3) ? 1.0 : -1.0);
                    double p  = fma(S3, z, S2);
                    double t3 = rs * z;
                    double t2 = z * t3;
                    double t1 = fma(t3, S1, rs);
                    return (float)fma(p, t2, t1);
                }
            }
            /* inf -> x86 indefinite qnan; nan -> quieted, payload preserved. */
            if ((u & 0x7fffffu) != 0)
                return dace_cosf_from_bits(u | 0x00400000u);
            return dace_cosf_from_bits(0xffc00000u);
        }

        /* Non-template exact match wins overload resolution for float operands. */
        DACE_DFI float cos(const float& a)
        {
            return dace_glibc_cosf(a);
        }

        /* ------------------------------------------------------------------
         * Bit-exact device lowering of glibc 2.34 x86-64 float `expf`.
         *
         * MEASURED (2026-10-22).  /lib64/libm.so.6 (glibc 2.34) exports expf as an
         * IFUNC; on this CPU (fma/avx2/avx512f) the resolver at 0x36690 binds the
         * FMA implementation at offset 0x6da60, NOT the SSE2 one at 0x36580.  The
         * routine below is transcribed instruction-for-instruction from 0x6da60:
         * it evaluates 2^(k/32) * P(r) with P in DOUBLE and rounds once to float
         * (cvtss2sd on entry, cvtsd2ss on exit).  Both kd and the reduced argument
         * r come from FUSED ops (vfmadd132sd / vfmsub132sd); the fusion order
         * matters.  Error is glibc's error, not correctly-rounded error.
         *
         * VALIDATED exhaustively on the host: all 2^32 float bit patterns, mine vs
         * the real libm expf behind a volatile function pointer -> n_differ = 0,
         * max_ulp = 0.  (The non-FMA SSE2 variant differs on 2 of 2^32 inputs by
         * 1 ulp, which is exactly the fusion difference this code reproduces.)
         *
         * nvcc -O2 -std=c++17 -arch=sm_90 -ptx: 5x fma.rn.f64, 0 calls, no local
         * memory (table lives in .const).
         * ------------------------------------------------------------------ */
        /* __exp2f_data.tab: 2^(i/32), exact bit patterns (libm .rodata @0xa7580). */


        DACE_DFI unsigned dace_glibc_expf_fbits(float f)
        {
#ifdef __CUDA_ARCH__
            return __float_as_uint(f);
#else
            unsigned u;
            __builtin_memcpy(&u, &f, 4);
            return u;
#endif
        }

        DACE_DFI float dace_glibc_expf_ffrom(unsigned u)
        {
#ifdef __CUDA_ARCH__
            return __uint_as_float(u);
#else
            float f;
            __builtin_memcpy(&f, &u, 4);
            return f;
#endif
        }

        DACE_DFI unsigned long long dace_glibc_expf_dbits(double d)
        {
#ifdef __CUDA_ARCH__
            return (unsigned long long)__double_as_longlong(d);
#else
            unsigned long long u;
            __builtin_memcpy(&u, &d, 8);
            return u;
#endif
        }

        DACE_DFI double dace_glibc_expf_dfrom(unsigned long long u)
        {
#ifdef __CUDA_ARCH__
            return __longlong_as_double((long long)u);
#else
            double d;
            __builtin_memcpy(&d, &u, 8);
            return d;
#endif
        }

        DACE_DFI float dace_glibc_expf(const float& a)
        {
            const double InvLn2N = 0x1.71547652b82fep+5;   /* 32/ln2 */
            const double SHIFT   = 0x1.8p52;               /* 1.5 * 2^52 */
            const double C1      = 0x1.62e42ff0c52d6p-6;   /* ln2/32 */
            const double C2      = 0x1.ebfce50fac4f3p-13;  /* (ln2/32)^2 / 2 */
            const double C3      = 0x1.c6af84b912394p-20;  /* (ln2/32)^3 / 6 */

            float    x      = a;
            unsigned bits   = dace_glibc_expf_fbits(x);
            unsigned abstop = (bits >> 20) & 0x7ffu;

            if (abstop > 0x42au)                   /* |x| >= 88, or inf/nan */
            {
                if (bits == 0xff800000u)           /* -inf -> +0 */
                    return 0.0f;
                if (abstop > 0x7f7u)               /* inf/nan -> x + x */
                    return x + x;
                if (x > 0x1.62e42ep6f)             /* > 88.7228317 -> overflow */
                    return dace_glibc_expf_ffrom(0x7f800000u);
                if (x < -0x1.9fe368p6f)            /* < -103.972076 -> +0 */
                    return 0.0f;
                if (x < -0x1.9d1d9ep6f)            /* < -103.278923 -> may_uflow */
                    return dace_glibc_expf_ffrom(0x00000001u);  /* FLT_TRUE_MIN */
                /* else: -103.278923 <= x, fall through to the fast path */
            }

            double   xd = (double)x;
            /* vfmadd132sd: shifted = x*InvLn2N + SHIFT */
            double   shifted = fma(xd, InvLn2N, SHIFT);
            unsigned long long rax = dace_glibc_expf_dbits(shifted);
            double   kd  = shifted - SHIFT;        /* vsubsd */
            unsigned idx = (unsigned)(rax & 0x1fu);
            unsigned long long top = rax << 47;
            /* vfmsub132sd: r = x*InvLn2N - kd */
            double   r   = fma(xd, InvLn2N, -kd);
            double   rr  = r * r;
            double   p   = fma(r, C3, C2);         /* c2 + c3*r  */
            double   q   = fma(r, C1, 1.0);        /* 1  + c1*r  */
            double   poly = fma(p, rr, q);         /* (c2+c3 r)r^2 + (1+c1 r) */
            double   scale = dace_glibc_expf_dfrom(top + dace_glibc_expf_tab[idx]);
            return (float)(poly * scale);          /* single rounding to float */
        }

        /* Non-template exact match wins overload resolution for float operands, so
           generated device dace::math::exp(<float32>) lowers to the glibc-exact routine. */
        DACE_DFI float exp(const float& a)
        {
            return dace_glibc_expf(a);
        }

/* ------------------------------------------------------------------
 * Bit-exact device lowering of glibc's float `logf`.
 *
 * MEASURED (2026-10-22).  The host build's transcendentals are glibc, and
 * glibc's `logf` is NOT correctly rounded, so CUDA's `logf` (or any other
 * CRT implementation) cannot agree with it bit-for-bit.  What does reproduce
 * it is glibc's own algorithm: logf evaluates in DOUBLE and rounds exactly
 * once to float (cvtss2sd on entry, cvtsd2ss on exit).
 *
 * Ground truth: libm.so.6 `logf@@GLIBC_2.27` is an IFUNC; on this CPU
 * (EPYC 9555, AVX2+FMA) the resolver binds libm.so.6+0x6dc90, the FMA
 * implementation (the SSE2 fallback is +0x39f20).  Verified at runtime with
 * dlsym + dladdr: dlsym(RTLD_DEFAULT,"logf") = libm.so.6 + 0x6dc90.
 * The routine below transcribes 0x6dc90 instruction for instruction,
 * including the five vfmadd fusion sites and the two's-complement bit tricks
 * used to split x = 2^k * z.
 *
 * VERIFIED exhaustively on the CPU transcription over ALL 2^32 float bit
 * patterns against the real glibc `logf` through a volatile function pointer:
 *   n_samples = 4,294,967,296   n_differ = 0   max_ulp = 0  -> BIT-EXACT
 * (special cases included: +-0 -> -inf, +inf -> +inf, negatives -> 0xffc00000
 *  invalid QNaN, NaN -> quieted NaN with payload preserved, subnormals.)
 * ------------------------------------------------------------------ */
        DACE_DFI unsigned dace_glibc_logf_bits(float f)
        {
#ifdef __CUDA_ARCH__
            return __float_as_uint(f);
#else
            unsigned u;
            __builtin_memcpy(&u, &f, 4);
            return u;
#endif
        }

        DACE_DFI float dace_glibc_logf_from_bits(unsigned u)
        {
#ifdef __CUDA_ARCH__
            return __uint_as_float(u);
#else
            float f;
            __builtin_memcpy(&f, &u, 4);
            return f;
#endif
        }

        /* glibc 0x64a50 (__math_invalidf): (x-x)/(x-x).  For a non-NaN this is
           0/0 = the x86 indefinite QNaN 0xffc00000; for a NaN the SSE ops return
           the same NaN with its quiet bit set and payload preserved. */
        DACE_DFI float dace_glibc_logf_invalid(float x)
        {
            unsigned u = dace_glibc_logf_bits(x);
            if ((u & 0x7fffffffu) > 0x7f800000u)
                return dace_glibc_logf_from_bits(u | 0x00400000u);
            return dace_glibc_logf_from_bits(0xffc00000u);
        }

        DACE_DFI float dace_glibc_logf(float a)
        {
            /* glibc's __logf_data (libm.so.6 .rodata @0xa76e0), {invc, logc}[16].
               static const -> read-only global array (ld.global.nc), no per-thread
               local copy. */
            static const double invc[16] = {
                0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0,
                0x1.49539f0f010b0p+0, 0x1.3c995b0b80385p+0,
                0x1.30d190c8864a5p+0, 0x1.25e227b0b8ea0p+0,
                0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0,
                0x1.0953f419900a7p+0, 0x1.0000000000000p+0,
                0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aa0p-1,
                0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1,
                0x1.886e6037841edp-1, 0x1.767dcf5534862p-1
            };
            static const double logc[16] = {
                -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2,
                -0x1.01eae7f513a67p-2, -0x1.b31d8a68224e9p-3,
                -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c8100p-3,
                -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4,
                -0x1.252f438e10c1ep-5,  0x0.0p+0,
                 0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
                 0x1.526e57720db08p-3,  0x1.bc2860d224770p-3,
                 0x1.1058bc8a07ee1p-2,  0x1.4043057b6ee09p-2
            };
            const double LN2 = 0x1.62e42fefa39efp-1;
            const double P1  =  0x1.5575b0be00b6ap-2;
            const double P2  = -0x1.ffffef20a4123p-2;
            const double P3  = -0x1.00ea348b88334p-2;

            unsigned ix = dace_glibc_logf_bits(a);
            unsigned eax;

            if (ix == 0x3f800000u)                  /* a == 1.0f -> +0.0f */
                return 0.0f;

            if ((unsigned)(ix - 0x00800000u) > 0x7effffffu)
            {
                unsigned ecx = ix + ix;
                if (ecx == 0)                       /* +-0 -> -inf */
                    return dace_glibc_logf_from_bits(0xff800000u);
                if (ix == 0x7f800000u)              /* +inf -> +inf */
                    return a;
                if (ecx > 0xfeffffffu)              /* -inf, nan */
                    return dace_glibc_logf_invalid(a);
                if ((int)ix < 0)                    /* negative finite/subnormal */
                    return dace_glibc_logf_invalid(a);
                /* positive subnormal: a *= 0x1p23 (exact), bias k by 23 */
                eax = dace_glibc_logf_bits(
                          a * dace_glibc_logf_from_bits(0x4b000000u)) - 0x0b800000u;
            }
            else
            {
                eax = ix;
            }

            unsigned edx = eax - 0x3f330000u;
            int      k   = (int)edx >> 23;              /* sar */
            unsigned idx = (edx >> 19) & 0xfu;          /* shr, then mask */
            unsigned esi = edx & 0xff800000u;
            double   z   = (double)dace_glibc_logf_from_bits(eax - esi);

            double t  = fma((double)k, LN2, logc[idx]); /* vfmadd132sd */
            double r  = fma(z, invc[idx], -1.0);        /* vfmadd132sd */
            double r2 = r * r;                          /* vmulsd      */
            double s  = r + t;                          /* vaddsd      */
            double p  = fma(r, P1, P2);                 /* vfmadd213sd */
            p = fma(r2, P3, p);                         /* vfmadd231sd */
            return (float)fma(r2, p, s);                /* vfmadd132sd; cvtsd2ss */
        }
        /* Non-template exact match wins overload resolution for float operands, so
           every generated device `dace::math::log(<float32>)` lowers to the
           glibc-exact routine.  (Paste below the existing `log<T>` template.) */
        DACE_DFI float log(const float& a)
        {
            return dace_glibc_logf(a);
        }

        /*
         * Bit-exact device lowering of glibc 2.34 x86-64 FLOAT `tanf`.
         *
         * Host ground truth: /lib64/libm.so.6, tanf@@GLIBC_2.2.5 at vaddr 0x3cd40
         * (weak, NOT an IFUNC -- unlike sincosf, the resolved entry on this CPU is
         * this scalar SSE body), with the shared __kernel_tanf at 0x3aff0.
         *
         * Every host arithmetic step is a separate mulss/addss/subss/divss or
         * mulsd/addsd/subsd/divsd: glibc's tanf contains NO fused multiply-add.
         * The device code therefore uses __f*_rn / __d*_rn intrinsics, which map to
         * non-contractible PTX (mul.rn.f32, add.rn.f32, ...).  Plain operators are
         * NOT safe here: nvcc's default -fmad=true would fuse a*b+c and change the
         * last bit.
         *
         * VERIFIED on host: exhaustive over all 2^32 bit patterns through a volatile
         * function pointer to glibc tanf -> n_differ = 0, max_ulp = 0.
         */
        /* 2/pi in 32-bit words (libm.so.6 .rodata @0xa7a60).  Declared in both the
           host and device passes (but only for CUDA/HIP compilers) so that nvcc can
           register the device symbol; plain g++ builds never see it. */
        #if defined(__CUDACC__) || defined(__HIPCC__)

        #endif


        DACE_DFI unsigned dace_tanf_bits(float f)
        {
            return __float_as_uint(f);
        }

        DACE_DFI float dace_tanf_from_bits(unsigned u)
        {
            return __uint_as_float(u);
        }

        /* libm.so.6 0x3aff0: __kernel_tanf(x, y, iy), fdlibm k_tanf.c */
        DACE_DFI float dace_glibc_tanf_kernel(float x, float y, int iy)
        {
            unsigned ecx = dace_tanf_bits(x) & 0x7fffffffu;
            float xmm6 = x;
            unsigned edx = dace_tanf_bits(x);
            int eax = iy;

            if ((int)ecx <= 0x38ffffff) {              /* |x| < 2^-14 */
                int esi = (int)x;                      /* cvttss2si */
                if (esi != 0)
                    goto poly;
                if (((unsigned)(iy + 1) | ecx) == 0) { /* iy == -1 && x == +-0 */
                    float ax = dace_tanf_from_bits(dace_tanf_bits(xmm6) & 0x7fffffffu);
                    return __fdiv_rn(1.0f, ax);
                }
                if (iy == 1) {
                    float ax = dace_tanf_from_bits(dace_tanf_bits(xmm6) & 0x7fffffffu);
                    if (dace_tanf_from_bits(0x00800000u) > ax) { /* raise underflow, value = x */
                        volatile float sink = __fmul_rn(ax, xmm6);
                        (void)sink;
                    }
                    return xmm6;
                }
                return __fdiv_rn(-1.0f, xmm6);
            }

            if ((int)ecx <= 0x3f2ca13f)
                goto poly;

            /* 0x3b040: |x| >= 0.674408, rebase around pi/4 */
            {
                if (edx & 0x80000000u) {
                    xmm6 = -xmm6;
                    y = -y;
                }
                {
                    float pio4   = dace_tanf_from_bits(0x3f490fdau);
                    float pio4lo = dace_tanf_from_bits(0x33222168u);
                    float a = __fsub_rn(pio4, xmm6);
                    float b = __fsub_rn(pio4lo, y);
                    xmm6 = __fadd_rn(b, a);            /* (pio4lo-y) + (pio4-x) */
                }
                {
                    float c = dace_tanf_from_bits(0x39000000u);   /* 2^-13 */
                    float s = dace_tanf_from_bits(dace_tanf_bits(xmm6) & 0x7fffffffu);
                    if (c > s) {                       /* 0x3b238, very near pi/4 */
                        int neg2 = ((edx >> 30) & 2u);
                        int f0   = 1 - neg2;
                        int p    = f0 * eax;
                        int two  = eax + eax;
                        float v2 = (float)two;
                        float v0 = (float)p;
                        float t  = __fmul_rn(v2, xmm6);
                        float om = __fsub_rn(1.0f, t);
                        return __fmul_rn(v0, om);
                    }
                    y = 0.0f;
                }
            }

        poly:
            {
                const float T0  = dace_tanf_from_bits(0x3eaaaaabu);
                const float T1  = dace_tanf_from_bits(0x3e088889u);
                const float T2  = dace_tanf_from_bits(0x3d5d0dd1u);
                const float T3  = dace_tanf_from_bits(0x3cb327a4u);
                const float T4  = dace_tanf_from_bits(0x3c11371fu);
                const float T5  = dace_tanf_from_bits(0x3b6b6916u);
                const float T6  = dace_tanf_from_bits(0x3abede48u);
                const float T7  = dace_tanf_from_bits(0x3a1a26c8u);
                const float T8  = dace_tanf_from_bits(0x398137b9u);
                const float T9  = dace_tanf_from_bits(0x38a3f445u);
                const float T10 = dace_tanf_from_bits(0x3895c07au);
                const float T11 = dace_tanf_from_bits(0xb79bae5fu);
                const float T12 = dace_tanf_from_bits(0x37d95384u);

                float z = __fmul_rn(xmm6, xmm6);       /* xmm3 */
                float s = __fmul_rn(xmm6, z);          /* xmm2 = x^3 */
                float w = __fmul_rn(z, z);             /* xmm5 = x^4 */
                float a, b;

                a = __fadd_rn(__fmul_rn(T12, w), T10);
                b = __fadd_rn(__fmul_rn(T11, w), T9);
                a = __fadd_rn(__fmul_rn(a, w), T8);
                b = __fadd_rn(__fmul_rn(b, w), T7);
                a = __fadd_rn(__fmul_rn(a, w), T6);
                b = __fadd_rn(__fmul_rn(b, w), T5);
                a = __fadd_rn(__fmul_rn(a, w), T4);
                b = __fadd_rn(__fmul_rn(b, w), T3);
                a = __fadd_rn(__fmul_rn(a, w), T2);
                b = __fadd_rn(__fmul_rn(b, w), T1);
                a = __fmul_rn(a, z);                   /* v chain * z */
                a = __fadd_rn(a, b);                   /* r + v */
                {
                    float q  = __fmul_rn(a, s);        /* s*(r+v) */
                    float t0 = __fmul_rn(s, T0);       /* T0*s */
                    float ww;
                    q = __fadd_rn(q, y);
                    q = __fmul_rn(q, z);
                    {
                        float yy = __fadd_rn(y, q);    /* y + z*(s*(r+v)+y) */
                        float rr = __fadd_rn(t0, yy);  /* final r */
                        ww = __fadd_rn(xmm6, rr);      /* w = x + r */
                        if ((int)ecx > 0x3f2ca13f) {   /* 0x3b1c0 */
                            float v   = (float)eax;
                            float sq  = __fmul_rn(ww, ww);
                            float den = __fadd_rn(ww, v);
                            float dv  = __fdiv_rn(sq, den);
                            float t2;
                            int neg2  = ((edx >> 30) & 2u);
                            int f0    = 1 - neg2;
                            dv = __fsub_rn(dv, rr);
                            t2 = __fsub_rn(xmm6, dv);
                            t2 = __fadd_rn(t2, t2);
                            v  = __fsub_rn(v, t2);
                            return __fmul_rn((float)f0, v);
                        }
                        if (eax == 1)
                            return ww;
                        /* 0x3b16e: iy == -1, reciprocal refinement */
                        {
                            float q0  = __fdiv_rn(-1.0f, ww);
                            float whi = dace_tanf_from_bits(dace_tanf_bits(ww) & 0xfffff000u);
                            float qhi = dace_tanf_from_bits(dace_tanf_bits(q0) & 0xfffff000u);
                            float d   = __fsub_rn(whi, xmm6);
                            float t1  = __fsub_rn(rr, d);
                            float t3;
                            t1 = __fmul_rn(t1, qhi);
                            t3 = __fmul_rn(whi, qhi);
                            t3 = __fadd_rn(t3, 1.0f);
                            t1 = __fadd_rn(t1, t3);
                            t1 = __fmul_rn(t1, q0);
                            return __fadd_rn(qhi, t1);
                        }
                    }
                }
            }
        }

        DACE_DFI float dace_glibc_tanf(float a)
        {
            unsigned u  = dace_tanf_bits(a);
            unsigned ix = u & 0x7fffffffu;

            if (ix <= 0x3f490fdau)                     /* |x| <= pi/4 */
                return dace_glibc_tanf_kernel(a, 0.0f, 1);

            if (ix <= 0x7f7fffffu) {                   /* finite */
                double dx = (double)a;
                unsigned top = (u >> 20) & 0x7ffu;
                if (top <= 0x42eu) {                   /* 0x3cd91: |x| < 120 */
                    const double c1 = 0x1.45f306dc9c883p+23; /* 2/pi * 2^24 */
                    const double c2 = 0x1.921fb54442d18p+0;  /* pi/2 */
                    int t = (int)__dmul_rn(dx, c1);
                    int n = (t + 0x800000) >> 24;
                    double r = __dsub_rn(dx, __dmul_rn((double)n, c2));
                    float hi = (float)r;
                    float lo = (float)__dsub_rn(r, (double)hi);
                    int iy = 1 - (((unsigned)n << 1) & 2u);
                    return dace_glibc_tanf_kernel(hi, lo, iy);
                } else {                               /* 0x3ce18: |x| >= 120 */
                    unsigned idx = (u >> 26) & 0xfu;
                    unsigned m   = (u & 0x7fffffu) | 0x800000u;
                    unsigned sh  = (u >> 23) & 7u;
                    unsigned m2  = m << sh;
                    unsigned long long p0  = (unsigned long long)m2 * (unsigned long long)dace_tanf_two_over_pi[idx];
                    unsigned long long p2  = (unsigned long long)m2 * (unsigned long long)dace_tanf_two_over_pi[idx + 8];
                    unsigned long long rax = ((unsigned long long)(unsigned)p0 << 32) | (p2 >> 32);
                    unsigned long long rcx = (unsigned long long)m2 * (unsigned long long)dace_tanf_two_over_pi[idx + 4];
                    unsigned long long rsi;
                    unsigned quad;
                    double r;
                    rcx += rax;
                    rsi = 0x2000000000000000ull + rcx;
                    rcx -= rsi & 0xc000000000000000ull;
                    quad = (unsigned)(rsi >> 62);
                    r = __dmul_rn((double)(long long)rcx, 0x1.921fb54442d18p-62);
                    if (u & 0x80000000u)
                        r = -r;
                    {
                        float hi = (float)r;
                        float lo = (float)__dsub_rn(r, (double)hi);
                        int iy = 1 - (((unsigned)quad << 1) & 2u);
                        return dace_glibc_tanf_kernel(hi, lo, iy);
                    }
                }
            }

            /* inf / nan: glibc evaluates x - x.  Encode x86 SSE's exact result
               instead of relying on the GPU's default-NaN convention:
                 +-inf -> 0xffc00000 (QNaN floating-point indefinite)
                 nan   -> input with the quiet bit set, sign/payload preserved. */
            if (ix == 0x7f800000u)
                return dace_tanf_from_bits(0xffc00000u);
            return dace_tanf_from_bits(u | 0x00400000u);
        }

        /* Non-template float overload wins overload resolution over
           template<class T> T tan(const T&), as for dace_glibc_sinf. */
        DACE_DFI float tan(const float& a)
        {
            return dace_glibc_tanf(a);
        }

/* ------------------------------------------------------------------
   Bit-exact device lowering of glibc 2.34's x86-64 FMA `powf`.
   (FUNCTION TO TRANSCRIBE: powf -- float, arity 2.)

   MEASURED GROUND TRUTH (2026-10-22).  On this host the default symbol
   `powf@@GLIBC_2.27` is an IFUNC; `nm -D` shows `i powf@@GLIBC_2.27`, and its
   resolver (libm.so.6 0x3a540) selects the FMA/VEX implementation at offset
   0x6ddb0 on this AMD EPYC 9555 (cpuid FMA + AVX2).  `/tmp/resolve_pow3`
   confirms `dlsym(RTLD_DEFAULT,"powf") - dladdr().dli_fbase == 0x6ddb0`.
   The SSE2 twin at 0x3a170 is the same algorithm, so this transcription is
   the FMA variant instruction-for-instruction.

   The routine therefore reproduces glibc's ERROR, not the correctly-rounded
   result: log2 via a 16-entry table + a degree-5 polynomial evaluated in
   DOUBLE with FMA, then exp2 via a 32-entry table + a degree-3 polynomial in
   DOUBLE, rounding to float exactly once at the end (cvtsd2ss).  The special
   -case lattice (0/inf/nan, negative bases and integer exponents, overflow at
   0x1.fffffffd1d571p+6, underflow at -150, may-underflow at -149) is kept
   verbatim.

   Tables are the exact .rodata bit patterns of libm.so.6: the log2 table at
   0xa7920 (16 x {invc,logc}) and the exp2 table at 0xa7580, which stores
   `asuint64(2^(i/32)) - (i << 47)` so that `ski << 47` can be added as an
   integer.  Constants: OFF=0x3f330000, SHIFT=0x1.8p47, SIGN_BIAS=1<<16.

   Verified against the real libm powf through a volatile function pointer:
   see the accompanying sweep (n_samples / n_differ / max_ulp reported with
   the transcription).  No libm calls, no long double, no std::.
   ------------------------------------------------------------------ */
        DACE_DFI unsigned dace_powf_u32(float f)
        {
#ifdef __CUDA_ARCH__
            return __float_as_uint(f);
#else
            unsigned u;
            __builtin_memcpy(&u, &f, 4);
            return u;
#endif
        }

        DACE_DFI float dace_powf_f32(unsigned u)
        {
#ifdef __CUDA_ARCH__
            return __uint_as_float(u);
#else
            float f;
            __builtin_memcpy(&f, &u, 4);
            return f;
#endif
        }

        DACE_DFI unsigned long long dace_powf_u64(double x)
        {
#ifdef __CUDA_ARCH__
            return (unsigned long long)__double_as_longlong(x);
#else
            unsigned long long u;
            __builtin_memcpy(&u, &x, 8);
            return u;
#endif
        }

        DACE_DFI double dace_powf_f64(unsigned long long u)
        {
#ifdef __CUDA_ARCH__
            return __longlong_as_double((long long)u);
#else
            double x;
            __builtin_memcpy(&x, &u, 8);
            return x;
#endif
        }

        /* x86 SSE returns the FIRST NaN operand, quieted, preserving sign and
           payload; measured on the reference for both add and multiply. */
        DACE_DFI int dace_powf_isnan(unsigned u)
        {
            return (u & 0x7fffffffu) > 0x7f800000u;
        }

        DACE_DFI float dace_powf_add(float a, float b)
        {
            unsigned ua = dace_powf_u32(a), ub = dace_powf_u32(b);
            if (dace_powf_isnan(ua)) return dace_powf_f32(ua | 0x00400000u);
            if (dace_powf_isnan(ub)) return dace_powf_f32(ub | 0x00400000u);
            return a + b;
        }

        DACE_DFI float dace_powf_sq(float x)
        {
            unsigned ux = dace_powf_u32(x);
            if (dace_powf_isnan(ux)) return dace_powf_f32(ux | 0x00400000u);
            return x * x;
        }

        /* Unary minus is an exact sign-bit flip (measured, including NaNs). */
        DACE_DFI float dace_powf_neg(float x)
        {
            return dace_powf_f32(dace_powf_u32(x) ^ 0x80000000u);
        }

        /* x86 divss: a NaN divisor is returned quieted, sign/payload intact. */
        DACE_DFI float dace_powf_rcp(float x)
        {
            unsigned ux = dace_powf_u32(x);
            if (dace_powf_isnan(ux)) return dace_powf_f32(ux | 0x00400000u);
            return 1.0f / x;
        }

        DACE_DFI int dace_powf_checkint(unsigned iy)
        {
            int e = (iy >> 23) & 0xff;
            if (e < 0x7f) return 0;
            if (e > 0x7f + 23) return 2;
            if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
            if (iy & (1u << (0x7f + 23 - e))) return 1;
            return 2;
        }

        DACE_DFI int dace_powf_zeroinfnan(unsigned ix)
        {
            return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
        }

        DACE_DFI int dace_powf_issignaling(unsigned ix)
        {
            return ((ix ^ 0x400000u) & 0x7fffffffu) > 0x7fc00000u;
        }

        DACE_DFI double dace_powf_log2(unsigned ix)
        {
            static const unsigned long long T[16][2] = {
                {0x3ff661ec79f8f3beULL, 0xbfdefec65b963019ULL},
                {0x3ff571ed4aaf883dULL, 0xbfdb0b6832d4fca4ULL},
                {0x3ff49539f0f010b0ULL, 0xbfd7418b0a1fb77bULL},
                {0x3ff3c995b0b80385ULL, 0xbfd39de91a6dcf7bULL},
                {0x3ff30d190c8864a5ULL, 0xbfd01d9bf3f2b631ULL},
                {0x3ff25e227b0b8ea0ULL, 0xbfc97c1d1b3b7af0ULL},
                {0x3ff1bb4a4a1a343fULL, 0xbfc2f9e393af3c9fULL},
                {0x3ff12358f08ae5baULL, 0xbfb960cbbf788d5cULL},
                {0x3ff0953f419900a7ULL, 0xbfaa6f9db6475fceULL},
                {0x3ff0000000000000ULL, 0x0000000000000000ULL},
                {0x3fee608cfd9a47acULL, 0x3fb338ca9f24f53dULL},
                {0x3feca4b31f026aa0ULL, 0x3fc476a9543891baULL},
                {0x3feb2036576afce6ULL, 0x3fce840b4ac4e4d2ULL},
                {0x3fe9c2d163a1aa2dULL, 0x3fd40645f0c6651cULL},
                {0x3fe886e6037841edULL, 0x3fd88e9c2c1b9ff8ULL},
                {0x3fe767dcf5534862ULL, 0x3fdce0a44eb17bccULL},
            };
            static const unsigned long long A[5] = {
                0x3fd27616c9496e0bULL,  /*  0.28845758110921399 */
                0xbfd71969a075c67aULL,  /* -0.36092606229713164 */
                0x3fdec70a6ca7baddULL,  /*  0.48089848147257702 */
                0xbfe7154748bef6c8ULL,  /* -0.72134746750062906 */
                0x3ff71547652ab82bULL,  /*  1.4426950408774342  */
            };

            unsigned tmp = ix - 0x3f330000u;              /* OFF */
            int i = (int)((tmp >> (23 - 4)) % 16u);
            unsigned top = tmp & 0xff800000u;
            unsigned iz = ix - top;
            int k = (int)((int)top >> 23);                /* arithmetic shift */

            double invc = dace_powf_f64(T[i][0]);
            double logc = dace_powf_f64(T[i][1]);
            double z = (double)dace_powf_f32(iz);

            double r  = fma(z, invc, -1.0);
            double y0 = logc + (double)k;

            double r2 = r * r;
            double y  = fma(dace_powf_f64(A[0]), r, dace_powf_f64(A[1]));
            double p  = fma(dace_powf_f64(A[2]), r, dace_powf_f64(A[3]));
            double r4 = r2 * r2;
            double q  = fma(dace_powf_f64(A[4]), r, y0);
            q = fma(p, r2, q);
            y = fma(y, r4, q);
            return y;
        }

        DACE_DFI double dace_powf_exp2(double xd, unsigned sign_bias)
        {
            static const unsigned long long T[32] = {
                0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL,
                0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
                0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
                0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
                0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL,
                0x3feedea64c123422ULL, 0x3feece086061892dULL,
                0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL,
                0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
                0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
                0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
                0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL,
                0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
                0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL,
                0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
                0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
                0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
            };
            static const unsigned long long C[3] = {
                0x3fac6af84b912394ULL,  /* 0.055503615593415351 */
                0x3fcebfce50fac4f3ULL,  /* 0.2402284522445722   */
                0x3fe62e42ff0c52d6ULL,  /* 0.69314718069162029  */
            };

            double kd = xd + 0x1.8p47;                    /* SHIFT */
            unsigned long long ki = dace_powf_u64(kd);
            kd -= 0x1.8p47;
            double r = xd - kd;

            unsigned long long t = T[ki % 32u];
            unsigned long long ski = ki + (unsigned long long)sign_bias;
            t += ski << (52 - 5);
            double s = dace_powf_f64(t);

            double z = fma(dace_powf_f64(C[0]), r, dace_powf_f64(C[1]));
            double r2 = r * r;
            double y = fma(dace_powf_f64(C[2]), r, 1.0);
            y = fma(z, r2, y);
            y = y * s;
            return y;
        }

        DACE_DFI float pow(const float& a, const float& b)
        {
            const unsigned SIGN_BIAS = 1u << 16;          /* 0x10000 */
            float x = a;
            float y = b;
            unsigned sign_bias = 0;
            unsigned ix = dace_powf_u32(x);
            unsigned iy = dace_powf_u32(y);

            if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u ||
                dace_powf_zeroinfnan(iy))
            {
                if (dace_powf_zeroinfnan(iy))
                {
                    if (2u * iy == 0)
                        return dace_powf_issignaling(ix) ? dace_powf_add(x, y) : 1.0f;
                    if (ix == 0x3f800000u)
                        return dace_powf_issignaling(iy) ? dace_powf_add(x, y) : 1.0f;
                    if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                        return dace_powf_add(x, y);
                    if (2u * ix == 2u * 0x3f800000u)
                        return 1.0f;
                    if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                        return 0.0f;
                    return y * y;
                }
                if (dace_powf_zeroinfnan(ix))
                {
                    float x2 = dace_powf_sq(x);
                    unsigned div_sign = 0;
                    if ((ix & 0x80000000u) && dace_powf_checkint(iy) == 1)
                    {
                        x2 = dace_powf_neg(x2);
                        div_sign = 1;
                    }
                    if (2u * ix == 0 && (iy & 0x80000000u))
                        return dace_powf_f32((div_sign ? 0x80000000u : 0u) |
                                             0x7f800000u);   /* __math_divzerof */
                    return (iy & 0x80000000u) ? dace_powf_rcp(x2) : x2;
                }
                /* x and y are non-zero finite. */
                if (ix & 0x80000000u)
                {
                    int yint = dace_powf_checkint(iy);
                    if (yint == 0)
                        return dace_powf_f32(0xffc00000u);  /* (x-x)/(x-x) */
                    if (yint == 1)
                        sign_bias = SIGN_BIAS;
                    ix &= 0x7fffffffu;
                }
                if (ix < 0x00800000u)
                {
                    /* Normalize subnormal x so the exponent becomes negative. */
                    ix = dace_powf_u32(x * 0x1p23f);
                    ix &= 0x7fffffffu;
                    ix -= 23u << 23;
                }
            }

            double logx = dace_powf_log2(ix);
            double ylogx = (double)y * logx;

            if (((dace_powf_u64(ylogx) >> 47) & 0xffffu) >= 0x80bfu)
            {
                if (ylogx > 0x1.fffffffd1d571p+6)
                    return dace_powf_f32(sign_bias ? 0xff800000u : 0x7f800000u);
                if (ylogx > 0x1.fffffffa3aae2p+6)
                {
                    /* glibc's round-away-from-zero check; both constants round
                       to 1.0f / -1.0f, so this is never taken in practice. */
                    float up = 1.0f + 0x1p-25f;
                    float dn = -1.0f - 0x1p-25f;
                    if ((!sign_bias && up != 1.0f) ||
                        ( sign_bias && dn != -1.0f))
                        return dace_powf_f32(sign_bias ? 0xff800000u : 0x7f800000u);
                }
                if (ylogx <= -150.0)
                    return dace_powf_f32(sign_bias ? 0x80000000u : 0x00000000u);
                if (ylogx < -149.0)
                    return dace_powf_f32(sign_bias ? 0x80000001u : 0x00000001u);
            }
            return (float)dace_powf_exp2(ylogx, sign_bias);
        }

        /* `base ** integer` call shapes lower onto the same glibc-exact
           routine (the host reference evaluates powf(base, (float)n)). */
        DACE_DFI float pow(const float& a, const int& b)
        {
            return pow(a, (float)b);
        }

        DACE_DFI float pow(const float& a, const unsigned int& b)
        {
            return pow(a, (float)b);
        }

/* ------------------------------------------------------------------
 * Bit-exact device lowering of glibc 2.34's float `log10f`.
 *
 * MEASURED on the host (AMD EPYC 9555): log10f@@GLIBC_2.2.5 @0x13870 is a
 * plain WEAK symbol (not IFUNC); positive finite inputs tail-jump to
 * __log10f_finite @0x3a080, which is COMPOSITIONAL -- it normalises x to
 * m in [0.5,2) and calls __logf_finite, then forms, in FLOAT with
 * separate roundings,
 *      result = (C0*kf + logf(m)*C1) + kf*C2
 * C0 = 0x1.a84fb60000000p-21f (log10(2) low split)
 * C1 = 0x1.bcb7b20000000p-2f  (1/ln 10)
 * C2 = 0x1.3441000000000p-2f  (log10(2) high split)
 *
 * __logf_finite is an IFUNC; its resolver @0x3a050 selects the FMA variant
 * @0x6dc90 on this CPU (cpu_features byte 0x9d bit 0x10 and byte 0xb8 bit
 * 0x20 both set), so only that variant is transcribed.  It evaluates a
 * 16-entry {invc, logc} table plus a quadratic in DOUBLE and rounds to
 * float once.
 *
 * Special cases (MEASURED, not inferred): ordered x <= 0 takes glibc's
 * __kernel_standard_f errno path: +-0 -> -inf (0xff800000), any ordered
 * negative (including -inf) -> +qNaN (0x7fc00000).  NaN is unordered and
 * stays on the finite path; x+x (positive) or the 0x3a158 quieting
 * (negative) leaves the payload and only sets the quiet bit.
 * The integer forms below reproduce those bits exactly, which also avoids
 * the device canonical-NaN of hardware x+x / 0/0.
 *
 * VERIFIED: exhaustive sweep of all 2^32 bit patterns against real libm
 * through a volatile function pointer: n_differ=0, max_ulp=0.
 *
 * Float products/adds use the _rn intrinsics so the device compiler cannot
 * contract the wrapper's three separate float roundings into FMAs
 * (glibc does them as separate mulss/addss).
 * ------------------------------------------------------------------ */
        DACE_DFI unsigned dace_log10f_bits(float f)
        {
#ifdef __CUDA_ARCH__
            return __float_as_uint(f);
#else
            union { float f; unsigned u; } cv;
            cv.f = f;
            return cv.u;
#endif
        }

        DACE_DFI float dace_log10f_frombits(unsigned u)
        {
#ifdef __CUDA_ARCH__
            return __uint_as_float(u);
#else
            union { float f; unsigned u; } cv;
            cv.u = u;
            return cv.f;
#endif
        }

        /* __logf_finite @0x6dc90 (FMA variant).  log10f only ever passes a
           normal m in [0.5,2), which never reaches logf's own error exits. */
        DACE_DFI float dace_glibc_logf_fma(float x)
        {
            /* .rodata @0xa76e0, 16 x {invc, logc} doubles, exact IO */
            static const double TAB[16][2] = {
                {0x1.661ec79f8f3bep+0, -0x1.57bf7808caadep-2},
                {0x1.571ed4aaf883dp+0, -0x1.2bef0a7c06ddbp-2},
                {0x1.49539f0f010b0p+0, -0x1.01eae7f513a67p-2},
                {0x1.3c995b0b80385p+0, -0x1.b31d8a68224e9p-3},
                {0x1.30d190c8864a5p+0, -0x1.6574f0ac07758p-3},
                {0x1.25e227b0b8ea0p+0, -0x1.1aa2bc79c8100p-3},
                {0x1.1bb4a4a1a343fp+0, -0x1.a4e76ce8c0e5ep-4},
                {0x1.12358f08ae5bap+0, -0x1.1973c5a611cccp-4},
                {0x1.0953f419900a7p+0, -0x1.252f438e10c1ep-5},
                {0x1.0000000000000p+0,  0x0.0p+0},
                {0x1.e608cfd9a47acp-1,  0x1.aa5aa5df25984p-5},
                {0x1.ca4b31f026aa0p-1,  0x1.c5e53aa362eb4p-4},
                {0x1.b2036576afce6p-1,  0x1.526e57720db08p-3},
                {0x1.9c2d163a1aa2dp-1,  0x1.bc2860d224770p-3},
                {0x1.886e6037841edp-1,  0x1.1058bc8a07ee1p-2},
                {0x1.767dcf5534862p-1,  0x1.4043057b6ee09p-2},
            };

            unsigned t = dace_log10f_bits(x);
            if (t == 0x3f800000u)               /* x == 1 -> +0 */
                return 0.0f;

            int d = (int)t - 0x3f330000;
            unsigned idx = ((unsigned)d >> 19) & 0xfu;
            double invc = TAB[idx][0];
            double logc = TAB[idx][1];

            int k = d >> 23;                    /* arithmetic shift */
            unsigned mant = t - ((unsigned)d & 0xff800000u);
            double z = (double)dace_log10f_frombits(mant);

            /* vfmadd132sd (%rcx),%xmm5,%xmm1 : z*invc + (-1.0) */
            double q = fma(z, invc, -1.0);
            /* vfmadd213sd P0,%xmm1,%xmm3 : q*P1 + P0 */
            double w = fma(q, 0x1.5575b0be00b6ap-2, -0x1.ffffef20a4123p-2);
            double q2 = q * q;
            /* vfmadd132sd ln2,%xmm6,%xmm2 : k*ln2 + logc */
            double a = fma((double)k, 0x1.62e42fefa39efp-1, logc);
            double b = q + a;
            /* vfmadd231sd P2,%xmm0,%xmm3 : q2*P2 + w */
            w = fma(q2, -0x1.00ea348b88334p-2, w);
            /* vfmadd132sd %xmm3,%xmm1,%xmm0 : q2*w + b */
            double r = fma(q2, w, b);
            return (float)r;
        }

        DACE_DFI float log10(const float& a)
        {
            unsigned u = dace_log10f_bits(a);
            int ix = (int)u;

            /* log10f@0x13870 outer errno branch (ucomiss 0,x; jae) */
            if ((u & 0x7fffffffu) == 0u)
                return dace_log10f_frombits(0xff800000u);  /* +-0 -> -inf */
            if (ix < 0) {
                if (u > 0xff800000u)                       /* negative NaN */
                    return dace_log10f_frombits(u | 0x00400000u);
                return dace_log10f_frombits(0x7fc00000u);  /* ordered < 0 */
            }
            if (u > 0x7f7fffffu) {                         /* +inf / +NaN */
                if (u == 0x7f800000u)
                    return dace_log10f_frombits(0x7f800000u);
                return dace_log10f_frombits(u | 0x00400000u);
            }

            int off;
            float xx = a;
            if (ix > 0x7fffff) {
                off = 0;                                   /* normal positive */
            } else {
                xx = __fmul_rn(a, 0x1p25f);                /* subnormal x 2^25 */
                off = -25;
                u = dace_log10f_bits(xx);
            }

            int ecx = (int)u;
            unsigned mant = u & 0x7fffffu;
            int kk = ecx >> 23;
            int kraw = off + kk - 127;
            int neg = (int)((unsigned)kraw >> 31);         /* 0 or 1 */
            int kad = kraw + neg;
            float kf = (float)kad;
            unsigned mb = ((unsigned)(0x7f - neg) << 23) | mant;
            float m = dace_log10f_frombits(mb);

            float lg = dace_glibc_logf_fma(m);

            /* separate float roundings, as glibc's mulss/mulss/mulss/addss/addss */
            float t0 = __fmul_rn(0x1.a84fb60000000p-21f, kf);
            float t1 = __fmul_rn(lg, 0x1.bcb7b20000000p-2f);
            float t2 = __fmul_rn(kf, 0x1.3441000000000p-2f);
            return __fadd_rn(__fadd_rn(t0, t1), t2);
        }

        /* ------------------------------------------------------------------
         * Bit-exact device lowering of glibc 2.34 x86-64 `atan2f`.
         *
         * glibc 2.34 exports `atan2f` as a plain WEAK FUNC (no IFUNC) at libm.so.6+0x13170: a thin
         * wrapper that tail-calls `atanf` when x == +1.0f and otherwise calls `__atan2f_finite` at
         * +0x360c0.  The lattice below is that binary, read off the disassembly -- including the
         * ORDER of the tests (x==+-inf is tested BEFORE y==+-inf, which is observable when both are
         * infinite) and two details a from-memory transcription gets wrong:
         *   * `m == 1` flips the SIGN BIT rather than negating arithmetically (observable at +0.0);
         *   * the NaN path is `x + y` with x as the SSE DESTINATION, so a NaN x wins and otherwise
         *     the quieted y does.  Written as bit operations, because the host compiler is free to
         *     swap the addends -- it did, and that was 56 of the last 56 mismatching pairs.
         *
         * `dace_glibc_atanf` (already proven exact over all 2^32 inputs) is the ONLY transcendental.
         *
         * MEASURED: 211,716,234 pairs -- 200M uniform random bit pairs, the full special-value cross
         * grid, and a dense +-512-ulp sweep around |y| == |x| plus ix == iy -- give n_differ = 0.
         * ------------------------------------------------------------------ */
        /* No inner __CUDA_ARCH__ conditional: this whole block is already inside the device-only
           guard, and a nested one would be mistaken for the outer guard by the assembler. */
        DACE_DFI unsigned dace_atan2f_bits(float f)
        {
            return __float_as_uint(f);
        }

        DACE_DFI float dace_atan2f_from_bits(unsigned u)
        {
            return __uint_as_float(u);
        }

        DACE_DFI float atan2(const float& yy, const float& xx)
        {
            const float y = yy, x = xx;
            const unsigned hy = dace_atan2f_bits(y), hx = dace_atan2f_bits(x);
            const unsigned iy = hy & 0x7fffffffu, ix = hx & 0x7fffffffu;

            /* NaN -> x + y, with x as the destination register */
            if (ix > 0x7f800000u || iy > 0x7f800000u)
                return dace_atan2f_from_bits((ix > 0x7f800000u ? hx : hy) | 0x00400000u);

            const float PIO2  = dace_atan2f_from_bits(0x3fc90fdbu);
            const float PIO4  = dace_atan2f_from_bits(0x3f490fdbu);
            const float PI    = dace_atan2f_from_bits(0x40490fdbu);
            const float TINY  = dace_atan2f_from_bits(0x0da24260u);
            const float NPI   = dace_atan2f_from_bits(0xc0490fdbu);
            const float NPIO2 = dace_atan2f_from_bits(0xbfc90fdbu);
            const float NPIO4 = dace_atan2f_from_bits(0xbf490fdbu);
            const float K1    = dace_atan2f_from_bits(0x333bbd2eu);   /* pi/2 low part */
            const float K2    = dace_atan2f_from_bits(0x33bbbd2eu);   /* pi   low part */

            if (hx == 0x3f800000u)                 /* x == +1.0 -> atanf(y) */
                return dace_glibc_atanf(y);

            const int m = (int)(((hx >> 30) & 2u) | (hy >> 31));

            if (iy == 0u)                          /* y == +-0 */
            {
                if (m == 2) return TINY + PI;
                if (m == 3) return NPI - TINY;
                return y;
            }
            if (ix == 0u)                          /* x == +-0 */
            {
                if (hy >> 31) return NPIO2 - TINY;
                return TINY + PIO2;
            }
            if (ix == 0x7f800000u)                 /* x == +-inf */
            {
                if (iy == 0x7f800000u)             /* both infinite */
                {
                    if (m == 2) return (3.0f * PIO4) + TINY;
                    if (m == 3) return (-3.0f * PIO4) - TINY;
                    if (m == 1) return NPIO4 - TINY;
                    return TINY + PIO4;
                }
                if (m == 2) return TINY + PI;
                if (m == 3) return NPI - TINY;
                if (m == 1) return dace_atan2f_from_bits(0x80000000u);   /* -0.0 */
                return 0.0f;
            }
            if (iy == 0x7f800000u)                 /* y == +-inf, x finite non-zero */
            {
                if (hy >> 31) return NPIO2 - TINY;
                return TINY + PIO2;
            }

            const int diff = (int)iy - (int)ix;
            const int k = diff >> 23;
            float z;
            if (diff > 0x1e7fffff) z = PIO2 - K1;
            else if ((int)hx < 0 && k < -60) z = 0.0f;
            else z = dace_glibc_atanf(fabsf(y / x));

            if (m == 0) return z;
            if (m == 1) return dace_atan2f_from_bits(dace_atan2f_bits(z) ^ 0x80000000u);
            if (m == 2) return PI - (z + K2);
            return (z + K2) - PI;
        }

        DACE_DFI float tanh(const float& x) { return dace_glibc_tanhf(x); }

#endif  // device
    }  // namespace math
}  // namespace dace

#endif  // DACE_GLIBC_F32

#endif  // __DACE_GLIBC_F32_H
