// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// Adapted from glibc 2.34 sysdeps/ieee754/flt-32/{s_tanhf,s_expm1f}.c:
// https://github.com/bminor/glibc/blob/glibc-2.34/sysdeps/ieee754/flt-32/s_tanhf.c
// https://github.com/bminor/glibc/blob/glibc-2.34/sysdeps/ieee754/flt-32/s_expm1f.c
// Numeric round-to-nearest compatibility only; device errno and floating-point
// exception flags are not emulated. Explicit CUDA operations prevent contraction.
#ifndef __DACE_GLIBC_TANHF_H
#define __DACE_GLIBC_TANHF_H
#include "types.h"
#include <cstring>

namespace dace { namespace math {
DACE_HDFI unsigned dace_tanhf_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    unsigned u; std::memcpy(&u, &x, sizeof(u)); return u;
#endif
}
DACE_HDFI float dace_tanhf_float(unsigned u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float x; std::memcpy(&x, &u, sizeof(x)); return x;
#endif
}
DACE_HDFI float dace_tanhf_add(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a,b);
#else
    return a+b;
#endif
}
DACE_HDFI float dace_tanhf_sub(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fsub_rn(a,b);
#else
    return a-b;
#endif
}
DACE_HDFI float dace_tanhf_mul(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a,b);
#else
    return a*b;
#endif
}
DACE_HDFI float dace_tanhf_div(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a,b);
#else
    return a/b;
#endif
}
// Internal expm1 reduction: tanh only supplies values in [-2,44]. This bounded
// helper does not implement the unrelated expm1 overflow/errno interface.
DACE_HDFI float dace_tanhf_expm1(float x) {
    const float ln2_hi=0x1.62e300p-1f, ln2_lo=0x1.2fefa2p-17f;
    const float invln2=0x1.715476p+0f;
    const float q1=-0x1.111112p-5f, q2=0x1.a01a02p-10f;
    const float q3=-0x1.4ce19ap-14f, q4=0x1.0cfca8p-18f, q5=-0x1.afdb76p-23f;
    unsigned word=dace_tanhf_bits(x), hx=word&0x7fffffffu;
    int k=0; float hi=0,lo=0,c=0,t,e,y;
    if (hx>0x3eb17218u) {
        if (hx<0x3f851592u) {
            if (word>>31) { hi=dace_tanhf_add(x,ln2_hi);lo=-ln2_lo;k=-1; }
            else { hi=dace_tanhf_sub(x,ln2_hi);lo=ln2_lo;k=1; }
        } else {
            k=int(dace_tanhf_add(dace_tanhf_mul(invln2,x),(word>>31)?-0.5f:0.5f));
            t=float(k);hi=dace_tanhf_sub(x,dace_tanhf_mul(t,ln2_hi));lo=dace_tanhf_mul(t,ln2_lo);
        }
        x=dace_tanhf_sub(hi,lo);c=dace_tanhf_sub(dace_tanhf_sub(hi,x),lo);
    } else if (hx<0x33000000u) return x;
    const float hfx=dace_tanhf_mul(0.5f,x),hxs=dace_tanhf_mul(x,hfx);
    float r1=dace_tanhf_add(q4,dace_tanhf_mul(hxs,q5));
    r1=dace_tanhf_add(q3,dace_tanhf_mul(hxs,r1));
    r1=dace_tanhf_add(q2,dace_tanhf_mul(hxs,r1));
    r1=dace_tanhf_add(q1,dace_tanhf_mul(hxs,r1));
    r1=dace_tanhf_add(1.f,dace_tanhf_mul(hxs,r1));
    t=dace_tanhf_sub(3.f,dace_tanhf_mul(r1,hfx));
    e=dace_tanhf_mul(hxs,dace_tanhf_div(dace_tanhf_sub(r1,t),dace_tanhf_sub(6.f,dace_tanhf_mul(x,t))));
    if (k==0) return dace_tanhf_sub(x,dace_tanhf_sub(dace_tanhf_mul(x,e),hxs));
    e=dace_tanhf_sub(dace_tanhf_sub(dace_tanhf_mul(x,dace_tanhf_sub(e,c)),c),hxs);
    if (k==-1) return dace_tanhf_sub(dace_tanhf_mul(0.5f,dace_tanhf_sub(x,e)),0.5f);
    if (k==1) {
        if (x<-0.25f) return dace_tanhf_mul(-2.f,dace_tanhf_sub(e,dace_tanhf_add(x,0.5f)));
        return dace_tanhf_add(1.f,dace_tanhf_mul(2.f,dace_tanhf_sub(x,e)));
    }
    // Unsigned exponent adjustment avoids shifting a negative signed integer.
    const unsigned scale=unsigned(k)*0x800000u;
    if (k<=-2 || k>56) {
        y=dace_tanhf_sub(1.f,dace_tanhf_sub(e,x));
        y=dace_tanhf_float(dace_tanhf_bits(y)+scale);
        return dace_tanhf_sub(y,1.f);
    }
    if (k<23) {
        t=dace_tanhf_float(0x3f800000u-(0x1000000u>>k));
        y=dace_tanhf_sub(t,dace_tanhf_sub(e,x));
    } else {
        t=dace_tanhf_float(unsigned(0x7f-k)*0x800000u);
        y=dace_tanhf_add(dace_tanhf_sub(x,dace_tanhf_add(e,t)),1.f);
    }
    return dace_tanhf_float(dace_tanhf_bits(y)+scale);
}
/// Match scalar glibc 2.34 tanhf values in round-to-nearest mode.
/// Preserve signed zero; NaNs remain NaNs. Device errno/fenv are not emulated.
DACE_HDFI float dace_glibc_tanhf(float x) {
    const unsigned word=dace_tanhf_bits(x),hx=word&0x7fffffffu;
    if (hx>0x7f800000u) return dace_tanhf_add(x,x);
    if (hx<0x24000000u) return x; // Includes both zeros and all subnormals.
    float z=1.f;
    if (hx<0x41b00000u) {
        float magnitude=dace_tanhf_float(hx),t;
        if (hx>=0x3f800000u) {
            t=dace_tanhf_expm1(dace_tanhf_mul(2.f,magnitude));
            z=dace_tanhf_sub(1.f,dace_tanhf_div(2.f,dace_tanhf_add(t,2.f)));
        } else {
            t=dace_tanhf_expm1(dace_tanhf_mul(-2.f,magnitude));
            z=dace_tanhf_div(-t,dace_tanhf_add(t,2.f));
        }
    }
    return (word>>31)?-z:z;
}
}} // namespace dace::math
#endif
