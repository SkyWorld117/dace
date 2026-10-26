// Copyright 2026 ETH Zurich and the DaCe authors. All rights reserved.
// Scalar glibc 2.34 numerical compatibility (round-to-nearest, no fast math).
// Build with nvcc -std=c++20 -DDACE_GLIBC_F32=1 --fmad=false --ftz=false
// -Xcompiler=-fopenmp,-ffp-contract=off,-fno-tree-vectorize -I../dace/runtime/include.
// Default: stratified/edge cases. --exhaustive: every binary32 bit pattern.
#include <dace/dace.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void checked(cudaError_t status) {
    if (status!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status)); std::exit(1); }
}
DACE_HDFI unsigned input_bits(unsigned long long i,bool exhaustive) {
    if (!exhaustive && i<54) {
        const unsigned edges[]={0u,0x00800000u,0x24000000u,0x33000000u,0x3eb17218u,
                                0x3f800000u,0x41b00000u,0x7f800000u,0x7fc00000u};
        return (edges[i/6]+unsigned(int(i%3)-1))^((i%6>=3)?0x80000000u:0u);
    }
    return exhaustive?unsigned(i):unsigned(i)*747796405u+2891336453u;
}
static bool same(float a,float b) {
    unsigned ua=dace::math::dace_tanhf_bits(a),ub=dace::math::dace_tanhf_bits(b);
    return ua==ub || ((ua&0x7fffffffu)>0x7f800000u && (ub&0x7fffffffu)>0x7f800000u);
}
__global__ void evaluate(unsigned long long start,unsigned n,bool exhaustive,float *actual,float *native) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) {
        float x=__uint_as_float(input_bits(start+i,exhaustive));
        actual[i]=dace::math::tanh(x);native[i]=::tanhf(x);
    }
}
int main(int argc,char **argv) {
    bool exhaustive=argc==2 && std::strcmp(argv[1],"--exhaustive")==0;
    const unsigned chunk=1u<<20;
    const unsigned long long count=exhaustive?(1ull<<32):(1ull<<23);
    std::vector<float> actual(chunk),native(chunk);float *device,*control;
    checked(cudaMalloc(&device,chunk*sizeof(float)));checked(cudaMalloc(&control,chunk*sizeof(float)));
    float (*volatile reference)(float)=::tanhf;
    unsigned long long host_bad=0,device_bad=0,native_bad=0;
    for (unsigned long long start=0;start<count;start+=chunk) {
        evaluate<<<(chunk+255)/256,256>>>(start,chunk,exhaustive,device,control);
        checked(cudaGetLastError());
        checked(cudaMemcpy(actual.data(),device,chunk*sizeof(float),cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(native.data(),control,chunk*sizeof(float),cudaMemcpyDeviceToHost));
        #pragma omp parallel for num_threads(32) reduction(+:host_bad,device_bad,native_bad)
        for (unsigned i=0;i<chunk;++i) {
            float x=dace::math::dace_tanhf_float(input_bits(start+i,exhaustive));
            float expected=reference(x);
            host_bad+=!same(expected,dace::math::dace_glibc_tanhf(x));
            device_bad+=!same(expected,actual[i]);native_bad+=!same(expected,native[i]);
        }
    }
    checked(cudaFree(device));checked(cudaFree(control));
    std::printf("inputs=%llu host_mismatches=%llu device_mismatches=%llu native_control_mismatches=%llu\n",
                count,host_bad,device_bad,native_bad);
    return host_bad || device_bad || !native_bad;
}
