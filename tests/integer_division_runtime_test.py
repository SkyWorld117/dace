"""Runtime integer division must compile and execute on CPU and CUDA."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _source(cuda):
    # Products remain representable; INT_MIN / -1 has no C++/Fortran result.
    cases = [(a, b) for a in range(-19, 20) for b in range(-13, 14) if b]
    cases += [(a, b) for a in (-2**60 + 17, 2**60 - 17) for b in (-101, -3, 3, 101)]
    rows = []
    for a, b in cases:
        quotient = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        rows.append('{%dLL,%dLL,%dLL,%dLL,%dLL,%dLL}' %
                    (a, b, a // b, a % b, quotient, a - quotient * b))
    prefix = '#include <cuda_runtime.h>\n' if cuda else ''
    qualifier = '__global__ ' if cuda else ''
    body = r'''
#include <dace/dace.h>
#include <cstdint>
#include <cstdio>
struct Case { int64_t a,b,floor,remainder,quotient,truncated; };
QUALIFIERvoid check(const Case* cases, int count, int* failures) {
    for (int i=0; i<count; ++i) {
        auto c=cases[i];
        int64_t q,r;
        cpp_divmod(c.a,c.b,q,r);
        if (int_floor_ni(c.a,c.b)!=c.floor || py_floor(c.a,c.b)!=c.floor ||
            floor_mod(c.a,c.b)!=c.remainder || cpp_mod(c.a,c.b)!=c.truncated ||
            q!=c.quotient || r!=c.truncated) ++*failures;
        if (c.a>=-19 && c.a<=19) {
            int32_t a=c.a,b=c.b,q32,r32;
            cpp_divmod(a,b,q32,r32);
            if (int_floor_ni(a,b)!=c.floor || floor_mod(a,b)!=c.remainder ||
                q32!=c.quotient || r32!=c.truncated) ++*failures;
            // Mixed-kind Fortran indices must reach the same device-safe helper.
            if (floor_mod(c.a,b)!=c.remainder) ++*failures;
        }
    }
}
int main() {
    Case cases[] = {ROWS};
    int count=sizeof(cases)/sizeof(cases[0]);
    int failures=0;
    EXECUTE
    if (failures) { std::fprintf(stderr,"%d integer-division failures\n",failures); return 1; }
    return 0;
}
'''.replace('QUALIFIER', qualifier).replace('ROWS', ',\n'.join(rows))
    cpu = 'check(cases,count,&failures);'
    gpu = r'''
    Case* device_cases; int* device_failures;
    #define CHECK_CUDA(call) do { auto rc=(call); if(rc!=cudaSuccess) { \
        std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(rc)); return 2; } } while(0)
    CHECK_CUDA(cudaMalloc(&device_cases,sizeof(cases)));
    CHECK_CUDA(cudaMalloc(&device_failures,sizeof(failures)));
    CHECK_CUDA(cudaMemcpy(device_cases,cases,sizeof(cases),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(device_failures,0,sizeof(failures)));
    check<<<1,1>>>(device_cases,count,device_failures);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&failures,device_failures,sizeof(failures),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_failures)); CHECK_CUDA(cudaFree(device_cases));
'''
    return prefix + body.replace('EXECUTE', gpu if cuda else cpu)


def test_integer_division_cpu(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'integer.cpp';source.write_text(_source(False))
    compiler=os.environ.get('CXX','g++')
    subprocess.run([compiler,'-std=c++20','-O2',
                    '-I'+str(root/'dace/runtime/include'),str(source),'-o',str(tmp_path/'check')],check=True)
    subprocess.run([str(tmp_path/'check')],check=True)


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which('nvcc') is None,reason='CUDA compiler unavailable')
def test_integer_division_cuda(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'integer.cu';source.write_text(_source(True))
    subprocess.run(['nvcc','-std=c++20','-O2','-arch=native','--expt-relaxed-constexpr','-Werror','cross-execution-space-call',
                    '-I'+str(root/'dace/runtime/include'),str(source),'-o',str(tmp_path/'check')],check=True)
    subprocess.run([str(tmp_path/'check')],check=True)
