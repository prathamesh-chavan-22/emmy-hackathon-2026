#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add(const float* add__partial, const __half* x2, __half* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1 = (_gid) % 128;
        float acc0 = 0.0f;
        for (int _a2_ks = 0; _a2_ks < 2; _a2_ks++) {
            float acc0__p = add__partial[_a2_ks * 16384 + a0 * 128 + a1];
            acc0 += acc0__p;
        }
        __half v1 = __float2half(acc0);
        __half in2 = x2[a1];
        __half v2 = in2 + v1;
        add[a0 * 128 + a1] = v2;
}
