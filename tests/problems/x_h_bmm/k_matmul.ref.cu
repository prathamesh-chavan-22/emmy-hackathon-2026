#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* matmul__partial, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 4096;
        int a1 = (_gid / 64) % 64;
        int a2 = (_gid) % 64;
        float acc0 = 0.0f;
        for (int _a3_ks = 0; _a3_ks < 32; _a3_ks++) {
            float acc0__p = matmul__partial[_a3_ks * 16384 + a0 * 4096 + a1 * 64 + a2];
            acc0 += acc0__p;
        }
        matmul[a0 * 4096 + a1 * 64 + a2] = __float2half(acc0);
}
