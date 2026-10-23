#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* matmul__partial, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _um = _gid / 256;
        int a0 = (_gid) % 256;
        float acc0 = 0.0f;
        for (int _a1_ks = 0; _a1_ks < 4; _a1_ks++) {
            float acc0__p = matmul__partial[_a1_ks * 256 + _um * 256 + a0];
            acc0 += acc0__p;
        }
        matmul[a0] = __float2half(acc0);
}
