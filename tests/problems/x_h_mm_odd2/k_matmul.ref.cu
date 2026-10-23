#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* matmul__partial, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 561) {
        int a0 = _gid / 17;
        int a1 = (_gid) % 17;
        float acc0 = 0.0f;
        for (int _a2_ks = 0; _a2_ks < 4; _a2_ks++) {
            float acc0__p = matmul__partial[_a2_ks * 561 + a0 * 17 + a1];
            acc0 += acc0__p;
        }
        matmul[a0 * 17 + a1] = __float2half(acc0);
    }
}
