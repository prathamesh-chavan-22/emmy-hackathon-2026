#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_to_2(const float* to_2__partial, float* to_2) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 40;
        int a1 = (_gid) % 40;
        float acc0 = 0.0f;
        for (int _a2_ks = 0; _a2_ks < 4; _a2_ks++) {
            float acc0__p = to_2__partial[_a2_ks * 1280 + a0 * 40 + a1];
            acc0 += acc0__p;
        }
        float v3 = acc0;
        to_2[a0 * 40 + a1] = v3;
}
