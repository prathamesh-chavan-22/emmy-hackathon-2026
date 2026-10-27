#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(16) void k_to_2__partial(const float* x1, const float* x0, float* to_2__partial) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 20480;
        int a0 = (_gid / 640) % 32;
        int a1 = (_gid / 16) % 40;
        int a2_co = (_gid) % 16;
        float acc0 = 0.0f;
        for (int a2 = a2_co; a2 < 24; a2 += 16) {
            float in0 = x0[a0 * 48 + (_a2_ks * 24 + a2)];
            __half v0 = __float2half(in0);
            float in1 = x1[a1 * 48 + (_a2_ks * 24 + a2)];
            __half v1 = __float2half(in1);
            __half v2 = v0 * v1;
            acc0 += __half2float(v2);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        if (a2_co == 0) {
            to_2__partial[_a2_ks * 1280 + a0 * 40 + a1] = acc0;
        }
}
