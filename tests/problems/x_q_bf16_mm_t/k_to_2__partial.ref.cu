#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(32) void k_to_2__partial(const float* x1, const float* x0, float* to_2__partial) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 40960;
        int a0 = (_gid / 1280) % 32;
        int a1 = (_gid / 32) % 40;
        int a2_co = (_gid) % 32;
        float acc0 = 0.0f;
        float acc0__r1 = 0.0f;
        for (int a2 = a2_co; a2 < 12; a2 += 64) {
            float in0 = x0[a0 * 48 + (_a2_ks * 12 + a2)];
            float v0 = in0;
            float in1 = x1[a1 * 48 + (_a2_ks * 12 + a2)];
            float v1 = in1;
            float v2 = v0 * v1;
            acc0 += v2;
            float in0__r1 = x0[a0 * 48 + (_a2_ks * 12 + (a2 + 32) % 12)];
            float v0__r1 = in0__r1;
            float in1__r1 = x1[a1 * 48 + (_a2_ks * 12 + (a2 + 32) % 12)];
            float v1__r1 = in1__r1;
            float v2__r1 = v0__r1 * v1__r1;
            float v2__r1__id = 0.0f;
            float v2__r1__m = ((a2 + 32 < 12) ? (((float)(v2__r1))) : (((float)(v2__r1__id))));
            acc0__r1 += v2__r1__m;
        }
        acc0 += acc0__r1;
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
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
