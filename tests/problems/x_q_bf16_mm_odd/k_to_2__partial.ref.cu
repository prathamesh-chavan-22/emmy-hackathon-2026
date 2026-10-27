#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_to_2__partial(const float* x1, const float* x0, float* to_2__partial) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 5120;
        int a0_b = (_gid / 1280) % 4;
        int a1_b = (_gid / 256) % 5;
        int a0_u = (_gid / 32) % 8;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 50; a2++) {
            float in0__ar0 = x0[(a0_b * 32 + a0_u * 4) * 100 + (_a2_ks * 50 + a2)];
            float v0__ar0 = in0__ar0;
            float in0__ar1 = x0[(a0_b * 32 + a0_u * 4 + 1) * 100 + (_a2_ks * 50 + a2)];
            float v0__ar1 = in0__ar1;
            float in0__ar2 = x0[(a0_b * 32 + a0_u * 4 + 2) * 100 + (_a2_ks * 50 + a2)];
            float v0__ar2 = in0__ar2;
            float in0__ar3 = x0[(a0_b * 32 + a0_u * 4 + 3) * 100 + (_a2_ks * 50 + a2)];
            float v0__ar3 = in0__ar3;
            float in1__bc0 = x1[(_a2_ks * 50 + a2) * 130 + (a1_b * 32 + a1_u) % 130];
            float v1__bc0 = in1__bc0;
            float v2__c0_0 = v0__ar0 * v1__bc0;
            acc0__c0_0 += v2__c0_0;
            float v2__c1_0 = v0__ar1 * v1__bc0;
            acc0__c1_0 += v2__c1_0;
            float v2__c2_0 = v0__ar2 * v1__bc0;
            acc0__c2_0 += v2__c2_0;
            float v2__c3_0 = v0__ar3 * v1__bc0;
            acc0__c3_0 += v2__c3_0;
        }
        if (a1_b * 32 + a1_u * 1 + 0 < 130) {
            to_2__partial[_a2_ks * 16640 + (a0_b * 32 + a0_u * 4) * 130 + (a1_b * 32 + a1_u) % 130] = acc0__c0_0;
        }
        if (a1_b * 32 + a1_u * 1 + 0 < 130) {
            to_2__partial[_a2_ks * 16640 + (a0_b * 32 + a0_u * 4 + 1) * 130 + (a1_b * 32 + a1_u) % 130] = acc0__c1_0;
        }
        if (a1_b * 32 + a1_u * 1 + 0 < 130) {
            to_2__partial[_a2_ks * 16640 + (a0_b * 32 + a0_u * 4 + 2) * 130 + (a1_b * 32 + a1_u) % 130] = acc0__c2_0;
        }
        if (a1_b * 32 + a1_u * 1 + 0 < 130) {
            to_2__partial[_a2_ks * 16640 + (a0_b * 32 + a0_u * 4 + 3) * 130 + (a1_b * 32 + a1_u) % 130] = acc0__c3_0;
        }
}
