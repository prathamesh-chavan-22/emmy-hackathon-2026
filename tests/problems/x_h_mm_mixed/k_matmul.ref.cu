#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul(const __half* x0, const float* x1, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 1024;
        int a0_b = (_gid / 512) % 2;
        int a1_b = (_gid / 256) % 2;
        int a0_u = (_gid / 16) % 16;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 2; a2++) {
            __half in0__ar0 = x0[(a0_b * 32 + a0_u * 2) * 64 + (_a2_ks * 2 + a2)];
            float v0__ar0 = __half2float(in0__ar0);
            __half in0__ar1 = x0[(a0_b * 32 + a0_u * 2 + 1) * 64 + (_a2_ks * 2 + a2)];
            float v0__ar1 = __half2float(in0__ar1);
            float2 _v_in1__bc0 = *reinterpret_cast<const float2*>(&x1[(_a2_ks * 2 + a2) * 64 + (a1_b * 32 + a1_u * 2)]);
            float in1__bc0 = _v_in1__bc0.x;
            float in1__bc1 = _v_in1__bc0.y;
            float v1__c0_0 = in1__bc0 * v0__ar0;
            acc0__c0_0 += v1__c0_0;
            float v1__c0_1 = in1__bc1 * v0__ar0;
            acc0__c0_1 += v1__c0_1;
            float v1__c1_0 = in1__bc0 * v0__ar1;
            acc0__c1_0 += v1__c1_0;
            float v1__c1_1 = in1__bc1 * v0__ar1;
            acc0__c1_1 += v1__c1_1;
        }
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 2) * 64 + (a1_b * 32 + a1_u * 2)], acc0__c0_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 2) * 64 + (a1_b * 32 + a1_u * 2 + 1)], acc0__c0_1);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 2 + 1) * 64 + (a1_b * 32 + a1_u * 2)], acc0__c1_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 2 + 1) * 64 + (a1_b * 32 + a1_u * 2 + 1)], acc0__c1_1);
}
