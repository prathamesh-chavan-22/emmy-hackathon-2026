static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 6144;
        int a0_b = (_gid / 1024) % 6;
        int a1_b = (_gid / 256) % 4;
        int a0_u = (_gid / 32) % 8;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c4_1 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c5_1 = 0.0f;
        __shared__ __align__(16) float _a_smem[768];
        __shared__ __align__(16) float _b_smem[1024];
        for (int _fa = a0_u * 32 + a1_u; _fa < 96; _fa += 256) {
            emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[((a0_b * 48 + _fa / 2 < 256) ? (a0_b * 48 + _fa / 2) : (255)) * 384 + (_a2_ks * 24 + 4 * (_fa % 2))]);
        }
        for (int _fb = a0_u * 32 + a1_u; _fb < 128; _fb += 256) {
            emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[(_a2_ks * 24 + _fb / 16) * 256 + (a1_b * 64 + 4 * (_fb % 16))]);
        }
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 24; _ks += 8) {
            for (int _fa = a0_u * 32 + a1_u; _fa < 96; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[((_ks / 8 + 1) % 2 * 48 + _fa / 2) * 8 + 4 * (_fa % 2)], &x0[((a0_b * 48 + _fa / 2 < 256) ? (a0_b * 48 + _fa / 2) : (255)) * 384 + (_a2_ks * 24 + (((_ks + 8 < 24) ? (_ks + 8) : (16)) + 4 * (_fa % 2)))]);
            }
            for (int _fb = a0_u * 32 + a1_u; _fb < 128; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[((_ks / 8 + 1) % 2 * 8 + _fb / 16) * 64 + 4 * (_fb % 16)], &x1[(_a2_ks * 24 + (((_ks + 8 < 24) ? (_ks + 8) : (16)) + _fb / 16)) * 256 + (a1_b * 64 + 4 * (_fb % 16))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<1>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 8; _ki++) {
                float in1__c0_0 = _b_smem[(_ks / 8 % 2 * 8 + _ki) * 64 + (a1_b * 64 + a1_u * 2 - a1_b * 64)];
                float in0__c0_0 = _a_smem[(_ks / 8 % 2 * 48 + (a0_b * 48 + a0_u * 6 - a0_b * 48)) * 8 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[(_ks / 8 % 2 * 8 + _ki) * 64 + (a1_b * 64 + a1_u * 2 + 1 - a1_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(_ks / 8 % 2 * 48 + (a0_b * 48 + a0_u * 6 + 1 - a0_b * 48)) * 8 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float in0__c2_0 = _a_smem[(_ks / 8 % 2 * 48 + (a0_b * 48 + a0_u * 6 + 2 - a0_b * 48)) * 8 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float in0__c3_0 = _a_smem[(_ks / 8 % 2 * 48 + (a0_b * 48 + a0_u * 6 + 3 - a0_b * 48)) * 8 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
                float in0__c4_0 = _a_smem[(_ks / 8 % 2 * 48 + (a0_b * 48 + a0_u * 6 + 4 - a0_b * 48)) * 8 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float acc0__v__c4_1 = in1__c0_1 * in0__c4_0;
                acc0__c4_1 += acc0__v__c4_1;
                float in0__c5_0 = _a_smem[(_ks / 8 % 2 * 48 + (a0_b * 48 + a0_u * 6 + 5 - a0_b * 48)) * 8 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float acc0__v__c5_1 = in1__c0_1 * in0__c5_0;
                acc0__c5_1 += acc0__v__c5_1;
            }
            __syncthreads();
        }
        if (a0_b * 48 + a0_u * 6 + 0 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6) % 256 * 256 + (a1_b * 64 + a1_u * 2)], acc0__c0_0);
        }
        if (a0_b * 48 + a0_u * 6 + 0 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6) % 256 * 256 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c0_1);
        }
        if (a0_b * 48 + a0_u * 6 + 1 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 1) % 256 * 256 + (a1_b * 64 + a1_u * 2)], acc0__c1_0);
        }
        if (a0_b * 48 + a0_u * 6 + 1 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 1) % 256 * 256 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c1_1);
        }
        if (a0_b * 48 + a0_u * 6 + 2 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 2) % 256 * 256 + (a1_b * 64 + a1_u * 2)], acc0__c2_0);
        }
        if (a0_b * 48 + a0_u * 6 + 2 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 2) % 256 * 256 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c2_1);
        }
        if (a0_b * 48 + a0_u * 6 + 3 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 3) % 256 * 256 + (a1_b * 64 + a1_u * 2)], acc0__c3_0);
        }
        if (a0_b * 48 + a0_u * 6 + 3 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 3) % 256 * 256 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c3_1);
        }
        if (a0_b * 48 + a0_u * 6 + 4 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 4) % 256 * 256 + (a1_b * 64 + a1_u * 2)], acc0__c4_0);
        }
        if (a0_b * 48 + a0_u * 6 + 4 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 4) % 256 * 256 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c4_1);
        }
        if (a0_b * 48 + a0_u * 6 + 5 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 5) % 256 * 256 + (a1_b * 64 + a1_u * 2)], acc0__c5_0);
        }
        if (a0_b * 48 + a0_u * 6 + 5 < 256) {
            atomicAdd(&matmul[(a0_b * 48 + a0_u * 6 + 5) % 256 * 256 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c5_1);
        }
}
