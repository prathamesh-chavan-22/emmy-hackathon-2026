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
        int _a3_ks = _gid / 6144;
        int a0 = (_gid / 1536) % 4;
        int a1_b = (_gid / 768) % 2;
        int a2_b = (_gid / 256) % 3;
        int a1_u = (_gid / 16) % 16;
        int a2_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[256];
        __shared__ __align__(16) float _b_smem[128];
        for (int _ks = 0; _ks < 8; _ks += 8) {
            for (int _fa = a1_u * 16 + a2_u; _fa < 64; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[a0 * 2048 + (a1_b * 32 + _fa / 2) * 32 + (_a3_ks * 8 + (_ks + 4 * (_fa % 2)))]);
            }
            for (int _fb = a1_u * 16 + a2_u; _fb < 32; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[a0 * 1536 + (_a3_ks * 8 + (_ks + _fb / 4)) * 48 + (a2_b * 16 + 4 * (_fb % 4))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 8; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 16 + (a2_b * 16 + a2_u - a2_b * 16)];
                float in0__c0_0 = _a_smem[(a1_b * 32 + a1_u * 2 - a1_b * 32) * 8 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in0__c1_0 = _a_smem[(a1_b * 32 + a1_u * 2 + 1 - a1_b * 32) * 8 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
            }
            __syncthreads();
        }
        atomicAdd(&matmul[a0 * 3072 + (a1_b * 32 + a1_u * 2) * 48 + (a2_b * 16 + a2_u)], acc0__c0_0);
        atomicAdd(&matmul[a0 * 3072 + (a1_b * 32 + a1_u * 2 + 1) * 48 + (a2_b * 16 + a2_u)], acc0__c1_0);
}
