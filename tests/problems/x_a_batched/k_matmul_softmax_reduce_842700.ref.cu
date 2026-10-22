extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_842700(const float* div_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1408) {
        int a0 = _gid / 352;
        int a1_b = (_gid / 11) % 32;
        int a6_b = (_gid) % 11;
        int a1_u = (_gid) % 1;
        int a6_u = (_gid) % 1;
        float in0 = div_c1[0];
        float v1 = 1.0f / in0;
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c0_2 = 0.0f;
        float acc5__sum__c0_2 = 0.0f;
        float acc1__c0_2 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        float acc3__c1_2 = 0.0f;
        float acc5__sum__c1_2 = 0.0f;
        float acc1__c1_2 = -1e+30f;
        #pragma unroll
        for (int a2 = 0; a2 < 64; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar0 = q[a0 * 2048 + (a1_b * 2 + a1_u * 2) * 32 + a3];
                float in1__ar0 = kt[a0 * 2048 + a3 * 64 + a2];
                float v3__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v3__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar1 = q[a0 * 2048 + (a1_b * 2 + a1_u * 2 + 1) * 32 + a3];
                float in1__ar1 = kt[a0 * 2048 + a3 * 64 + a2];
                float v3__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v3__ar1;
            }
            float v4__c0_0 = acc0__ar0 * v1;
            float one__c0_0 = 1.0f;
            float v4__gn__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__alpha__c0_0 = expf((acc1__c0_0 - v4__gn__c0_0));
            float v4__beta__c0_0 = expf((v4__c0_0 - v4__gn__c0_0));
            float v4__acc3_sa__c0_0 = acc3__c0_0 * v4__alpha__c0_0;
            float v4__acc3_sb__c0_0 = one__c0_0 * v4__beta__c0_0;
            acc3__c0_0 = v4__acc3_sa__c0_0 + v4__acc3_sb__c0_0;
            float v4__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v4__alpha__c0_0;
            float in7__bc0 = v[a0 * 2048 + a2 * 32 + (a6_b * 3 + a6_u * 3) % 32];
            float v4__acc5__sum_sb__c0_0 = in7__bc0 * v4__beta__c0_0;
            acc5__sum__c0_0 = v4__acc5__sum_sa__c0_0 + v4__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__c0_1 = acc0__ar0 * v1;
            float one__c0_1 = 1.0f;
            float v4__gn__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
            float v4__alpha__c0_1 = expf((acc1__c0_1 - v4__gn__c0_1));
            float v4__beta__c0_1 = expf((v4__c0_1 - v4__gn__c0_1));
            float v4__acc3_sa__c0_1 = acc3__c0_1 * v4__alpha__c0_1;
            float v4__acc3_sb__c0_1 = one__c0_1 * v4__beta__c0_1;
            acc3__c0_1 = v4__acc3_sa__c0_1 + v4__acc3_sb__c0_1;
            float v4__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v4__alpha__c0_1;
            float in7__bc1 = v[a0 * 2048 + a2 * 32 + (a6_b * 3 + a6_u * 3 + 1) % 32];
            float v4__acc5__sum_sb__c0_1 = in7__bc1 * v4__beta__c0_1;
            acc5__sum__c0_1 = v4__acc5__sum_sa__c0_1 + v4__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
            float v4__c0_2 = acc0__ar0 * v1;
            float one__c0_2 = 1.0f;
            float v4__gn__c0_2 = fmaxf(acc1__c0_2, v4__c0_2);
            float v4__alpha__c0_2 = expf((acc1__c0_2 - v4__gn__c0_2));
            float v4__beta__c0_2 = expf((v4__c0_2 - v4__gn__c0_2));
            float v4__acc3_sa__c0_2 = acc3__c0_2 * v4__alpha__c0_2;
            float v4__acc3_sb__c0_2 = one__c0_2 * v4__beta__c0_2;
            acc3__c0_2 = v4__acc3_sa__c0_2 + v4__acc3_sb__c0_2;
            float v4__acc5__sum_sa__c0_2 = acc5__sum__c0_2 * v4__alpha__c0_2;
            float in7__bc2 = v[a0 * 2048 + a2 * 32 + (a6_b * 3 + a6_u * 3 + 2) % 32];
            float v4__acc5__sum_sb__c0_2 = in7__bc2 * v4__beta__c0_2;
            acc5__sum__c0_2 = v4__acc5__sum_sa__c0_2 + v4__acc5__sum_sb__c0_2;
            acc1__c0_2 = fmaxf(acc1__c0_2, v4__c0_2);
            float v4__c1_0 = acc0__ar1 * v1;
            float one__c1_0 = 1.0f;
            float v4__gn__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__alpha__c1_0 = expf((acc1__c1_0 - v4__gn__c1_0));
            float v4__beta__c1_0 = expf((v4__c1_0 - v4__gn__c1_0));
            float v4__acc3_sa__c1_0 = acc3__c1_0 * v4__alpha__c1_0;
            float v4__acc3_sb__c1_0 = one__c1_0 * v4__beta__c1_0;
            acc3__c1_0 = v4__acc3_sa__c1_0 + v4__acc3_sb__c1_0;
            float v4__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v4__alpha__c1_0;
            float v4__acc5__sum_sb__c1_0 = in7__bc0 * v4__beta__c1_0;
            acc5__sum__c1_0 = v4__acc5__sum_sa__c1_0 + v4__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__c1_1 = acc0__ar1 * v1;
            float one__c1_1 = 1.0f;
            float v4__gn__c1_1 = fmaxf(acc1__c1_1, v4__c1_1);
            float v4__alpha__c1_1 = expf((acc1__c1_1 - v4__gn__c1_1));
            float v4__beta__c1_1 = expf((v4__c1_1 - v4__gn__c1_1));
            float v4__acc3_sa__c1_1 = acc3__c1_1 * v4__alpha__c1_1;
            float v4__acc3_sb__c1_1 = one__c1_1 * v4__beta__c1_1;
            acc3__c1_1 = v4__acc3_sa__c1_1 + v4__acc3_sb__c1_1;
            float v4__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v4__alpha__c1_1;
            float v4__acc5__sum_sb__c1_1 = in7__bc1 * v4__beta__c1_1;
            acc5__sum__c1_1 = v4__acc5__sum_sa__c1_1 + v4__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v4__c1_1);
            float v4__c1_2 = acc0__ar1 * v1;
            float one__c1_2 = 1.0f;
            float v4__gn__c1_2 = fmaxf(acc1__c1_2, v4__c1_2);
            float v4__alpha__c1_2 = expf((acc1__c1_2 - v4__gn__c1_2));
            float v4__beta__c1_2 = expf((v4__c1_2 - v4__gn__c1_2));
            float v4__acc3_sa__c1_2 = acc3__c1_2 * v4__alpha__c1_2;
            float v4__acc3_sb__c1_2 = one__c1_2 * v4__beta__c1_2;
            acc3__c1_2 = v4__acc3_sa__c1_2 + v4__acc3_sb__c1_2;
            float v4__acc5__sum_sa__c1_2 = acc5__sum__c1_2 * v4__alpha__c1_2;
            float v4__acc5__sum_sb__c1_2 = in7__bc2 * v4__beta__c1_2;
            acc5__sum__c1_2 = v4__acc5__sum_sa__c1_2 + v4__acc5__sum_sb__c1_2;
            acc1__c1_2 = fmaxf(acc1__c1_2, v4__c1_2);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a6_b * 3 + a6_u * 3 + 0 < 32) {
            matmul_1[a0 * 2048 + (a1_b * 2 + a1_u * 2) * 32 + (a6_b * 3 + a6_u * 3) % 32] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a6_b * 3 + a6_u * 3 + 1 < 32) {
            matmul_1[a0 * 2048 + (a1_b * 2 + a1_u * 2) * 32 + (a6_b * 3 + a6_u * 3 + 1) % 32] = acc5__c0_1;
        }
        float acc5__c0_2 = acc5__sum__c0_2 * (1.0f / acc3__c0_2);
        if (a6_b * 3 + a6_u * 3 + 2 < 32) {
            matmul_1[a0 * 2048 + (a1_b * 2 + a1_u * 2) * 32 + (a6_b * 3 + a6_u * 3 + 2) % 32] = acc5__c0_2;
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a6_b * 3 + a6_u * 3 + 0 < 32) {
            matmul_1[a0 * 2048 + (a1_b * 2 + a1_u * 2 + 1) * 32 + (a6_b * 3 + a6_u * 3) % 32] = acc5__c1_0;
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a6_b * 3 + a6_u * 3 + 1 < 32) {
            matmul_1[a0 * 2048 + (a1_b * 2 + a1_u * 2 + 1) * 32 + (a6_b * 3 + a6_u * 3 + 1) % 32] = acc5__c1_1;
        }
        float acc5__c1_2 = acc5__sum__c1_2 * (1.0f / acc3__c1_2);
        if (a6_b * 3 + a6_u * 3 + 2 < 32) {
            matmul_1[a0 * 2048 + (a1_b * 2 + a1_u * 2 + 1) * 32 + (a6_b * 3 + a6_u * 3 + 2) % 32] = acc5__c1_2;
        }
    }
}
