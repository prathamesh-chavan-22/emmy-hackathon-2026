extern "C" __global__
__launch_bounds__(256) void k_sdpa_9d6e20(const float* scaled_dot_product_attention_scale, const float* x0, const float* x1, const float* x2, float* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 1024;
        int a1_b = (_gid / 256) % 4;
        int a6_b = (_gid / 256) % 1;
        int a1_u = (_gid / 16) % 16;
        int a6_u = (_gid) % 16;
        float in0 = scaled_dot_product_attention_scale[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        #pragma unroll
        for (int a2 = 0; a2 < 64; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar0 = x0[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + a3];
                float in1__ar0 = x1[a0 / 4 * 2048 + a2 * 32 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__c0_0 = acc0__ar0 * in0;
            float one__c0_0 = 1.0f;
            float v1__gn__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__alpha__c0_0 = expf((acc1__c0_0 - v1__gn__c0_0));
            float v1__beta__c0_0 = expf((v1__c0_0 - v1__gn__c0_0));
            float v1__acc3_sa__c0_0 = acc3__c0_0 * v1__alpha__c0_0;
            float v1__acc3_sb__c0_0 = one__c0_0 * v1__beta__c0_0;
            acc3__c0_0 = v1__acc3_sa__c0_0 + v1__acc3_sb__c0_0;
            float v1__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v1__alpha__c0_0;
            float2 _v_in7__bc0 = *reinterpret_cast<const float2*>(&x2[a0 / 4 * 2048 + a2 * 32 + (a6_b * 32 + a6_u * 2)]);
            float in7__bc0 = _v_in7__bc0.x;
            float in7__bc1 = _v_in7__bc0.y;
            float v1__acc5__sum_sb__c0_0 = in7__bc0 * v1__beta__c0_0;
            acc5__sum__c0_0 = v1__acc5__sum_sa__c0_0 + v1__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__c0_1 = acc0__ar0 * in0;
            float one__c0_1 = 1.0f;
            float v1__gn__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__alpha__c0_1 = expf((acc1__c0_1 - v1__gn__c0_1));
            float v1__beta__c0_1 = expf((v1__c0_1 - v1__gn__c0_1));
            float v1__acc3_sa__c0_1 = acc3__c0_1 * v1__alpha__c0_1;
            float v1__acc3_sb__c0_1 = one__c0_1 * v1__beta__c0_1;
            acc3__c0_1 = v1__acc3_sa__c0_1 + v1__acc3_sb__c0_1;
            float v1__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v1__alpha__c0_1;
            float v1__acc5__sum_sb__c0_1 = in7__bc1 * v1__beta__c0_1;
            acc5__sum__c0_1 = v1__acc5__sum_sa__c0_1 + v1__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        scaled_dot_product_attention[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + (a6_b * 32 + a6_u * 2)] = acc5__c0_0;
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        scaled_dot_product_attention[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + (a6_b * 32 + a6_u * 2 + 1)] = acc5__c0_1;
}
