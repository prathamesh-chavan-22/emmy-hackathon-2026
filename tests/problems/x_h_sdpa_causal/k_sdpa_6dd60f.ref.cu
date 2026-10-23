#include <cuda_fp16.h>
static __device__ __forceinline__ void emmy_ldmatrix_x4(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

static __device__ __forceinline__ void emmy_ldmatrix_x2_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// x4.trans: two col-adjacent canonical-B fragments in one ldmatrix — lanes 0-15
// address the 16 K rows at the first fragment's column, lanes 16-31 at col+8
// (the 096_pair_ldmatrix_loads fusion; r[0..1] / r[2..3] are the two fragments).
static __device__ __forceinline__ void emmy_ldmatrix_x4_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// Paired x4 → two B fragments in one ldmatrix (096_pair_ldmatrix_loads): the four
// loaded registers land DIRECTLY as b0[0..1] (lanes 0-15) and b1[0..1] (lanes 16-31)
// — no staging temp / register shuffle. ``_pair`` is the plain x4 (transposed-B slab,
// N-adjacent pair); ``_trans_pair`` is x4.trans (canonical-B slab, col-adjacent pair).
static __device__ __forceinline__ void emmy_ldmatrix_x4_pair(unsigned* b0, unsigned* b1, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

static __device__ __forceinline__ void emmy_ldmatrix_x4_trans_pair(unsigned* b0, unsigned* b1, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

// Plain (no .trans) x2: a transposed-B operand staged as its native N-major
// slab (Q@K^T's K rows) — each 8x8 matrix's rows ARE the mma B fragment's
// col-major columns, so no transpose is needed (cf. emmy_mma_load_b_gmem_trans).
static __device__ __forceinline__ void emmy_ldmatrix_x2(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// C->A register repack: the m16n8k16 f32 C fragment is lane-map ALIGNED with the
// 16-bit A fragment's k-halves, so two k-adjacent C fragments (c0 low, c1 high)
// convert into one A operand fragment per lane — no shuffle, no smem round-trip.
// cvt.rn packs (hi, lo) with round-to-nearest-even — the same rounding the
// retired smem handoff's RegStore vec2 pack applied, so the repack is bit-identical.
static __device__ __forceinline__ void emmy_c_to_a_f16(unsigned* a, const float* c0, const float* c1) {
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[0]) : "f"(c0[1]), "f"(c0[0]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[1]) : "f"(c0[3]), "f"(c0[2]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[2]) : "f"(c1[1]), "f"(c1[0]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[3]) : "f"(c1[3]), "f"(c1[2]));
}

static __device__ __forceinline__ void emmy_c_to_a_bf16(unsigned* a, const float* c0, const float* c1) {
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[0]) : "f"(c0[1]), "f"(c0[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[1]) : "f"(c0[3]), "f"(c0[2]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[2]) : "f"(c1[1]), "f"(c1[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[3]) : "f"(c1[3]), "f"(c1[2]));
}

// gmem-direct fragment loads — the fallback when an mma.sync operand was NOT
// staged into shared memory (ldmatrix is smem-only, so we read the fragment
// straight from gmem instead, replicating the PTX m16n8k16 lane→element map).
// Slower than ldmatrix (no smem reuse) but correct; ``005_lower_atom_tile``
// emits these for an unstaged operand. ``ldm`` is the operand's gmem row stride
// (K for the row-major A[M,K]; N for the row-major B[K,N]); ``g`` points at the
// atom cell's base element, each lane adds its own (row,col) within the tile.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 1) + ((i & 2) ? 8 : 0);   // K: 2*threadID_in_group, +8 for the k16 half
        const T* p = g + row * ldm + col;
        unsigned packed;
        ((F*)&packed)[0] = F(p[0]);                     // .f16x2: low half = col, high half = col+1
        ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed;
        ((F*)&packed)[0] = F(g[k * ldm + n]);           // .f16x2: low half = k, high half = k+1
        ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Masked-tile (M9) variants of the gmem-direct fragment loads: a tile straddling a masked axis's
// bound would read rows / cols past the runtime-sized buffer, so the lane coordinate on the gated
// axis clamps INTO the live range — ``left - 1`` (``left`` = in-range elements from the tile base)
// where the fragment straddles the bound, the tile base where it OVERHANGS it entirely. The latter
// is a tile wider than its axis (M=1 under a 128-row tile): ``left`` goes <= 0, and a bare
// ``left - 1`` addresses tens of KB BELOW the buffer — an out-of-bounds read that faults the context
// wherever that memory is unmapped. Clamped lanes read a duplicate in-bounds value — harmless,
// their stores are masked by the RegStore guard (the tile path's ``clamp_last``, same contract).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp(unsigned* r, const T* g, int ldm, int rows_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = max(rows_left - 1, 0);   // M: clamp to the runtime extent
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed;
        ((F*)&packed)[0] = F(p[0]);
        ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp(unsigned* r, const T* g, int ldm, int cols_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed;
        ((F*)&packed)[0] = F(g[k * ldm + n]);
        ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Transposed-B (Q @ K^T): B stored N×K (``g[n][k]``, K contiguous) — the native
// ``mma.row.col`` col-major B, so no ldmatrix ``.trans`` is needed. The fragment
// lane→element map is the same (n = groupID, k = 2·threadID_in_group + k16 half),
// but each lane now reads a contiguous (k, k+1) pair from row ``n`` of B. ``ldm``
// is B's gmem row stride (the K extent). Mirrors ``emmy_mma_load_b_gmem`` with the
// (k, n) index roles swapped to (n, k).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed;
        ((F*)&packed)[0] = F(g[n * ldm + k]);           // .f16x2: contiguous (k, k+1) in row n
        ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Masked-tile variant of the transposed-B gmem load: the gated N axis is the
// row (``n``) here, so clamp ``n`` to the runtime extent (cf. _b_gmem_nclamp,
// which clamps N as the column). Clamped lanes read a duplicate in-bounds row;
// their stores are masked by the RegStore guard.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp(unsigned* r, const T* g, int ldm, int cols_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed;
        ((F*)&packed)[0] = F(g[n * ldm + k]);
        ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Masked-K (symbolic reduce) variants of the gmem-direct fragment loads: when the
// operand's REDUCE (K) axis is mask-padded (SDPA P@V over ``seq_len``), the final
// K tile straddles the runtime extent. Unlike the M/N clamp above, a K element
// past the extent must be ZERO-FILLED, not clamped to a duplicate — K is summed
// by the mma, so a duplicate corrupts the reduction. ``k_left`` = in-range K
// elements from the tile base; a half past it reads as +0.0 and is never
// dereferenced. Mirrors the staged path's slab zero-fill (``_stage_expand``).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed = 0;
        if (col < k_left) ((F*)&packed)[0] = F(p[0]);
        if (col + 1 < k_left) ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

// A: masked-M (clamp rows) AND masked-K (zero-fill cols).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp_kzero(unsigned* r, const T* g, int ldm, int rows_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = max(rows_left - 1, 0);
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed = 0;
        if (col < k_left) ((F*)&packed)[0] = F(p[0]);
        if (col + 1 < k_left) ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

// B (row-major K×N, NOT transposed): K is the row, zero-fill past the extent.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[k * ldm + n]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// B: masked-N (clamp col) AND masked-K (zero-fill row).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp_kzero(unsigned* r, const T* g, int ldm, int cols_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[k * ldm + n]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Transposed-B (Q@K^T, ``g[n][k]`` with K contiguous) masked-K: zero-fill the K
// halves past the runtime extent. K is summed by the mma, so a past-extent element
// must read as +0.0 (never a duplicate). Mirrors ``emmy_mma_load_b_gmem_kzero`` with
// the (k, n) index roles swapped to (n, k) — cf. ``emmy_mma_load_b_gmem_trans``.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[n * ldm + k]);       // .f16x2: contiguous (k, k+1) in row n
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Transposed-B: masked-N (clamp the ``n`` row) AND masked-K (zero-fill the contiguous k).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp_kzero(
    unsigned* r, const T* g, int ldm, int cols_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[n * ldm + k]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

static __device__ __forceinline__ void emmy_mma_m16n8k16_f16_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void emmy_mma_m16n8k16_bf16_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// f16-accumulate HMMA (the ``mma_m16n8k16_f16_f16`` atom): on the consumer GeForce dies the .f32-accumulate
// form above runs at HALF the tensor-core rate, so this variant keeps the whole mma chain on the
// full-rate .f16 accumulator — ``c``/``d`` are 2 packed b32 regs (two halfs each, the same
// element map as the four f32 regs, pair-packed). Paired with emmy_mma_promote_f16acc below,
// which periodically folds the f16 partials into the f32 shadow accumulator.
static __device__ __forceinline__ void emmy_mma_m16n8k16_f16_f16(unsigned* d, const unsigned* a, const unsigned* b, const unsigned* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]));
}

// The chunk promote of the f16-accumulate scheme: fold the packed f16 C fragment into the f32
// shadow accumulator and rezero it, bounding the f16 accumulation error to one K chunk. Inline
// PTX cvt (not __half2 intrinsics) keeps this prelude header-free, like the wrappers above.
static __device__ __forceinline__ void emmy_mma_promote_f16acc(float* c, unsigned* h) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float lo, hi;
        asm("{.reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "cvt.f32.f16 %0, lo;\n\t"
            "cvt.f32.f16 %1, hi;}\n"
            : "=f"(lo), "=f"(hi) : "r"(h[i]));
        c[2 * i] += lo;
        c[2 * i + 1] += hi;
        h[i] = 0u;
    }
}

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

// B32 slab swizzle: XOR a b16 smem element index the way the fill permuted the
// 16-byte chunks (TMA in-copy, or the same XOR on a cp.async fill destination);
// the ldmatrix drain must read them back through the identical permutation.
static __device__ __forceinline__ int emmy_swizzle_b32(int e) {
    return e ^ (((e >> 6) & 1) << 3);
}

// B64 slab swizzle: XOR a b16 smem element index the way the fill permuted the
// 16-byte chunks (TMA in-copy, or the same XOR on a cp.async fill destination);
// the ldmatrix drain must read them back through the identical permutation.
static __device__ __forceinline__ int emmy_swizzle_b64(int e) {
    return e ^ (((e >> 6) & 3) << 3);
}

extern "C" __global__
__launch_bounds__(128) void k_sdpa_6dd60f(const __half* scaled_dot_product_attention_mask_fill_c, const __half* scaled_dot_product_attention_mask_zero_c, const __half* scaled_dot_product_attention_scale, const __half* x1, const __half* x0, const __half* x2, __half* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 256;
        int a1_b = (_gid / 256) % 1;
        int a6_b = (_gid / 128) % 2;
        int a1_u = (_gid / 32) % 4;
        int a6_u = (_gid / 32) % 1;
        int _lane = (_gid) % 32;
        __half in0 = scaled_dot_product_attention_mask_fill_c[0];
        float _c0_[2][4] = {};
        float _c1_[2][4] = {};
        float _c2_[2][4] = {};
        float _c3_[2][4] = {};
        float acc1__r0_0 = -1e+30f;
        float acc1__r0_1 = -1e+30f;
        float acc1__r1_0 = -1e+30f;
        float acc1__r1_1 = -1e+30f;
        float acc1__r2_0 = -1e+30f;
        float acc1__r2_1 = -1e+30f;
        float acc1__r3_0 = -1e+30f;
        float acc1__r3_1 = -1e+30f;
        float acc3__r0_0 = 0.0f;
        float acc3__r0_1 = 0.0f;
        float acc3__r1_0 = 0.0f;
        float acc3__r1_1 = 0.0f;
        float acc3__r2_0 = 0.0f;
        float acc3__r2_1 = 0.0f;
        float acc3__r3_0 = 0.0f;
        float acc3__r3_1 = 0.0f;
        unsigned _qa0_0[4];
        emmy_mma_load_a_gmem_mclamp(_qa0_0, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64) * 32], 32, (64) - (a1_b * 256 + a1_u * 64 + 0));
        unsigned _qa0_1[4];
        emmy_mma_load_a_gmem_mclamp(_qa0_1, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64) * 32 + 16], 32, (64) - (a1_b * 256 + a1_u * 64 + 0));
        unsigned _qa1_0[4];
        emmy_mma_load_a_gmem_mclamp(_qa1_0, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 16) * 32], 32, (64) - (a1_b * 256 + a1_u * 64 + 16));
        unsigned _qa1_1[4];
        emmy_mma_load_a_gmem_mclamp(_qa1_1, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 16) * 32 + 16], 32, (64) - (a1_b * 256 + a1_u * 64 + 16));
        unsigned _qa2_0[4];
        emmy_mma_load_a_gmem_mclamp(_qa2_0, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 32) * 32], 32, (64) - (a1_b * 256 + a1_u * 64 + 32));
        unsigned _qa2_1[4];
        emmy_mma_load_a_gmem_mclamp(_qa2_1, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 32) * 32 + 16], 32, (64) - (a1_b * 256 + a1_u * 64 + 32));
        unsigned _qa3_0[4];
        emmy_mma_load_a_gmem_mclamp(_qa3_0, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 48) * 32], 32, (64) - (a1_b * 256 + a1_u * 64 + 48));
        unsigned _qa3_1[4];
        emmy_mma_load_a_gmem_mclamp(_qa3_1, &x0[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 48) * 32 + 16], 32, (64) - (a1_b * 256 + a1_u * 64 + 48));
        __shared__ __align__(512) __half _a_smem[2048];
        __shared__ __align__(256) __half _b_smem[1024];
        emmy_cp_async_cg(&_a_smem[(emmy_swizzle_b64((((a1_u * 1 + a6_u) * 32 + _lane) * 8 / 32) * 32 + (((a1_u * 1 + a6_u) * 32 + _lane) * 8 % 32)) ^ emmy_swizzle_b64(0)) + (0) * 32], &x1[((a1_u + a6_u) * 32 + _lane) * 8 + a0 * 2048]);
        for (int _fb = (a1_u * 1 + a6_u) * 32 + _lane; _fb < 64; _fb += 128) {
            emmy_cp_async_cg(&_b_smem[emmy_swizzle_b32(_fb * 8)], &x2[a0 * 2048 + _fb / 2 * 32 + (a6_b * 16 + 8 * (_fb % 2))]);
        }
        emmy_cp_async_commit();
        __half in1 = scaled_dot_product_attention_mask_zero_c[0];
        __half in2 = scaled_dot_product_attention_scale[0];
        for (int a2__ck = 0; a2__ck < 64; a2__ck += 32) {
            emmy_cp_async_cg(&_a_smem[(emmy_swizzle_b64((((a1_u * 1 + a6_u) * 32 + _lane) * 8 / 32) * 32 + (((a1_u * 1 + a6_u) * 32 + _lane) * 8 % 32)) ^ emmy_swizzle_b64(0)) + ((a2__ck / 32 + 1) % 2 * 32 + 0) * 32], &x1[a0 * 2048 + (((a2__ck + 32 < 64) ? (a2__ck + 32) : (32)) + ((a1_u + a6_u) * 32 + _lane) / 4) * 32 + 8 * (((a1_u + a6_u) * 32 + _lane) % 4)]);
            for (int _fb = (a1_u * 1 + a6_u) * 32 + _lane; _fb < 64; _fb += 128) {
                emmy_cp_async_cg(&_b_smem[emmy_swizzle_b32(((a2__ck / 32 + 1) % 2 * 32 + _fb / 2) * 16 + 8 * (_fb % 2))], &x2[a0 * 2048 + (((a2__ck + 32 < 64) ? (a2__ck + 32) : (32)) + _fb / 2) * 32 + (a6_b * 16 + 8 * (_fb % 2))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<1>();
            __syncthreads();
            unsigned _kb[4][2];
            float _s0_[4][4] = {};
            float _s1_[4][4] = {};
            float _s2_[4][4] = {};
            float _s3_[4][4] = {};
            emmy_ldmatrix_x4_pair(_kb[0], _kb[1], &_a_smem[(emmy_swizzle_b64((((threadIdx.x & 31) % 8) + ((threadIdx.x & 31) / 16) * 8) * 32 + (((threadIdx.x & 31) / 8) % 2) * 8) ^ emmy_swizzle_b64(0)) + (a2__ck / 32 % 2 * 32 + 0) * 32]);
            emmy_ldmatrix_x4_pair(_kb[2], _kb[3], &_a_smem[(emmy_swizzle_b64((((threadIdx.x & 31) % 8) + ((threadIdx.x & 31) / 16) * 8) * 32 + (((threadIdx.x & 31) / 8) % 2) * 8) ^ emmy_swizzle_b64(0)) + (a2__ck / 32 % 2 * 32 + 16) * 32]);
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s0_[_r0], _qa0_0, _kb[_r0], _s0_[_r0]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s1_[_r0], _qa1_0, _kb[_r0], _s1_[_r0]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s2_[_r0], _qa2_0, _kb[_r0], _s2_[_r0]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s3_[_r0], _qa3_0, _kb[_r0], _s3_[_r0]);
            }
            emmy_ldmatrix_x4_pair(_kb[0], _kb[1], &_a_smem[(emmy_swizzle_b64((((threadIdx.x & 31) % 8) + ((threadIdx.x & 31) / 16) * 8) * 32 + (((threadIdx.x & 31) / 8) % 2) * 8) ^ emmy_swizzle_b64(16)) + (a2__ck / 32 % 2 * 32 + 0) * 32]);
            emmy_ldmatrix_x4_pair(_kb[2], _kb[3], &_a_smem[(emmy_swizzle_b64((((threadIdx.x & 31) % 8) + ((threadIdx.x & 31) / 16) * 8) * 32 + (((threadIdx.x & 31) / 8) % 2) * 8) ^ emmy_swizzle_b64(16)) + (a2__ck / 32 % 2 * 32 + 16) * 32]);
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s0_[_r0], _qa0_1, _kb[_r0], _s0_[_r0]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s1_[_r0], _qa1_1, _kb[_r0], _s1_[_r0]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s2_[_r0], _qa2_1, _kb[_r0], _s2_[_r0]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 4; _r0 += 1) {
                emmy_mma_m16n8k16_f16_f32(_s3_[_r0], _qa3_1, _kb[_r0], _s3_[_r0]);
            }
            float _w0_0_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_0_v1[_e] = _s0_[0][_e] * __half2float(in2);
            }
            float _w0_0_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_0_v3[_e] = _w0_0_v1[_e] + __half2float(in1);
            }
            const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
            _w0_0_v3[0] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_0_v3[0]));
            _w0_0_v3[1] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_0_v3[1]));
            _w0_0_v3[2] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_0_v3[2]));
            _w0_0_v3[3] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_0_v3[3]));
            float _w0_1_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_1_v1[_e] = _s0_[1][_e] * __half2float(in2);
            }
            float _w0_1_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_1_v3[_e] = _w0_1_v1[_e] + __half2float(in1);
            }
            _w0_1_v3[0] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_1_v3[0]));
            _w0_1_v3[1] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_1_v3[1]));
            _w0_1_v3[2] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_1_v3[2]));
            _w0_1_v3[3] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_1_v3[3]));
            float _w0_2_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_2_v1[_e] = _s0_[2][_e] * __half2float(in2);
            }
            float _w0_2_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_2_v3[_e] = _w0_2_v1[_e] + __half2float(in1);
            }
            _w0_2_v3[0] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_2_v3[0]));
            _w0_2_v3[1] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_2_v3[1]));
            _w0_2_v3[2] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_2_v3[2]));
            _w0_2_v3[3] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_2_v3[3]));
            float _w0_3_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_3_v1[_e] = _s0_[3][_e] * __half2float(in2);
            }
            float _w0_3_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w0_3_v3[_e] = _w0_3_v1[_e] + __half2float(in1);
            }
            _w0_3_v3[0] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_3_v3[0]));
            _w0_3_v3[1] = ((a1_b * 256 + a1_u * 64 + 0 + _g < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_3_v3[1]));
            _w0_3_v3[2] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w0_3_v3[2]));
            _w0_3_v3[3] = ((a1_b * 256 + a1_u * 64 + 0 + (_g + 8) < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w0_3_v3[3]));
            float _g__r0_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w0_0_v3[0], _w0_0_v3[1]), _w0_1_v3[0]), _w0_1_v3[1]), _w0_2_v3[0]), _w0_2_v3[1]), _w0_3_v3[0]), _w0_3_v3[1]);
            float _g__r0_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w0_0_v3[2], _w0_0_v3[3]), _w0_1_v3[2]), _w0_1_v3[3]), _w0_2_v3[2]), _w0_2_v3[3]), _w0_3_v3[2]), _w0_3_v3[3]);
            _g__r0_0 = fmaxf(_g__r0_0, __shfl_xor_sync(0xffffffff, _g__r0_0, 2));
            _g__r0_1 = fmaxf(_g__r0_1, __shfl_xor_sync(0xffffffff, _g__r0_1, 2));
            _g__r0_0 = fmaxf(_g__r0_0, __shfl_xor_sync(0xffffffff, _g__r0_0, 1));
            _g__r0_1 = fmaxf(_g__r0_1, __shfl_xor_sync(0xffffffff, _g__r0_1, 1));
            float _mg0_gn_0 = fmaxf(_g__r0_0, acc1__r0_0);
            float _mg0_gn_1 = fmaxf(_g__r0_1, acc1__r0_1);
            float _mg0_alpha_0 = expf((acc1__r0_0 - _mg0_gn_0));
            float _mg0_alpha_1 = expf((acc1__r0_1 - _mg0_gn_1));
            float _w1_0_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_0_v1[_e] = _s1_[0][_e] * __half2float(in2);
            }
            float _w1_0_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_0_v3[_e] = _w1_0_v1[_e] + __half2float(in1);
            }
            _w1_0_v3[0] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_0_v3[0]));
            _w1_0_v3[1] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_0_v3[1]));
            _w1_0_v3[2] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_0_v3[2]));
            _w1_0_v3[3] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_0_v3[3]));
            float _w1_1_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_1_v1[_e] = _s1_[1][_e] * __half2float(in2);
            }
            float _w1_1_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_1_v3[_e] = _w1_1_v1[_e] + __half2float(in1);
            }
            _w1_1_v3[0] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_1_v3[0]));
            _w1_1_v3[1] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_1_v3[1]));
            _w1_1_v3[2] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_1_v3[2]));
            _w1_1_v3[3] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_1_v3[3]));
            float _w1_2_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_2_v1[_e] = _s1_[2][_e] * __half2float(in2);
            }
            float _w1_2_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_2_v3[_e] = _w1_2_v1[_e] + __half2float(in1);
            }
            _w1_2_v3[0] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_2_v3[0]));
            _w1_2_v3[1] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_2_v3[1]));
            _w1_2_v3[2] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_2_v3[2]));
            _w1_2_v3[3] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_2_v3[3]));
            float _w1_3_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_3_v1[_e] = _s1_[3][_e] * __half2float(in2);
            }
            float _w1_3_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w1_3_v3[_e] = _w1_3_v1[_e] + __half2float(in1);
            }
            _w1_3_v3[0] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_3_v3[0]));
            _w1_3_v3[1] = ((a1_b * 256 + a1_u * 64 + 16 + _g < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_3_v3[1]));
            _w1_3_v3[2] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w1_3_v3[2]));
            _w1_3_v3[3] = ((a1_b * 256 + a1_u * 64 + 16 + (_g + 8) < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w1_3_v3[3]));
            float _g__r1_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w1_0_v3[0], _w1_0_v3[1]), _w1_1_v3[0]), _w1_1_v3[1]), _w1_2_v3[0]), _w1_2_v3[1]), _w1_3_v3[0]), _w1_3_v3[1]);
            float _g__r1_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w1_0_v3[2], _w1_0_v3[3]), _w1_1_v3[2]), _w1_1_v3[3]), _w1_2_v3[2]), _w1_2_v3[3]), _w1_3_v3[2]), _w1_3_v3[3]);
            _g__r1_0 = fmaxf(_g__r1_0, __shfl_xor_sync(0xffffffff, _g__r1_0, 2));
            _g__r1_1 = fmaxf(_g__r1_1, __shfl_xor_sync(0xffffffff, _g__r1_1, 2));
            _g__r1_0 = fmaxf(_g__r1_0, __shfl_xor_sync(0xffffffff, _g__r1_0, 1));
            _g__r1_1 = fmaxf(_g__r1_1, __shfl_xor_sync(0xffffffff, _g__r1_1, 1));
            float _mg1_gn_0 = fmaxf(_g__r1_0, acc1__r1_0);
            float _mg1_gn_1 = fmaxf(_g__r1_1, acc1__r1_1);
            float _mg1_alpha_0 = expf((acc1__r1_0 - _mg1_gn_0));
            float _mg1_alpha_1 = expf((acc1__r1_1 - _mg1_gn_1));
            float _w2_0_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_0_v1[_e] = _s2_[0][_e] * __half2float(in2);
            }
            float _w2_0_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_0_v3[_e] = _w2_0_v1[_e] + __half2float(in1);
            }
            _w2_0_v3[0] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_0_v3[0]));
            _w2_0_v3[1] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_0_v3[1]));
            _w2_0_v3[2] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_0_v3[2]));
            _w2_0_v3[3] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_0_v3[3]));
            float _w2_1_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_1_v1[_e] = _s2_[1][_e] * __half2float(in2);
            }
            float _w2_1_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_1_v3[_e] = _w2_1_v1[_e] + __half2float(in1);
            }
            _w2_1_v3[0] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_1_v3[0]));
            _w2_1_v3[1] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_1_v3[1]));
            _w2_1_v3[2] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_1_v3[2]));
            _w2_1_v3[3] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_1_v3[3]));
            float _w2_2_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_2_v1[_e] = _s2_[2][_e] * __half2float(in2);
            }
            float _w2_2_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_2_v3[_e] = _w2_2_v1[_e] + __half2float(in1);
            }
            _w2_2_v3[0] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_2_v3[0]));
            _w2_2_v3[1] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_2_v3[1]));
            _w2_2_v3[2] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_2_v3[2]));
            _w2_2_v3[3] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_2_v3[3]));
            float _w2_3_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_3_v1[_e] = _s2_[3][_e] * __half2float(in2);
            }
            float _w2_3_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w2_3_v3[_e] = _w2_3_v1[_e] + __half2float(in1);
            }
            _w2_3_v3[0] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_3_v3[0]));
            _w2_3_v3[1] = ((a1_b * 256 + a1_u * 64 + 32 + _g < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_3_v3[1]));
            _w2_3_v3[2] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w2_3_v3[2]));
            _w2_3_v3[3] = ((a1_b * 256 + a1_u * 64 + 32 + (_g + 8) < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w2_3_v3[3]));
            float _g__r2_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w2_0_v3[0], _w2_0_v3[1]), _w2_1_v3[0]), _w2_1_v3[1]), _w2_2_v3[0]), _w2_2_v3[1]), _w2_3_v3[0]), _w2_3_v3[1]);
            float _g__r2_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w2_0_v3[2], _w2_0_v3[3]), _w2_1_v3[2]), _w2_1_v3[3]), _w2_2_v3[2]), _w2_2_v3[3]), _w2_3_v3[2]), _w2_3_v3[3]);
            _g__r2_0 = fmaxf(_g__r2_0, __shfl_xor_sync(0xffffffff, _g__r2_0, 2));
            _g__r2_1 = fmaxf(_g__r2_1, __shfl_xor_sync(0xffffffff, _g__r2_1, 2));
            _g__r2_0 = fmaxf(_g__r2_0, __shfl_xor_sync(0xffffffff, _g__r2_0, 1));
            _g__r2_1 = fmaxf(_g__r2_1, __shfl_xor_sync(0xffffffff, _g__r2_1, 1));
            float _mg2_gn_0 = fmaxf(_g__r2_0, acc1__r2_0);
            float _mg2_gn_1 = fmaxf(_g__r2_1, acc1__r2_1);
            float _mg2_alpha_0 = expf((acc1__r2_0 - _mg2_gn_0));
            float _mg2_alpha_1 = expf((acc1__r2_1 - _mg2_gn_1));
            float _w3_0_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_0_v1[_e] = _s3_[0][_e] * __half2float(in2);
            }
            float _w3_0_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_0_v3[_e] = _w3_0_v1[_e] + __half2float(in1);
            }
            _w3_0_v3[0] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_0_v3[0]));
            _w3_0_v3[1] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_0_v3[1]));
            _w3_0_v3[2] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 0 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_0_v3[2]));
            _w3_0_v3[3] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 0 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_0_v3[3]));
            float _w3_1_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_1_v1[_e] = _s3_[1][_e] * __half2float(in2);
            }
            float _w3_1_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_1_v3[_e] = _w3_1_v1[_e] + __half2float(in1);
            }
            _w3_1_v3[0] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_1_v3[0]));
            _w3_1_v3[1] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_1_v3[1]));
            _w3_1_v3[2] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 8 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_1_v3[2]));
            _w3_1_v3[3] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 8 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_1_v3[3]));
            float _w3_2_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_2_v1[_e] = _s3_[2][_e] * __half2float(in2);
            }
            float _w3_2_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_2_v3[_e] = _w3_2_v1[_e] + __half2float(in1);
            }
            _w3_2_v3[0] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_2_v3[0]));
            _w3_2_v3[1] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_2_v3[1]));
            _w3_2_v3[2] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 16 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_2_v3[2]));
            _w3_2_v3[3] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 16 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_2_v3[3]));
            float _w3_3_v1[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_3_v1[_e] = _s3_[3][_e] * __half2float(in2);
            }
            float _w3_3_v3[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _w3_3_v3[_e] = _w3_3_v1[_e] + __half2float(in1);
            }
            _w3_3_v3[0] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_3_v3[0]));
            _w3_3_v3[1] = ((a1_b * 256 + a1_u * 64 + 48 + _g < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_3_v3[1]));
            _w3_3_v3[2] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 24 + (_t * 2 + 0)) ? (-1e+30f) : (_w3_3_v3[2]));
            _w3_3_v3[3] = ((a1_b * 256 + a1_u * 64 + 48 + (_g + 8) < a2__ck + 24 + (_t * 2 + 1)) ? (-1e+30f) : (_w3_3_v3[3]));
            float _g__r3_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w3_0_v3[0], _w3_0_v3[1]), _w3_1_v3[0]), _w3_1_v3[1]), _w3_2_v3[0]), _w3_2_v3[1]), _w3_3_v3[0]), _w3_3_v3[1]);
            float _g__r3_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_w3_0_v3[2], _w3_0_v3[3]), _w3_1_v3[2]), _w3_1_v3[3]), _w3_2_v3[2]), _w3_2_v3[3]), _w3_3_v3[2]), _w3_3_v3[3]);
            _g__r3_0 = fmaxf(_g__r3_0, __shfl_xor_sync(0xffffffff, _g__r3_0, 2));
            _g__r3_1 = fmaxf(_g__r3_1, __shfl_xor_sync(0xffffffff, _g__r3_1, 2));
            _g__r3_0 = fmaxf(_g__r3_0, __shfl_xor_sync(0xffffffff, _g__r3_0, 1));
            _g__r3_1 = fmaxf(_g__r3_1, __shfl_xor_sync(0xffffffff, _g__r3_1, 1));
            float _mg3_gn_0 = fmaxf(_g__r3_0, acc1__r3_0);
            float _mg3_gn_1 = fmaxf(_g__r3_1, acc1__r3_1);
            float _mg3_alpha_0 = expf((acc1__r3_0 - _mg3_gn_0));
            float _mg3_alpha_1 = expf((acc1__r3_1 - _mg3_gn_1));
            float _ch0_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_0_d[_e] = _w0_0_v3[_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_0_w[_e] = expf(_ch0_0_d[_e]);
            }
            float _ch0_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_1_d[_e] = _w0_1_v3[_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_1_w[_e] = expf(_ch0_1_d[_e]);
            }
            float _ch0_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_2_d[_e] = _w0_2_v3[_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_2_w[_e] = expf(_ch0_2_d[_e]);
            }
            float _ch0_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_3_d[_e] = _w0_3_v3[_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_3_w[_e] = expf(_ch0_3_d[_e]);
            }
            float acc3__p0_0 = _ch0_0_w[0] + _ch0_0_w[1] + _ch0_1_w[0] + _ch0_1_w[1] + _ch0_2_w[0] + _ch0_2_w[1] + _ch0_3_w[0] + _ch0_3_w[1];
            float acc3__p0_1 = _ch0_0_w[2] + _ch0_0_w[3] + _ch0_1_w[2] + _ch0_1_w[3] + _ch0_2_w[2] + _ch0_2_w[3] + _ch0_3_w[2] + _ch0_3_w[3];
            acc3__p0_0 = acc3__p0_0 + __shfl_xor_sync(0xffffffff, acc3__p0_0, 2);
            acc3__p0_1 = acc3__p0_1 + __shfl_xor_sync(0xffffffff, acc3__p0_1, 2);
            acc3__p0_0 = acc3__p0_0 + __shfl_xor_sync(0xffffffff, acc3__p0_0, 1);
            acc3__p0_1 = acc3__p0_1 + __shfl_xor_sync(0xffffffff, acc3__p0_1, 1);
            float _mg0_acc3___scaled_0 = _mg0_alpha_0 * acc3__r0_0;
            float _mg0_acc3___scaled_1 = _mg0_alpha_1 * acc3__r0_1;
            acc3__r0_0 = _mg0_acc3___scaled_0 + acc3__p0_0;
            acc3__r0_1 = _mg0_acc3___scaled_1 + acc3__p0_1;
            float _ch1_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_0_d[_e] = _w1_0_v3[_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_0_w[_e] = expf(_ch1_0_d[_e]);
            }
            float _ch1_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_1_d[_e] = _w1_1_v3[_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_1_w[_e] = expf(_ch1_1_d[_e]);
            }
            float _ch1_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_2_d[_e] = _w1_2_v3[_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_2_w[_e] = expf(_ch1_2_d[_e]);
            }
            float _ch1_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_3_d[_e] = _w1_3_v3[_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_3_w[_e] = expf(_ch1_3_d[_e]);
            }
            float acc3__p1_0 = _ch1_0_w[0] + _ch1_0_w[1] + _ch1_1_w[0] + _ch1_1_w[1] + _ch1_2_w[0] + _ch1_2_w[1] + _ch1_3_w[0] + _ch1_3_w[1];
            float acc3__p1_1 = _ch1_0_w[2] + _ch1_0_w[3] + _ch1_1_w[2] + _ch1_1_w[3] + _ch1_2_w[2] + _ch1_2_w[3] + _ch1_3_w[2] + _ch1_3_w[3];
            acc3__p1_0 = acc3__p1_0 + __shfl_xor_sync(0xffffffff, acc3__p1_0, 2);
            acc3__p1_1 = acc3__p1_1 + __shfl_xor_sync(0xffffffff, acc3__p1_1, 2);
            acc3__p1_0 = acc3__p1_0 + __shfl_xor_sync(0xffffffff, acc3__p1_0, 1);
            acc3__p1_1 = acc3__p1_1 + __shfl_xor_sync(0xffffffff, acc3__p1_1, 1);
            float _mg1_acc3___scaled_0 = _mg1_alpha_0 * acc3__r1_0;
            float _mg1_acc3___scaled_1 = _mg1_alpha_1 * acc3__r1_1;
            acc3__r1_0 = _mg1_acc3___scaled_0 + acc3__p1_0;
            acc3__r1_1 = _mg1_acc3___scaled_1 + acc3__p1_1;
            float _ch2_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_0_d[_e] = _w2_0_v3[_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_0_w[_e] = expf(_ch2_0_d[_e]);
            }
            float _ch2_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_1_d[_e] = _w2_1_v3[_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_1_w[_e] = expf(_ch2_1_d[_e]);
            }
            float _ch2_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_2_d[_e] = _w2_2_v3[_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_2_w[_e] = expf(_ch2_2_d[_e]);
            }
            float _ch2_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_3_d[_e] = _w2_3_v3[_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_3_w[_e] = expf(_ch2_3_d[_e]);
            }
            float acc3__p2_0 = _ch2_0_w[0] + _ch2_0_w[1] + _ch2_1_w[0] + _ch2_1_w[1] + _ch2_2_w[0] + _ch2_2_w[1] + _ch2_3_w[0] + _ch2_3_w[1];
            float acc3__p2_1 = _ch2_0_w[2] + _ch2_0_w[3] + _ch2_1_w[2] + _ch2_1_w[3] + _ch2_2_w[2] + _ch2_2_w[3] + _ch2_3_w[2] + _ch2_3_w[3];
            acc3__p2_0 = acc3__p2_0 + __shfl_xor_sync(0xffffffff, acc3__p2_0, 2);
            acc3__p2_1 = acc3__p2_1 + __shfl_xor_sync(0xffffffff, acc3__p2_1, 2);
            acc3__p2_0 = acc3__p2_0 + __shfl_xor_sync(0xffffffff, acc3__p2_0, 1);
            acc3__p2_1 = acc3__p2_1 + __shfl_xor_sync(0xffffffff, acc3__p2_1, 1);
            float _mg2_acc3___scaled_0 = _mg2_alpha_0 * acc3__r2_0;
            float _mg2_acc3___scaled_1 = _mg2_alpha_1 * acc3__r2_1;
            acc3__r2_0 = _mg2_acc3___scaled_0 + acc3__p2_0;
            acc3__r2_1 = _mg2_acc3___scaled_1 + acc3__p2_1;
            float _ch3_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_0_d[_e] = _w3_0_v3[_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_0_w[_e] = expf(_ch3_0_d[_e]);
            }
            float _ch3_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_1_d[_e] = _w3_1_v3[_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_1_w[_e] = expf(_ch3_1_d[_e]);
            }
            float _ch3_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_2_d[_e] = _w3_2_v3[_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_2_w[_e] = expf(_ch3_2_d[_e]);
            }
            float _ch3_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_3_d[_e] = _w3_3_v3[_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_3_w[_e] = expf(_ch3_3_d[_e]);
            }
            float acc3__p3_0 = _ch3_0_w[0] + _ch3_0_w[1] + _ch3_1_w[0] + _ch3_1_w[1] + _ch3_2_w[0] + _ch3_2_w[1] + _ch3_3_w[0] + _ch3_3_w[1];
            float acc3__p3_1 = _ch3_0_w[2] + _ch3_0_w[3] + _ch3_1_w[2] + _ch3_1_w[3] + _ch3_2_w[2] + _ch3_2_w[3] + _ch3_3_w[2] + _ch3_3_w[3];
            acc3__p3_0 = acc3__p3_0 + __shfl_xor_sync(0xffffffff, acc3__p3_0, 2);
            acc3__p3_1 = acc3__p3_1 + __shfl_xor_sync(0xffffffff, acc3__p3_1, 2);
            acc3__p3_0 = acc3__p3_0 + __shfl_xor_sync(0xffffffff, acc3__p3_0, 1);
            acc3__p3_1 = acc3__p3_1 + __shfl_xor_sync(0xffffffff, acc3__p3_1, 1);
            float _mg3_acc3___scaled_0 = _mg3_alpha_0 * acc3__r3_0;
            float _mg3_acc3___scaled_1 = _mg3_alpha_1 * acc3__r3_1;
            acc3__r3_0 = _mg3_acc3___scaled_0 + acc3__p3_0;
            acc3__r3_1 = _mg3_acc3___scaled_1 + acc3__p3_1;
            #pragma unroll
            for (int _r0 = 0; _r0 < 2; _r0 += 1) {
                #pragma unroll
                for (int _e = 0; _e < 4; _e++) {
                    _c0_[_r0][_e] = _c0_[_r0][_e] * ((_e < 2) ? (_mg0_alpha_0) : (_mg0_alpha_1));
                }
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 2; _r0 += 1) {
                #pragma unroll
                for (int _e = 0; _e < 4; _e++) {
                    _c1_[_r0][_e] = _c1_[_r0][_e] * ((_e < 2) ? (_mg1_alpha_0) : (_mg1_alpha_1));
                }
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 2; _r0 += 1) {
                #pragma unroll
                for (int _e = 0; _e < 4; _e++) {
                    _c2_[_r0][_e] = _c2_[_r0][_e] * ((_e < 2) ? (_mg2_alpha_0) : (_mg2_alpha_1));
                }
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 2; _r0 += 1) {
                #pragma unroll
                for (int _e = 0; _e < 4; _e++) {
                    _c3_[_r0][_e] = _c3_[_r0][_e] * ((_e < 2) ? (_mg3_alpha_0) : (_mg3_alpha_1));
                }
            }
            unsigned _a0_0[4];
            unsigned _a0_1[4];
            unsigned _a1_0[4];
            unsigned _a1_1[4];
            unsigned _a2_0[4];
            unsigned _a2_1[4];
            unsigned _a3_0[4];
            unsigned _a3_1[4];
            unsigned _b0_0[2];
            unsigned _b0_1[2];
            unsigned _b1_0[2];
            unsigned _b1_1[2];
            emmy_c_to_a_f16(_a0_0, _ch0_0_w, _ch0_1_w);
            emmy_c_to_a_f16(_a1_0, _ch1_0_w, _ch1_1_w);
            emmy_c_to_a_f16(_a2_0, _ch2_0_w, _ch2_1_w);
            emmy_c_to_a_f16(_a3_0, _ch3_0_w, _ch3_1_w);
            emmy_ldmatrix_x4_trans_pair(_b0_0, _b1_0, &_b_smem[(emmy_swizzle_b32(((threadIdx.x & 31) % 16) * 16 + ((threadIdx.x & 31) / 16) * 8) ^ emmy_swizzle_b32(a6_u * 16 + 0)) + (a2__ck / 32 % 2 * 32 + 0) * 16]);
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_0, _b0_0, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_0, _b1_0, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_0, _b0_0, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_0, _b1_0, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_0, _b0_0, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_0, _b1_0, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_0, _b0_0, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_0, _b1_0, _c3_[1]);
            emmy_c_to_a_f16(_a0_1, _ch0_2_w, _ch0_3_w);
            emmy_c_to_a_f16(_a1_1, _ch1_2_w, _ch1_3_w);
            emmy_c_to_a_f16(_a2_1, _ch2_2_w, _ch2_3_w);
            emmy_c_to_a_f16(_a3_1, _ch3_2_w, _ch3_3_w);
            emmy_ldmatrix_x4_trans_pair(_b0_1, _b1_1, &_b_smem[(emmy_swizzle_b32(((threadIdx.x & 31) % 16) * 16 + ((threadIdx.x & 31) / 16) * 8) ^ emmy_swizzle_b32(a6_u * 16 + 0)) + (a2__ck / 32 % 2 * 32 + 16) * 16]);
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_1, _b0_1, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_1, _b1_1, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_1, _b0_1, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_1, _b1_1, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_1, _b0_1, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_1, _b1_1, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_1, _b0_1, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_1, _b1_1, _c3_[1]);
            acc1__r0_0 = fmaxf(acc1__r0_0, _g__r0_0);
            acc1__r0_1 = fmaxf(acc1__r0_1, _g__r0_1);
            acc1__r1_0 = fmaxf(acc1__r1_0, _g__r1_0);
            acc1__r1_1 = fmaxf(acc1__r1_1, _g__r1_1);
            acc1__r2_0 = fmaxf(acc1__r2_0, _g__r2_0);
            acc1__r2_1 = fmaxf(acc1__r2_1, _g__r2_1);
            acc1__r3_0 = fmaxf(acc1__r3_0, _g__r3_0);
            acc1__r3_1 = fmaxf(acc1__r3_1, _g__r3_1);
            __syncthreads();
        }
        float _ep0_0_v10_0 = 1.0f / acc3__r0_0;
        float _ep0_0_v10_1 = 1.0f / acc3__r0_1;
        float _ep0_0_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep0_0_acc5[_e] = _c0_[0][_e] * ((_e < 2) ? (_ep0_0_v10_0) : (_ep0_0_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 0) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64) * 32 + (a6_b * 16 + a6_u * 16) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep0_0_acc5[0], _ep0_0_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 0) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64) * 32 + (a6_b * 16 + a6_u * 16) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep0_0_acc5[2], _ep0_0_acc5[3]);
          }
        }
        float _ep0_1_v10_0 = 1.0f / acc3__r0_0;
        float _ep0_1_v10_1 = 1.0f / acc3__r0_1;
        float _ep0_1_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep0_1_acc5[_e] = _c0_[1][_e] * ((_e < 2) ? (_ep0_1_v10_0) : (_ep0_1_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 0) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64) * 32 + (a6_b * 16 + a6_u * 16 + 8) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep0_1_acc5[0], _ep0_1_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 0) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64) * 32 + (a6_b * 16 + a6_u * 16 + 8) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep0_1_acc5[2], _ep0_1_acc5[3]);
          }
        }
        float _ep1_0_v10_0 = 1.0f / acc3__r1_0;
        float _ep1_0_v10_1 = 1.0f / acc3__r1_1;
        float _ep1_0_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep1_0_acc5[_e] = _c1_[0][_e] * ((_e < 2) ? (_ep1_0_v10_0) : (_ep1_0_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 16) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 16) * 32 + (a6_b * 16 + a6_u * 16) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep1_0_acc5[0], _ep1_0_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 16) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 16) * 32 + (a6_b * 16 + a6_u * 16) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep1_0_acc5[2], _ep1_0_acc5[3]);
          }
        }
        float _ep1_1_v10_0 = 1.0f / acc3__r1_0;
        float _ep1_1_v10_1 = 1.0f / acc3__r1_1;
        float _ep1_1_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep1_1_acc5[_e] = _c1_[1][_e] * ((_e < 2) ? (_ep1_1_v10_0) : (_ep1_1_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 16) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 16) * 32 + (a6_b * 16 + a6_u * 16 + 8) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep1_1_acc5[0], _ep1_1_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 16) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 16) * 32 + (a6_b * 16 + a6_u * 16 + 8) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep1_1_acc5[2], _ep1_1_acc5[3]);
          }
        }
        float _ep2_0_v10_0 = 1.0f / acc3__r2_0;
        float _ep2_0_v10_1 = 1.0f / acc3__r2_1;
        float _ep2_0_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep2_0_acc5[_e] = _c2_[0][_e] * ((_e < 2) ? (_ep2_0_v10_0) : (_ep2_0_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 32) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 32) * 32 + (a6_b * 16 + a6_u * 16) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep2_0_acc5[0], _ep2_0_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 32) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 32) * 32 + (a6_b * 16 + a6_u * 16) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep2_0_acc5[2], _ep2_0_acc5[3]);
          }
        }
        float _ep2_1_v10_0 = 1.0f / acc3__r2_0;
        float _ep2_1_v10_1 = 1.0f / acc3__r2_1;
        float _ep2_1_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep2_1_acc5[_e] = _c2_[1][_e] * ((_e < 2) ? (_ep2_1_v10_0) : (_ep2_1_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 32) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 32) * 32 + (a6_b * 16 + a6_u * 16 + 8) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep2_1_acc5[0], _ep2_1_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 32) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 32) * 32 + (a6_b * 16 + a6_u * 16 + 8) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep2_1_acc5[2], _ep2_1_acc5[3]);
          }
        }
        float _ep3_0_v10_0 = 1.0f / acc3__r3_0;
        float _ep3_0_v10_1 = 1.0f / acc3__r3_1;
        float _ep3_0_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep3_0_acc5[_e] = _c3_[0][_e] * ((_e < 2) ? (_ep3_0_v10_0) : (_ep3_0_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 48) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 48) * 32 + (a6_b * 16 + a6_u * 16) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep3_0_acc5[0], _ep3_0_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 48) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 48) * 32 + (a6_b * 16 + a6_u * 16) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep3_0_acc5[2], _ep3_0_acc5[3]);
          }
        }
        float _ep3_1_v10_0 = 1.0f / acc3__r3_0;
        float _ep3_1_v10_1 = 1.0f / acc3__r3_1;
        float _ep3_1_acc5[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep3_1_acc5[_e] = _c3_[1][_e] * ((_e < 2) ? (_ep3_1_v10_0) : (_ep3_1_v10_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a1_b * 256 + a1_u * 64 + 48) + _g < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 48) * 32 + (a6_b * 16 + a6_u * 16 + 8) + _g * 32 + _t * 2]) = __floats2half2_rn(_ep3_1_acc5[0], _ep3_1_acc5[1]);
          }
          if ((a1_b * 256 + a1_u * 64 + 48) + _g + 8 < (64)) {
            *reinterpret_cast<__half2*>(&scaled_dot_product_attention[a0 * 2048 + (a1_b * 256 + a1_u * 64 + 48) * 32 + (a6_b * 16 + a6_u * 16 + 8) + (_g + 8) * 32 + _t * 2]) = __floats2half2_rn(_ep3_1_acc5[2], _ep3_1_acc5[3]);
          }
        }
}
