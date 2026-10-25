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

extern "C" __global__
__launch_bounds__(1024) void k_matmul_softmax_reduce_c1ee07(const __half* s, const __half* v, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 2048;
        int a3_b = (_gid / 1024) % 2;
        int a0_u = (_gid / 64) % 16;
        int a3_u = (_gid / 32) % 2;
        int _lane = (_gid) % 32;
        float _c0_[2][4] = {};
        float _c1_[2][4] = {};
        float _c2_[2][4] = {};
        float _c3_[2][4] = {};
        float acc0__r0_0 = -1e+30f;
        float acc0__r0_1 = -1e+30f;
        float acc0__r1_0 = -1e+30f;
        float acc0__r1_1 = -1e+30f;
        float acc0__r2_0 = -1e+30f;
        float acc0__r2_1 = -1e+30f;
        float acc0__r3_0 = -1e+30f;
        float acc0__r3_1 = -1e+30f;
        float acc1__r0_0 = 0.0f;
        float acc1__r0_1 = 0.0f;
        float acc1__r1_0 = 0.0f;
        float acc1__r1_1 = 0.0f;
        float acc1__r2_0 = 0.0f;
        float acc1__r2_1 = 0.0f;
        float acc1__r3_0 = 0.0f;
        float acc1__r3_1 = 0.0f;
        for (int a1__ck = 0; a1__ck < 70; a1__ck += 128) {
            float _s0_[16][4] = {};
            #pragma unroll
            for (int _r1 = 0; _r1 < 16; _r1 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s0_[_r1][0] = _s0_[_r1][0] + __half2float(s[(a0_b * 1024 + a0_u * 64 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s0_[_r1][1] = _s0_[_r1][1] + __half2float(s[(a0_b * 1024 + a0_u * 64 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
                _s0_[_r1][2] = _s0_[_r1][2] + __half2float(s[(a0_b * 1024 + a0_u * 64 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s0_[_r1][3] = _s0_[_r1][3] + __half2float(s[(a0_b * 1024 + a0_u * 64 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
            }
            float _s1_[16][4] = {};
            #pragma unroll
            for (int _r1 = 0; _r1 < 16; _r1 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s1_[_r1][0] = _s1_[_r1][0] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 16 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s1_[_r1][1] = _s1_[_r1][1] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 16 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
                _s1_[_r1][2] = _s1_[_r1][2] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 16 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s1_[_r1][3] = _s1_[_r1][3] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 16 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
            }
            float _s2_[16][4] = {};
            #pragma unroll
            for (int _r1 = 0; _r1 < 16; _r1 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s2_[_r1][0] = _s2_[_r1][0] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 32 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s2_[_r1][1] = _s2_[_r1][1] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 32 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
                _s2_[_r1][2] = _s2_[_r1][2] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 32 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s2_[_r1][3] = _s2_[_r1][3] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 32 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
            }
            float _s3_[16][4] = {};
            #pragma unroll
            for (int _r1 = 0; _r1 < 16; _r1 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s3_[_r1][0] = _s3_[_r1][0] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 48 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s3_[_r1][1] = _s3_[_r1][1] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 48 + _g) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
                _s3_[_r1][2] = _s3_[_r1][2] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 48 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + _t * 2) % 70]);
                _s3_[_r1][3] = _s3_[_r1][3] + __half2float(s[(a0_b * 1024 + a0_u * 64 + 48 + (_g + 8)) % 90 * 70 + (a1__ck + _r1 * 8 + (_t * 2 + 1)) % 70]);
            }
            #pragma unroll
            for (int _r0 = 0; _r0 < 16; _r0 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s0_[_r0][0] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s0_[_r0][0]));
                _s0_[_r0][1] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s0_[_r0][1]));
                _s0_[_r0][2] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s0_[_r0][2]));
                _s0_[_r0][3] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s0_[_r0][3]));
            }
            float _g__r0_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s0_[0][0], _s0_[0][1]), _s0_[1][0]), _s0_[1][1]), _s0_[2][0]), _s0_[2][1]), _s0_[3][0]), _s0_[3][1]), _s0_[4][0]), _s0_[4][1]), _s0_[5][0]), _s0_[5][1]), _s0_[6][0]), _s0_[6][1]), _s0_[7][0]), _s0_[7][1]), _s0_[8][0]), _s0_[8][1]), _s0_[9][0]), _s0_[9][1]), _s0_[10][0]), _s0_[10][1]), _s0_[11][0]), _s0_[11][1]), _s0_[12][0]), _s0_[12][1]), _s0_[13][0]), _s0_[13][1]), _s0_[14][0]), _s0_[14][1]), _s0_[15][0]), _s0_[15][1]);
            float _g__r0_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s0_[0][2], _s0_[0][3]), _s0_[1][2]), _s0_[1][3]), _s0_[2][2]), _s0_[2][3]), _s0_[3][2]), _s0_[3][3]), _s0_[4][2]), _s0_[4][3]), _s0_[5][2]), _s0_[5][3]), _s0_[6][2]), _s0_[6][3]), _s0_[7][2]), _s0_[7][3]), _s0_[8][2]), _s0_[8][3]), _s0_[9][2]), _s0_[9][3]), _s0_[10][2]), _s0_[10][3]), _s0_[11][2]), _s0_[11][3]), _s0_[12][2]), _s0_[12][3]), _s0_[13][2]), _s0_[13][3]), _s0_[14][2]), _s0_[14][3]), _s0_[15][2]), _s0_[15][3]);
            _g__r0_0 = fmaxf(_g__r0_0, __shfl_xor_sync(0xffffffff, _g__r0_0, 2));
            _g__r0_1 = fmaxf(_g__r0_1, __shfl_xor_sync(0xffffffff, _g__r0_1, 2));
            _g__r0_0 = fmaxf(_g__r0_0, __shfl_xor_sync(0xffffffff, _g__r0_0, 1));
            _g__r0_1 = fmaxf(_g__r0_1, __shfl_xor_sync(0xffffffff, _g__r0_1, 1));
            float _mg0_gn_0 = fmaxf(_g__r0_0, acc0__r0_0);
            float _mg0_gn_1 = fmaxf(_g__r0_1, acc0__r0_1);
            float _mg0_alpha_0 = expf((acc0__r0_0 - _mg0_gn_0));
            float _mg0_alpha_1 = expf((acc0__r0_1 - _mg0_gn_1));
            #pragma unroll
            for (int _r0 = 0; _r0 < 16; _r0 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s1_[_r0][0] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s1_[_r0][0]));
                _s1_[_r0][1] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s1_[_r0][1]));
                _s1_[_r0][2] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s1_[_r0][2]));
                _s1_[_r0][3] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s1_[_r0][3]));
            }
            float _g__r1_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s1_[0][0], _s1_[0][1]), _s1_[1][0]), _s1_[1][1]), _s1_[2][0]), _s1_[2][1]), _s1_[3][0]), _s1_[3][1]), _s1_[4][0]), _s1_[4][1]), _s1_[5][0]), _s1_[5][1]), _s1_[6][0]), _s1_[6][1]), _s1_[7][0]), _s1_[7][1]), _s1_[8][0]), _s1_[8][1]), _s1_[9][0]), _s1_[9][1]), _s1_[10][0]), _s1_[10][1]), _s1_[11][0]), _s1_[11][1]), _s1_[12][0]), _s1_[12][1]), _s1_[13][0]), _s1_[13][1]), _s1_[14][0]), _s1_[14][1]), _s1_[15][0]), _s1_[15][1]);
            float _g__r1_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s1_[0][2], _s1_[0][3]), _s1_[1][2]), _s1_[1][3]), _s1_[2][2]), _s1_[2][3]), _s1_[3][2]), _s1_[3][3]), _s1_[4][2]), _s1_[4][3]), _s1_[5][2]), _s1_[5][3]), _s1_[6][2]), _s1_[6][3]), _s1_[7][2]), _s1_[7][3]), _s1_[8][2]), _s1_[8][3]), _s1_[9][2]), _s1_[9][3]), _s1_[10][2]), _s1_[10][3]), _s1_[11][2]), _s1_[11][3]), _s1_[12][2]), _s1_[12][3]), _s1_[13][2]), _s1_[13][3]), _s1_[14][2]), _s1_[14][3]), _s1_[15][2]), _s1_[15][3]);
            _g__r1_0 = fmaxf(_g__r1_0, __shfl_xor_sync(0xffffffff, _g__r1_0, 2));
            _g__r1_1 = fmaxf(_g__r1_1, __shfl_xor_sync(0xffffffff, _g__r1_1, 2));
            _g__r1_0 = fmaxf(_g__r1_0, __shfl_xor_sync(0xffffffff, _g__r1_0, 1));
            _g__r1_1 = fmaxf(_g__r1_1, __shfl_xor_sync(0xffffffff, _g__r1_1, 1));
            float _mg1_gn_0 = fmaxf(_g__r1_0, acc0__r1_0);
            float _mg1_gn_1 = fmaxf(_g__r1_1, acc0__r1_1);
            float _mg1_alpha_0 = expf((acc0__r1_0 - _mg1_gn_0));
            float _mg1_alpha_1 = expf((acc0__r1_1 - _mg1_gn_1));
            #pragma unroll
            for (int _r0 = 0; _r0 < 16; _r0 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s2_[_r0][0] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s2_[_r0][0]));
                _s2_[_r0][1] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s2_[_r0][1]));
                _s2_[_r0][2] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s2_[_r0][2]));
                _s2_[_r0][3] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s2_[_r0][3]));
            }
            float _g__r2_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s2_[0][0], _s2_[0][1]), _s2_[1][0]), _s2_[1][1]), _s2_[2][0]), _s2_[2][1]), _s2_[3][0]), _s2_[3][1]), _s2_[4][0]), _s2_[4][1]), _s2_[5][0]), _s2_[5][1]), _s2_[6][0]), _s2_[6][1]), _s2_[7][0]), _s2_[7][1]), _s2_[8][0]), _s2_[8][1]), _s2_[9][0]), _s2_[9][1]), _s2_[10][0]), _s2_[10][1]), _s2_[11][0]), _s2_[11][1]), _s2_[12][0]), _s2_[12][1]), _s2_[13][0]), _s2_[13][1]), _s2_[14][0]), _s2_[14][1]), _s2_[15][0]), _s2_[15][1]);
            float _g__r2_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s2_[0][2], _s2_[0][3]), _s2_[1][2]), _s2_[1][3]), _s2_[2][2]), _s2_[2][3]), _s2_[3][2]), _s2_[3][3]), _s2_[4][2]), _s2_[4][3]), _s2_[5][2]), _s2_[5][3]), _s2_[6][2]), _s2_[6][3]), _s2_[7][2]), _s2_[7][3]), _s2_[8][2]), _s2_[8][3]), _s2_[9][2]), _s2_[9][3]), _s2_[10][2]), _s2_[10][3]), _s2_[11][2]), _s2_[11][3]), _s2_[12][2]), _s2_[12][3]), _s2_[13][2]), _s2_[13][3]), _s2_[14][2]), _s2_[14][3]), _s2_[15][2]), _s2_[15][3]);
            _g__r2_0 = fmaxf(_g__r2_0, __shfl_xor_sync(0xffffffff, _g__r2_0, 2));
            _g__r2_1 = fmaxf(_g__r2_1, __shfl_xor_sync(0xffffffff, _g__r2_1, 2));
            _g__r2_0 = fmaxf(_g__r2_0, __shfl_xor_sync(0xffffffff, _g__r2_0, 1));
            _g__r2_1 = fmaxf(_g__r2_1, __shfl_xor_sync(0xffffffff, _g__r2_1, 1));
            float _mg2_gn_0 = fmaxf(_g__r2_0, acc0__r2_0);
            float _mg2_gn_1 = fmaxf(_g__r2_1, acc0__r2_1);
            float _mg2_alpha_0 = expf((acc0__r2_0 - _mg2_gn_0));
            float _mg2_alpha_1 = expf((acc0__r2_1 - _mg2_gn_1));
            #pragma unroll
            for (int _r0 = 0; _r0 < 16; _r0 += 1) {
                const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;
                _s3_[_r0][0] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s3_[_r0][0]));
                _s3_[_r0][1] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s3_[_r0][1]));
                _s3_[_r0][2] = ((a1__ck + _r0 * 8 + (_t * 2 + 0) >= 70) ? (-1e+30f) : (_s3_[_r0][2]));
                _s3_[_r0][3] = ((a1__ck + _r0 * 8 + (_t * 2 + 1) >= 70) ? (-1e+30f) : (_s3_[_r0][3]));
            }
            float _g__r3_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s3_[0][0], _s3_[0][1]), _s3_[1][0]), _s3_[1][1]), _s3_[2][0]), _s3_[2][1]), _s3_[3][0]), _s3_[3][1]), _s3_[4][0]), _s3_[4][1]), _s3_[5][0]), _s3_[5][1]), _s3_[6][0]), _s3_[6][1]), _s3_[7][0]), _s3_[7][1]), _s3_[8][0]), _s3_[8][1]), _s3_[9][0]), _s3_[9][1]), _s3_[10][0]), _s3_[10][1]), _s3_[11][0]), _s3_[11][1]), _s3_[12][0]), _s3_[12][1]), _s3_[13][0]), _s3_[13][1]), _s3_[14][0]), _s3_[14][1]), _s3_[15][0]), _s3_[15][1]);
            float _g__r3_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(_s3_[0][2], _s3_[0][3]), _s3_[1][2]), _s3_[1][3]), _s3_[2][2]), _s3_[2][3]), _s3_[3][2]), _s3_[3][3]), _s3_[4][2]), _s3_[4][3]), _s3_[5][2]), _s3_[5][3]), _s3_[6][2]), _s3_[6][3]), _s3_[7][2]), _s3_[7][3]), _s3_[8][2]), _s3_[8][3]), _s3_[9][2]), _s3_[9][3]), _s3_[10][2]), _s3_[10][3]), _s3_[11][2]), _s3_[11][3]), _s3_[12][2]), _s3_[12][3]), _s3_[13][2]), _s3_[13][3]), _s3_[14][2]), _s3_[14][3]), _s3_[15][2]), _s3_[15][3]);
            _g__r3_0 = fmaxf(_g__r3_0, __shfl_xor_sync(0xffffffff, _g__r3_0, 2));
            _g__r3_1 = fmaxf(_g__r3_1, __shfl_xor_sync(0xffffffff, _g__r3_1, 2));
            _g__r3_0 = fmaxf(_g__r3_0, __shfl_xor_sync(0xffffffff, _g__r3_0, 1));
            _g__r3_1 = fmaxf(_g__r3_1, __shfl_xor_sync(0xffffffff, _g__r3_1, 1));
            float _mg3_gn_0 = fmaxf(_g__r3_0, acc0__r3_0);
            float _mg3_gn_1 = fmaxf(_g__r3_1, acc0__r3_1);
            float _mg3_alpha_0 = expf((acc0__r3_0 - _mg3_gn_0));
            float _mg3_alpha_1 = expf((acc0__r3_1 - _mg3_gn_1));
            float _ch0_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_0_d[_e] = _s0_[0][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_0_w[_e] = expf(_ch0_0_d[_e]);
            }
            float _ch0_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_1_d[_e] = _s0_[1][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_1_w[_e] = expf(_ch0_1_d[_e]);
            }
            float _ch0_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_2_d[_e] = _s0_[2][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_2_w[_e] = expf(_ch0_2_d[_e]);
            }
            float _ch0_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_3_d[_e] = _s0_[3][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_3_w[_e] = expf(_ch0_3_d[_e]);
            }
            float _ch0_4_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_4_d[_e] = _s0_[4][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_4_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_4_w[_e] = expf(_ch0_4_d[_e]);
            }
            float _ch0_5_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_5_d[_e] = _s0_[5][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_5_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_5_w[_e] = expf(_ch0_5_d[_e]);
            }
            float _ch0_6_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_6_d[_e] = _s0_[6][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_6_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_6_w[_e] = expf(_ch0_6_d[_e]);
            }
            float _ch0_7_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_7_d[_e] = _s0_[7][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_7_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_7_w[_e] = expf(_ch0_7_d[_e]);
            }
            float _ch0_8_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_8_d[_e] = _s0_[8][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_8_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_8_w[_e] = expf(_ch0_8_d[_e]);
            }
            float _ch0_9_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_9_d[_e] = _s0_[9][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_9_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_9_w[_e] = expf(_ch0_9_d[_e]);
            }
            float _ch0_10_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_10_d[_e] = _s0_[10][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_10_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_10_w[_e] = expf(_ch0_10_d[_e]);
            }
            float _ch0_11_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_11_d[_e] = _s0_[11][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_11_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_11_w[_e] = expf(_ch0_11_d[_e]);
            }
            float _ch0_12_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_12_d[_e] = _s0_[12][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_12_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_12_w[_e] = expf(_ch0_12_d[_e]);
            }
            float _ch0_13_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_13_d[_e] = _s0_[13][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_13_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_13_w[_e] = expf(_ch0_13_d[_e]);
            }
            float _ch0_14_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_14_d[_e] = _s0_[14][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_14_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_14_w[_e] = expf(_ch0_14_d[_e]);
            }
            float _ch0_15_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_15_d[_e] = _s0_[15][_e] - ((_e < 2) ? (_mg0_gn_0) : (_mg0_gn_1));
            }
            float _ch0_15_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch0_15_w[_e] = expf(_ch0_15_d[_e]);
            }
            float acc1__p0_0 = _ch0_0_w[0] + _ch0_0_w[1] + _ch0_1_w[0] + _ch0_1_w[1] + _ch0_2_w[0] + _ch0_2_w[1] + _ch0_3_w[0] + _ch0_3_w[1] + _ch0_4_w[0] + _ch0_4_w[1] + _ch0_5_w[0] + _ch0_5_w[1] + _ch0_6_w[0] + _ch0_6_w[1] + _ch0_7_w[0] + _ch0_7_w[1] + _ch0_8_w[0] + _ch0_8_w[1] + _ch0_9_w[0] + _ch0_9_w[1] + _ch0_10_w[0] + _ch0_10_w[1] + _ch0_11_w[0] + _ch0_11_w[1] + _ch0_12_w[0] + _ch0_12_w[1] + _ch0_13_w[0] + _ch0_13_w[1] + _ch0_14_w[0] + _ch0_14_w[1] + _ch0_15_w[0] + _ch0_15_w[1];
            float acc1__p0_1 = _ch0_0_w[2] + _ch0_0_w[3] + _ch0_1_w[2] + _ch0_1_w[3] + _ch0_2_w[2] + _ch0_2_w[3] + _ch0_3_w[2] + _ch0_3_w[3] + _ch0_4_w[2] + _ch0_4_w[3] + _ch0_5_w[2] + _ch0_5_w[3] + _ch0_6_w[2] + _ch0_6_w[3] + _ch0_7_w[2] + _ch0_7_w[3] + _ch0_8_w[2] + _ch0_8_w[3] + _ch0_9_w[2] + _ch0_9_w[3] + _ch0_10_w[2] + _ch0_10_w[3] + _ch0_11_w[2] + _ch0_11_w[3] + _ch0_12_w[2] + _ch0_12_w[3] + _ch0_13_w[2] + _ch0_13_w[3] + _ch0_14_w[2] + _ch0_14_w[3] + _ch0_15_w[2] + _ch0_15_w[3];
            acc1__p0_0 = acc1__p0_0 + __shfl_xor_sync(0xffffffff, acc1__p0_0, 2);
            acc1__p0_1 = acc1__p0_1 + __shfl_xor_sync(0xffffffff, acc1__p0_1, 2);
            acc1__p0_0 = acc1__p0_0 + __shfl_xor_sync(0xffffffff, acc1__p0_0, 1);
            acc1__p0_1 = acc1__p0_1 + __shfl_xor_sync(0xffffffff, acc1__p0_1, 1);
            float _mg0_acc1___scaled_0 = _mg0_alpha_0 * acc1__r0_0;
            float _mg0_acc1___scaled_1 = _mg0_alpha_1 * acc1__r0_1;
            acc1__r0_0 = _mg0_acc1___scaled_0 + acc1__p0_0;
            acc1__r0_1 = _mg0_acc1___scaled_1 + acc1__p0_1;
            float _ch1_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_0_d[_e] = _s1_[0][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_0_w[_e] = expf(_ch1_0_d[_e]);
            }
            float _ch1_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_1_d[_e] = _s1_[1][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_1_w[_e] = expf(_ch1_1_d[_e]);
            }
            float _ch1_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_2_d[_e] = _s1_[2][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_2_w[_e] = expf(_ch1_2_d[_e]);
            }
            float _ch1_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_3_d[_e] = _s1_[3][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_3_w[_e] = expf(_ch1_3_d[_e]);
            }
            float _ch1_4_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_4_d[_e] = _s1_[4][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_4_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_4_w[_e] = expf(_ch1_4_d[_e]);
            }
            float _ch1_5_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_5_d[_e] = _s1_[5][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_5_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_5_w[_e] = expf(_ch1_5_d[_e]);
            }
            float _ch1_6_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_6_d[_e] = _s1_[6][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_6_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_6_w[_e] = expf(_ch1_6_d[_e]);
            }
            float _ch1_7_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_7_d[_e] = _s1_[7][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_7_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_7_w[_e] = expf(_ch1_7_d[_e]);
            }
            float _ch1_8_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_8_d[_e] = _s1_[8][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_8_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_8_w[_e] = expf(_ch1_8_d[_e]);
            }
            float _ch1_9_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_9_d[_e] = _s1_[9][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_9_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_9_w[_e] = expf(_ch1_9_d[_e]);
            }
            float _ch1_10_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_10_d[_e] = _s1_[10][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_10_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_10_w[_e] = expf(_ch1_10_d[_e]);
            }
            float _ch1_11_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_11_d[_e] = _s1_[11][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_11_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_11_w[_e] = expf(_ch1_11_d[_e]);
            }
            float _ch1_12_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_12_d[_e] = _s1_[12][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_12_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_12_w[_e] = expf(_ch1_12_d[_e]);
            }
            float _ch1_13_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_13_d[_e] = _s1_[13][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_13_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_13_w[_e] = expf(_ch1_13_d[_e]);
            }
            float _ch1_14_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_14_d[_e] = _s1_[14][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_14_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_14_w[_e] = expf(_ch1_14_d[_e]);
            }
            float _ch1_15_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_15_d[_e] = _s1_[15][_e] - ((_e < 2) ? (_mg1_gn_0) : (_mg1_gn_1));
            }
            float _ch1_15_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch1_15_w[_e] = expf(_ch1_15_d[_e]);
            }
            float acc1__p1_0 = _ch1_0_w[0] + _ch1_0_w[1] + _ch1_1_w[0] + _ch1_1_w[1] + _ch1_2_w[0] + _ch1_2_w[1] + _ch1_3_w[0] + _ch1_3_w[1] + _ch1_4_w[0] + _ch1_4_w[1] + _ch1_5_w[0] + _ch1_5_w[1] + _ch1_6_w[0] + _ch1_6_w[1] + _ch1_7_w[0] + _ch1_7_w[1] + _ch1_8_w[0] + _ch1_8_w[1] + _ch1_9_w[0] + _ch1_9_w[1] + _ch1_10_w[0] + _ch1_10_w[1] + _ch1_11_w[0] + _ch1_11_w[1] + _ch1_12_w[0] + _ch1_12_w[1] + _ch1_13_w[0] + _ch1_13_w[1] + _ch1_14_w[0] + _ch1_14_w[1] + _ch1_15_w[0] + _ch1_15_w[1];
            float acc1__p1_1 = _ch1_0_w[2] + _ch1_0_w[3] + _ch1_1_w[2] + _ch1_1_w[3] + _ch1_2_w[2] + _ch1_2_w[3] + _ch1_3_w[2] + _ch1_3_w[3] + _ch1_4_w[2] + _ch1_4_w[3] + _ch1_5_w[2] + _ch1_5_w[3] + _ch1_6_w[2] + _ch1_6_w[3] + _ch1_7_w[2] + _ch1_7_w[3] + _ch1_8_w[2] + _ch1_8_w[3] + _ch1_9_w[2] + _ch1_9_w[3] + _ch1_10_w[2] + _ch1_10_w[3] + _ch1_11_w[2] + _ch1_11_w[3] + _ch1_12_w[2] + _ch1_12_w[3] + _ch1_13_w[2] + _ch1_13_w[3] + _ch1_14_w[2] + _ch1_14_w[3] + _ch1_15_w[2] + _ch1_15_w[3];
            acc1__p1_0 = acc1__p1_0 + __shfl_xor_sync(0xffffffff, acc1__p1_0, 2);
            acc1__p1_1 = acc1__p1_1 + __shfl_xor_sync(0xffffffff, acc1__p1_1, 2);
            acc1__p1_0 = acc1__p1_0 + __shfl_xor_sync(0xffffffff, acc1__p1_0, 1);
            acc1__p1_1 = acc1__p1_1 + __shfl_xor_sync(0xffffffff, acc1__p1_1, 1);
            float _mg1_acc1___scaled_0 = _mg1_alpha_0 * acc1__r1_0;
            float _mg1_acc1___scaled_1 = _mg1_alpha_1 * acc1__r1_1;
            acc1__r1_0 = _mg1_acc1___scaled_0 + acc1__p1_0;
            acc1__r1_1 = _mg1_acc1___scaled_1 + acc1__p1_1;
            float _ch2_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_0_d[_e] = _s2_[0][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_0_w[_e] = expf(_ch2_0_d[_e]);
            }
            float _ch2_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_1_d[_e] = _s2_[1][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_1_w[_e] = expf(_ch2_1_d[_e]);
            }
            float _ch2_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_2_d[_e] = _s2_[2][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_2_w[_e] = expf(_ch2_2_d[_e]);
            }
            float _ch2_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_3_d[_e] = _s2_[3][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_3_w[_e] = expf(_ch2_3_d[_e]);
            }
            float _ch2_4_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_4_d[_e] = _s2_[4][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_4_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_4_w[_e] = expf(_ch2_4_d[_e]);
            }
            float _ch2_5_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_5_d[_e] = _s2_[5][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_5_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_5_w[_e] = expf(_ch2_5_d[_e]);
            }
            float _ch2_6_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_6_d[_e] = _s2_[6][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_6_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_6_w[_e] = expf(_ch2_6_d[_e]);
            }
            float _ch2_7_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_7_d[_e] = _s2_[7][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_7_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_7_w[_e] = expf(_ch2_7_d[_e]);
            }
            float _ch2_8_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_8_d[_e] = _s2_[8][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_8_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_8_w[_e] = expf(_ch2_8_d[_e]);
            }
            float _ch2_9_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_9_d[_e] = _s2_[9][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_9_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_9_w[_e] = expf(_ch2_9_d[_e]);
            }
            float _ch2_10_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_10_d[_e] = _s2_[10][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_10_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_10_w[_e] = expf(_ch2_10_d[_e]);
            }
            float _ch2_11_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_11_d[_e] = _s2_[11][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_11_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_11_w[_e] = expf(_ch2_11_d[_e]);
            }
            float _ch2_12_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_12_d[_e] = _s2_[12][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_12_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_12_w[_e] = expf(_ch2_12_d[_e]);
            }
            float _ch2_13_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_13_d[_e] = _s2_[13][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_13_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_13_w[_e] = expf(_ch2_13_d[_e]);
            }
            float _ch2_14_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_14_d[_e] = _s2_[14][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_14_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_14_w[_e] = expf(_ch2_14_d[_e]);
            }
            float _ch2_15_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_15_d[_e] = _s2_[15][_e] - ((_e < 2) ? (_mg2_gn_0) : (_mg2_gn_1));
            }
            float _ch2_15_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch2_15_w[_e] = expf(_ch2_15_d[_e]);
            }
            float acc1__p2_0 = _ch2_0_w[0] + _ch2_0_w[1] + _ch2_1_w[0] + _ch2_1_w[1] + _ch2_2_w[0] + _ch2_2_w[1] + _ch2_3_w[0] + _ch2_3_w[1] + _ch2_4_w[0] + _ch2_4_w[1] + _ch2_5_w[0] + _ch2_5_w[1] + _ch2_6_w[0] + _ch2_6_w[1] + _ch2_7_w[0] + _ch2_7_w[1] + _ch2_8_w[0] + _ch2_8_w[1] + _ch2_9_w[0] + _ch2_9_w[1] + _ch2_10_w[0] + _ch2_10_w[1] + _ch2_11_w[0] + _ch2_11_w[1] + _ch2_12_w[0] + _ch2_12_w[1] + _ch2_13_w[0] + _ch2_13_w[1] + _ch2_14_w[0] + _ch2_14_w[1] + _ch2_15_w[0] + _ch2_15_w[1];
            float acc1__p2_1 = _ch2_0_w[2] + _ch2_0_w[3] + _ch2_1_w[2] + _ch2_1_w[3] + _ch2_2_w[2] + _ch2_2_w[3] + _ch2_3_w[2] + _ch2_3_w[3] + _ch2_4_w[2] + _ch2_4_w[3] + _ch2_5_w[2] + _ch2_5_w[3] + _ch2_6_w[2] + _ch2_6_w[3] + _ch2_7_w[2] + _ch2_7_w[3] + _ch2_8_w[2] + _ch2_8_w[3] + _ch2_9_w[2] + _ch2_9_w[3] + _ch2_10_w[2] + _ch2_10_w[3] + _ch2_11_w[2] + _ch2_11_w[3] + _ch2_12_w[2] + _ch2_12_w[3] + _ch2_13_w[2] + _ch2_13_w[3] + _ch2_14_w[2] + _ch2_14_w[3] + _ch2_15_w[2] + _ch2_15_w[3];
            acc1__p2_0 = acc1__p2_0 + __shfl_xor_sync(0xffffffff, acc1__p2_0, 2);
            acc1__p2_1 = acc1__p2_1 + __shfl_xor_sync(0xffffffff, acc1__p2_1, 2);
            acc1__p2_0 = acc1__p2_0 + __shfl_xor_sync(0xffffffff, acc1__p2_0, 1);
            acc1__p2_1 = acc1__p2_1 + __shfl_xor_sync(0xffffffff, acc1__p2_1, 1);
            float _mg2_acc1___scaled_0 = _mg2_alpha_0 * acc1__r2_0;
            float _mg2_acc1___scaled_1 = _mg2_alpha_1 * acc1__r2_1;
            acc1__r2_0 = _mg2_acc1___scaled_0 + acc1__p2_0;
            acc1__r2_1 = _mg2_acc1___scaled_1 + acc1__p2_1;
            float _ch3_0_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_0_d[_e] = _s3_[0][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_0_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_0_w[_e] = expf(_ch3_0_d[_e]);
            }
            float _ch3_1_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_1_d[_e] = _s3_[1][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_1_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_1_w[_e] = expf(_ch3_1_d[_e]);
            }
            float _ch3_2_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_2_d[_e] = _s3_[2][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_2_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_2_w[_e] = expf(_ch3_2_d[_e]);
            }
            float _ch3_3_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_3_d[_e] = _s3_[3][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_3_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_3_w[_e] = expf(_ch3_3_d[_e]);
            }
            float _ch3_4_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_4_d[_e] = _s3_[4][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_4_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_4_w[_e] = expf(_ch3_4_d[_e]);
            }
            float _ch3_5_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_5_d[_e] = _s3_[5][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_5_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_5_w[_e] = expf(_ch3_5_d[_e]);
            }
            float _ch3_6_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_6_d[_e] = _s3_[6][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_6_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_6_w[_e] = expf(_ch3_6_d[_e]);
            }
            float _ch3_7_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_7_d[_e] = _s3_[7][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_7_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_7_w[_e] = expf(_ch3_7_d[_e]);
            }
            float _ch3_8_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_8_d[_e] = _s3_[8][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_8_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_8_w[_e] = expf(_ch3_8_d[_e]);
            }
            float _ch3_9_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_9_d[_e] = _s3_[9][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_9_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_9_w[_e] = expf(_ch3_9_d[_e]);
            }
            float _ch3_10_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_10_d[_e] = _s3_[10][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_10_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_10_w[_e] = expf(_ch3_10_d[_e]);
            }
            float _ch3_11_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_11_d[_e] = _s3_[11][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_11_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_11_w[_e] = expf(_ch3_11_d[_e]);
            }
            float _ch3_12_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_12_d[_e] = _s3_[12][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_12_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_12_w[_e] = expf(_ch3_12_d[_e]);
            }
            float _ch3_13_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_13_d[_e] = _s3_[13][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_13_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_13_w[_e] = expf(_ch3_13_d[_e]);
            }
            float _ch3_14_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_14_d[_e] = _s3_[14][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_14_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_14_w[_e] = expf(_ch3_14_d[_e]);
            }
            float _ch3_15_d[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_15_d[_e] = _s3_[15][_e] - ((_e < 2) ? (_mg3_gn_0) : (_mg3_gn_1));
            }
            float _ch3_15_w[4];
            #pragma unroll
            for (int _e = 0; _e < 4; _e++) {
                _ch3_15_w[_e] = expf(_ch3_15_d[_e]);
            }
            float acc1__p3_0 = _ch3_0_w[0] + _ch3_0_w[1] + _ch3_1_w[0] + _ch3_1_w[1] + _ch3_2_w[0] + _ch3_2_w[1] + _ch3_3_w[0] + _ch3_3_w[1] + _ch3_4_w[0] + _ch3_4_w[1] + _ch3_5_w[0] + _ch3_5_w[1] + _ch3_6_w[0] + _ch3_6_w[1] + _ch3_7_w[0] + _ch3_7_w[1] + _ch3_8_w[0] + _ch3_8_w[1] + _ch3_9_w[0] + _ch3_9_w[1] + _ch3_10_w[0] + _ch3_10_w[1] + _ch3_11_w[0] + _ch3_11_w[1] + _ch3_12_w[0] + _ch3_12_w[1] + _ch3_13_w[0] + _ch3_13_w[1] + _ch3_14_w[0] + _ch3_14_w[1] + _ch3_15_w[0] + _ch3_15_w[1];
            float acc1__p3_1 = _ch3_0_w[2] + _ch3_0_w[3] + _ch3_1_w[2] + _ch3_1_w[3] + _ch3_2_w[2] + _ch3_2_w[3] + _ch3_3_w[2] + _ch3_3_w[3] + _ch3_4_w[2] + _ch3_4_w[3] + _ch3_5_w[2] + _ch3_5_w[3] + _ch3_6_w[2] + _ch3_6_w[3] + _ch3_7_w[2] + _ch3_7_w[3] + _ch3_8_w[2] + _ch3_8_w[3] + _ch3_9_w[2] + _ch3_9_w[3] + _ch3_10_w[2] + _ch3_10_w[3] + _ch3_11_w[2] + _ch3_11_w[3] + _ch3_12_w[2] + _ch3_12_w[3] + _ch3_13_w[2] + _ch3_13_w[3] + _ch3_14_w[2] + _ch3_14_w[3] + _ch3_15_w[2] + _ch3_15_w[3];
            acc1__p3_0 = acc1__p3_0 + __shfl_xor_sync(0xffffffff, acc1__p3_0, 2);
            acc1__p3_1 = acc1__p3_1 + __shfl_xor_sync(0xffffffff, acc1__p3_1, 2);
            acc1__p3_0 = acc1__p3_0 + __shfl_xor_sync(0xffffffff, acc1__p3_0, 1);
            acc1__p3_1 = acc1__p3_1 + __shfl_xor_sync(0xffffffff, acc1__p3_1, 1);
            float _mg3_acc1___scaled_0 = _mg3_alpha_0 * acc1__r3_0;
            float _mg3_acc1___scaled_1 = _mg3_alpha_1 * acc1__r3_1;
            acc1__r3_0 = _mg3_acc1___scaled_0 + acc1__p3_0;
            acc1__r3_1 = _mg3_acc1___scaled_1 + acc1__p3_1;
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
            unsigned _a0_2[4];
            unsigned _a0_3[4];
            unsigned _a0_4[4];
            unsigned _a0_5[4];
            unsigned _a0_6[4];
            unsigned _a0_7[4];
            unsigned _a1_0[4];
            unsigned _a1_1[4];
            unsigned _a1_2[4];
            unsigned _a1_3[4];
            unsigned _a1_4[4];
            unsigned _a1_5[4];
            unsigned _a1_6[4];
            unsigned _a1_7[4];
            unsigned _a2_0[4];
            unsigned _a2_1[4];
            unsigned _a2_2[4];
            unsigned _a2_3[4];
            unsigned _a2_4[4];
            unsigned _a2_5[4];
            unsigned _a2_6[4];
            unsigned _a2_7[4];
            unsigned _a3_0[4];
            unsigned _a3_1[4];
            unsigned _a3_2[4];
            unsigned _a3_3[4];
            unsigned _a3_4[4];
            unsigned _a3_5[4];
            unsigned _a3_6[4];
            unsigned _a3_7[4];
            unsigned _b0_0[2];
            unsigned _b0_1[2];
            unsigned _b0_2[2];
            unsigned _b0_3[2];
            unsigned _b0_4[2];
            unsigned _b0_5[2];
            unsigned _b0_6[2];
            unsigned _b0_7[2];
            unsigned _b1_0[2];
            unsigned _b1_1[2];
            unsigned _b1_2[2];
            unsigned _b1_3[2];
            unsigned _b1_4[2];
            unsigned _b1_5[2];
            unsigned _b1_6[2];
            unsigned _b1_7[2];
            emmy_c_to_a_f16(_a0_0, _ch0_0_w, _ch0_1_w);
            emmy_c_to_a_f16(_a1_0, _ch1_0_w, _ch1_1_w);
            emmy_c_to_a_f16(_a2_0, _ch2_0_w, _ch2_1_w);
            emmy_c_to_a_f16(_a3_0, _ch3_0_w, _ch3_1_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_0, &v[a1__ck * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 0));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_0, &v[a1__ck * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 0));
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
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_1, &v[(a1__ck + 16) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 16));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_1, &v[(a1__ck + 16) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 16));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_1, _b0_1, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_1, _b1_1, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_1, _b0_1, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_1, _b1_1, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_1, _b0_1, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_1, _b1_1, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_1, _b0_1, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_1, _b1_1, _c3_[1]);
            emmy_c_to_a_f16(_a0_2, _ch0_4_w, _ch0_5_w);
            emmy_c_to_a_f16(_a1_2, _ch1_4_w, _ch1_5_w);
            emmy_c_to_a_f16(_a2_2, _ch2_4_w, _ch2_5_w);
            emmy_c_to_a_f16(_a3_2, _ch3_4_w, _ch3_5_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_2, &v[(a1__ck + 32) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 32));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_2, &v[(a1__ck + 32) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 32));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_2, _b0_2, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_2, _b1_2, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_2, _b0_2, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_2, _b1_2, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_2, _b0_2, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_2, _b1_2, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_2, _b0_2, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_2, _b1_2, _c3_[1]);
            emmy_c_to_a_f16(_a0_3, _ch0_6_w, _ch0_7_w);
            emmy_c_to_a_f16(_a1_3, _ch1_6_w, _ch1_7_w);
            emmy_c_to_a_f16(_a2_3, _ch2_6_w, _ch2_7_w);
            emmy_c_to_a_f16(_a3_3, _ch3_6_w, _ch3_7_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_3, &v[(a1__ck + 48) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 48));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_3, &v[(a1__ck + 48) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 48));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_3, _b0_3, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_3, _b1_3, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_3, _b0_3, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_3, _b1_3, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_3, _b0_3, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_3, _b1_3, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_3, _b0_3, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_3, _b1_3, _c3_[1]);
            emmy_c_to_a_f16(_a0_4, _ch0_8_w, _ch0_9_w);
            emmy_c_to_a_f16(_a1_4, _ch1_8_w, _ch1_9_w);
            emmy_c_to_a_f16(_a2_4, _ch2_8_w, _ch2_9_w);
            emmy_c_to_a_f16(_a3_4, _ch3_8_w, _ch3_9_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_4, &v[(a1__ck + 64) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 64));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_4, &v[(a1__ck + 64) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 64));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_4, _b0_4, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_4, _b1_4, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_4, _b0_4, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_4, _b1_4, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_4, _b0_4, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_4, _b1_4, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_4, _b0_4, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_4, _b1_4, _c3_[1]);
            emmy_c_to_a_f16(_a0_5, _ch0_10_w, _ch0_11_w);
            emmy_c_to_a_f16(_a1_5, _ch1_10_w, _ch1_11_w);
            emmy_c_to_a_f16(_a2_5, _ch2_10_w, _ch2_11_w);
            emmy_c_to_a_f16(_a3_5, _ch3_10_w, _ch3_11_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_5, &v[(a1__ck + 80) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 80));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_5, &v[(a1__ck + 80) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 80));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_5, _b0_5, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_5, _b1_5, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_5, _b0_5, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_5, _b1_5, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_5, _b0_5, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_5, _b1_5, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_5, _b0_5, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_5, _b1_5, _c3_[1]);
            emmy_c_to_a_f16(_a0_6, _ch0_12_w, _ch0_13_w);
            emmy_c_to_a_f16(_a1_6, _ch1_12_w, _ch1_13_w);
            emmy_c_to_a_f16(_a2_6, _ch2_12_w, _ch2_13_w);
            emmy_c_to_a_f16(_a3_6, _ch3_12_w, _ch3_13_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_6, &v[(a1__ck + 96) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 96));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_6, &v[(a1__ck + 96) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 96));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_6, _b0_6, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_6, _b1_6, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_6, _b0_6, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_6, _b1_6, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_6, _b0_6, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_6, _b1_6, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_6, _b0_6, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_6, _b1_6, _c3_[1]);
            emmy_c_to_a_f16(_a0_7, _ch0_14_w, _ch0_15_w);
            emmy_c_to_a_f16(_a1_7, _ch1_14_w, _ch1_15_w);
            emmy_c_to_a_f16(_a2_7, _ch2_14_w, _ch2_15_w);
            emmy_c_to_a_f16(_a3_7, _ch3_14_w, _ch3_15_w);
            emmy_mma_load_b_gmem_nclamp_kzero(_b0_7, &v[(a1__ck + 112) * 40 + (a3_b * 32 + a3_u * 16)], 40, (40) - (a3_b * 32 + a3_u * 16 + 0), (70) - (a1__ck + 112));
            emmy_mma_load_b_gmem_nclamp_kzero(_b1_7, &v[(a1__ck + 112) * 40 + (a3_b * 32 + a3_u * 16 + 8)], 40, (40) - (a3_b * 32 + a3_u * 16 + 8), (70) - (a1__ck + 112));
            emmy_mma_m16n8k16_f16_f32(_c0_[0], _a0_7, _b0_7, _c0_[0]);
            emmy_mma_m16n8k16_f16_f32(_c0_[1], _a0_7, _b1_7, _c0_[1]);
            emmy_mma_m16n8k16_f16_f32(_c1_[0], _a1_7, _b0_7, _c1_[0]);
            emmy_mma_m16n8k16_f16_f32(_c1_[1], _a1_7, _b1_7, _c1_[1]);
            emmy_mma_m16n8k16_f16_f32(_c2_[0], _a2_7, _b0_7, _c2_[0]);
            emmy_mma_m16n8k16_f16_f32(_c2_[1], _a2_7, _b1_7, _c2_[1]);
            emmy_mma_m16n8k16_f16_f32(_c3_[0], _a3_7, _b0_7, _c3_[0]);
            emmy_mma_m16n8k16_f16_f32(_c3_[1], _a3_7, _b1_7, _c3_[1]);
            acc0__r0_0 = fmaxf(acc0__r0_0, _g__r0_0);
            acc0__r0_1 = fmaxf(acc0__r0_1, _g__r0_1);
            acc0__r1_0 = fmaxf(acc0__r1_0, _g__r1_0);
            acc0__r1_1 = fmaxf(acc0__r1_1, _g__r1_1);
            acc0__r2_0 = fmaxf(acc0__r2_0, _g__r2_0);
            acc0__r2_1 = fmaxf(acc0__r2_1, _g__r2_1);
            acc0__r3_0 = fmaxf(acc0__r3_0, _g__r3_0);
            acc0__r3_1 = fmaxf(acc0__r3_1, _g__r3_1);
        }
        float _ep0_0_v2_0 = 1.0f / acc1__r0_0;
        float _ep0_0_v2_1 = 1.0f / acc1__r0_1;
        float _ep0_0_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep0_0_acc2[_e] = _c0_[0][_e] * ((_e < 2) ? (_ep0_0_v2_0) : (_ep0_0_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 0] = _ep0_0_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 1] = _ep0_0_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep0_0_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep0_0_acc2[3];
          }
        }
        float _ep0_1_v2_0 = 1.0f / acc1__r0_0;
        float _ep0_1_v2_1 = 1.0f / acc1__r0_1;
        float _ep0_1_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep0_1_acc2[_e] = _c0_[1][_e] * ((_e < 2) ? (_ep0_1_v2_0) : (_ep0_1_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 0] = _ep0_1_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 1] = _ep0_1_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep0_1_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 0) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep0_1_acc2[3];
          }
        }
        float _ep1_0_v2_0 = 1.0f / acc1__r1_0;
        float _ep1_0_v2_1 = 1.0f / acc1__r1_1;
        float _ep1_0_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep1_0_acc2[_e] = _c1_[0][_e] * ((_e < 2) ? (_ep1_0_v2_0) : (_ep1_0_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 0] = _ep1_0_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 1] = _ep1_0_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep1_0_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep1_0_acc2[3];
          }
        }
        float _ep1_1_v2_0 = 1.0f / acc1__r1_0;
        float _ep1_1_v2_1 = 1.0f / acc1__r1_1;
        float _ep1_1_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep1_1_acc2[_e] = _c1_[1][_e] * ((_e < 2) ? (_ep1_1_v2_0) : (_ep1_1_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 0] = _ep1_1_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 1] = _ep1_1_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep1_1_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 16) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 16) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep1_1_acc2[3];
          }
        }
        float _ep2_0_v2_0 = 1.0f / acc1__r2_0;
        float _ep2_0_v2_1 = 1.0f / acc1__r2_1;
        float _ep2_0_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep2_0_acc2[_e] = _c2_[0][_e] * ((_e < 2) ? (_ep2_0_v2_0) : (_ep2_0_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 0] = _ep2_0_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 1] = _ep2_0_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep2_0_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep2_0_acc2[3];
          }
        }
        float _ep2_1_v2_0 = 1.0f / acc1__r2_0;
        float _ep2_1_v2_1 = 1.0f / acc1__r2_1;
        float _ep2_1_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep2_1_acc2[_e] = _c2_[1][_e] * ((_e < 2) ? (_ep2_1_v2_0) : (_ep2_1_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 0] = _ep2_1_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 1] = _ep2_1_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep2_1_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 32) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 32) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep2_1_acc2[3];
          }
        }
        float _ep3_0_v2_0 = 1.0f / acc1__r3_0;
        float _ep3_0_v2_1 = 1.0f / acc1__r3_1;
        float _ep3_0_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep3_0_acc2[_e] = _c3_[0][_e] * ((_e < 2) ? (_ep3_0_v2_0) : (_ep3_0_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 0] = _ep3_0_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16) + (_g) * 40 + _t * 2 + 1] = _ep3_0_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep3_0_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 0) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep3_0_acc2[3];
          }
        }
        float _ep3_1_v2_0 = 1.0f / acc1__r3_0;
        float _ep3_1_v2_1 = 1.0f / acc1__r3_1;
        float _ep3_1_acc2[4];
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _ep3_1_acc2[_e] = _c3_[1][_e] * ((_e < 2) ? (_ep3_1_v2_0) : (_ep3_1_v2_1));
        }
        { const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 0] = _ep3_1_acc2[0];
          }
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16 + 8) + (_g) * 40 + _t * 2 + 1] = _ep3_1_acc2[1];
          }
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 0] = _ep3_1_acc2[2];
          }
          if ((a0_b * 1024 + a0_u * 64 + 48) + _g + 8 < (90) && (a3_b * 32 + a3_u * 16 + 8) + _t * 2 + 1 < (40)) {
            matmul[(a0_b * 1024 + a0_u * 64 + 48) * 40 + (a3_b * 32 + a3_u * 16 + 8) + ((_g + 8)) * 40 + _t * 2 + 1] = _ep3_1_acc2[3];
          }
        }
}
