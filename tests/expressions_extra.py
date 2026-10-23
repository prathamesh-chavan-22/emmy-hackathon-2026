"""Extra test problems: our own torch expressions, same format as tools/expressions_open.py.

Generate (needs the torch venv, test-only - never part of compile.py):

    .venv-gen/Scripts/python tools/generate.py tests/expressions_extra.py \
        --emmy "<absolute path>/.venv-gen/Scripts/emmy.exe" --out tests/problems

The venv is Python 3.12 + torch CPU + emmy-ml==0.3.11 + scipy. On Windows three extra
steps are needed: PYTHONUTF8=1 (emmy writes its dumps in the locale encoding), an
absolute --emmy path, and a no-op `fcntl.py` in the venv's site-packages (emmy's CLI
imports the Unix-only fcntl for one flock call; generation runs single-process).

Each spec stresses something the open set does not: odd sizes (unroll tails), sizes with
more cells than threads, 2-D/3-D shapes and broadcasting, more elementwise ops, chains,
and more reduction shapes. Tolerances are deliberately tight - the hidden set's are unknown:
0 for ops that are exact in f32, 1e-6 for transcendentals, 1e-4..1e-3 for reductions.
"""

import numpy as np
from scipy.special import erf

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"
EXACT = dict(rtol=0, atol=0)
TRANS = dict(rtol=1e-6, atol=1e-6)
RED = dict(rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS)
MM = dict(rtol=1e-3, atol=1e-3)

f64 = lambda a: a.astype(np.float64)
f32 = lambda a: np.asarray(a).astype(np.float32)
f16 = lambda a: np.asarray(a).astype(np.float16)

# Other dtypes. Runner inputs are standard normal cast to the dtype, so ints are
# small and signed (truncated toward zero), which exercises // and % on negatives.
H = "dtype=torch.float16"
D = "dtype=torch.float64"
I32 = "torch.randint(-3, 3, (1024,), dtype=torch.int32)"
I64 = "torch.randint(-3, 3, (1024,), dtype=torch.int64)"
HALF_ULP = dict(rtol=1e-3, atol=1e-3)     # one f16 ulp is 2^-10 relative
# submit.py reads results back as float32, so an exact f64 result is off by that
# rounding alone (2^-24 relative); f32 math (emmy's for f64) would round twice.
F64_EXACT = dict(rtol=1e-7, atol=0)


# f16 tensor cores: f16 inputs, f32 accumulation, one rounding to f16 at the end
# (two when torch rounds an intermediate), so a result is within about one f16 ulp.
MMH = dict(rtol=2e-3, atol=2e-3)
ATTH = dict(rtol=1e-2, atol=1e-2)   # the softmax weights are rounded to f16 too


def _mmh(a, b):
    return f16(f64(a) @ f64(b))


def _attn_h(q, kt, v, scale, causal=False, mask=None):
    """torch's f16 attention: each op's result rounded to f16."""
    s = f16(f16(f64(q) @ f64(kt)) * scale)
    if mask is not None:
        s = f16(s + mask)
    s = f64(s)
    if causal:
        s = np.where(np.tril(np.ones(s.shape[-2:], dtype=bool)), s, -np.inf)
    return f16(f64(f16(_softmax(s))) @ f64(v))


def _gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


OPEN = [
    # ---- category 1: pointwise -------------------------------------------
    # sizes: tails that do not divide the unroll factor, and more cells than threads
    dict(id="x_p_odd7", code="torch.randn(7) + torch.randn(7)",
         ref=lambda a, b: [a + b], **EXACT),
    dict(id="x_p_odd1001", code="torch.randn(1001) * torch.randn(1001)",
         ref=lambda a, b: [a * b], **EXACT),
    dict(id="x_p_sub1000", code="torch.randn(1000) - torch.randn(1000)",
         ref=lambda a, b: [a - b], **EXACT),
    dict(id="x_p_big", code="torch.randn(1048576) + torch.randn(1048576)",
         ref=lambda a, b: [a + b], **EXACT),
    # shapes and broadcasting
    dict(id="x_p_bcast_row", code="torch.randn(64, 128) + torch.randn(128)",
         ref=lambda a, b: [a + b], **EXACT),
    dict(id="x_p_bcast_col", code="torch.randn(64, 1) * torch.randn(64, 128)",
         ref=lambda a, b: [a * b], **EXACT),
    dict(id="x_p_3d", code="torch.randn(4, 8, 32) * torch.randn(4, 8, 32)",
         ref=lambda a, b: [a * b], **EXACT),
    dict(id="x_p_3d_bcast", code="torch.randn(4, 1, 32) + torch.randn(1, 8, 32)",
         ref=lambda a, b: [a + b], **EXACT),
    dict(id="x_p_odd_2d", code="torch.randn(96, 1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], **EXACT),
    # ops that are exact in f32
    dict(id="x_p_div", code="x = torch.randn(2048); y = torch.randn(2048); x / y",
         ref=lambda x, y: [x / y], **EXACT),
    dict(id="x_p_neg_abs", code="x = torch.randn(1024); -torch.abs(x)",
         ref=lambda x: [-np.abs(x)], **EXACT),
    dict(id="x_p_maxmin", code="x = torch.randn(2048); y = torch.randn(2048); "
                               "torch.maximum(x, y) - torch.minimum(x, y)",
         ref=lambda x, y: [np.maximum(x, y) - np.minimum(x, y)], **EXACT),
    dict(id="x_p_clamp", code="torch.clamp(torch.randn(4096), -0.5, 0.5)",
         ref=lambda x: [np.clip(x, np.float32(-0.5), np.float32(0.5))], **EXACT),
    dict(id="x_p_where", code="x = torch.randn(2048); y = torch.randn(2048); torch.where(x > 0, x, y)",
         ref=lambda x, y: [np.where(x > 0, x, y)], **EXACT),
    dict(id="x_p_square", code="x = torch.randn(2048); x * x",
         ref=lambda x: [x * x], **EXACT),
    dict(id="x_p_sub_const", code="-torch.randn(1024) + 3.0",
         ref=lambda x: [np.float32(3.0) - x], **EXACT),
    dict(id="x_p_sqrt", code="torch.sqrt(torch.abs(torch.randn(4096)))",
         ref=lambda x: [np.sqrt(np.abs(x))], **EXACT),
    dict(id="x_p_chain3", code="x = torch.randn(4096); y = torch.randn(4096); z = torch.randn(4096); "
                               "(x * 1.5 + y) * z - 0.25 * x",
         ref=lambda x, y, z: [f32((f64(x) * 1.5 + f64(y)) * f64(z) - 0.25 * f64(x))], **TRANS),
    # transcendentals
    dict(id="x_p_exp", code="torch.exp(torch.randn(4096))",
         ref=lambda x: [f32(np.exp(f64(x)))], **TRANS),
    dict(id="x_p_exp_wide", code="torch.exp(torch.randn(4096) * 20.0)",
         ref=lambda x: [f32(np.exp(f64(np.float32(20.0) * x)))], rtol=1e-6, atol=0),
    dict(id="x_p_log", code="torch.log(torch.abs(torch.randn(4096)) + 0.5)",
         ref=lambda x: [f32(np.log(f64(np.abs(x) + np.float32(0.5))))], **TRANS),
    dict(id="x_p_rsqrt", code="torch.rsqrt(torch.abs(torch.randn(4096)) + 0.1)",
         ref=lambda x: [f32(1.0 / np.sqrt(f64(np.abs(x) + np.float32(0.1))))], **TRANS),
    dict(id="x_p_recip", code="1.0 / (torch.abs(torch.randn(4096)) + 1.0)",
         ref=lambda x: [f32(1.0 / f64(np.abs(x) + np.float32(1.0)))], **TRANS),
    dict(id="x_p_sigmoid", code="torch.sigmoid(torch.randn(4096))",
         ref=lambda x: [f32(1.0 / (1.0 + np.exp(-f64(x))))], **TRANS),
    dict(id="x_p_tanh", code="torch.tanh(torch.randn(4096))",
         ref=lambda x: [f32(np.tanh(f64(x)))], **TRANS),
    dict(id="x_p_silu", code="torch.nn.functional.silu(torch.randn(4096))",
         ref=lambda x: [f32(f64(x) / (1.0 + np.exp(-f64(x))))], **TRANS),
    dict(id="x_p_gelu", code="torch.nn.functional.gelu(torch.randn(4096))",
         ref=lambda x: [f32(0.5 * f64(x) * (1.0 + erf(f64(x) / np.sqrt(2.0))))], **TRANS),
    dict(id="x_p_pow3", code="torch.randn(2048) ** 3",
         ref=lambda x: [f32(f64(x) ** 3)], **TRANS),
    # second batch: more ops a hidden pointwise problem could use. emmy 0.3.11 itself cannot
    # trace ceil, sign, leaky_relu, elu, hardtanh or min() (so the hidden set cannot contain them).
    dict(id="x_p_pow_frac", code="torch.abs(torch.randn(4096)) ** 0.7",
         ref=lambda x: [f32(np.abs(f64(x)) ** 0.7)], rtol=1e-5, atol=1e-6),
    dict(id="x_p_pow_half", code="torch.abs(torch.randn(4096)) ** 0.5",
         ref=lambda x: [f32(np.sqrt(f64(np.abs(x))))], **TRANS),
    dict(id="x_p_cmp_mask", code="x = torch.randn(4096); y = torch.randn(4096); "
                                 "torch.where((x >= y) & (x < 0.5), x, -y)",
         ref=lambda x, y: [np.where((x >= y) & (x < np.float32(0.5)), x, -y)], **EXACT),
    dict(id="x_p_softplus", code="torch.nn.functional.softplus(torch.randn(4096))",
         ref=lambda x: [f32(np.logaddexp(0.0, f64(x)))], **TRANS),
    dict(id="x_p_gelu_tanh", code="torch.nn.functional.gelu(torch.randn(4096), approximate='tanh')",
         ref=lambda x: [f32(0.5 * f64(x) * (1 + np.tanh(np.sqrt(2 / np.pi) * (f64(x) + 0.044715 * f64(x) ** 3))))],
         **TRANS),
    dict(id="x_p_mish", code="x = torch.randn(4096); x * torch.tanh(torch.nn.functional.softplus(x))",
         ref=lambda x: [f32(f64(x) * np.tanh(np.logaddexp(0.0, f64(x))))], **TRANS),
    # third batch: activations emmy has no single op for, written the way it can trace them.
    # Rejected by emmy 0.3.11 itself: expm1, log1p, relu6, hardsigmoid, hardswish, selu.
    dict(id="x_p_leaky_where", code="x = torch.randn(4096); torch.where(x > 0, x, x * 0.1)",
         ref=lambda x: [np.where(x > 0, x, x * np.float32(0.1))], **EXACT),
    dict(id="x_p_hardtanh_clamp", code="torch.clamp(torch.randn(4096) * 2.0, -1.0, 1.0)",
         ref=lambda x: [np.clip(x * np.float32(2), np.float32(-1), np.float32(1))], **EXACT),
    dict(id="x_p_elu_where", code="x = torch.randn(4096); torch.where(x > 0, x, torch.exp(x) - 1.0)",
         ref=lambda x: [f32(np.where(f64(x) > 0, f64(x), np.exp(f64(x)) - 1.0))], **TRANS),
    dict(id="x_p_sin_cos", code="x = torch.randn(4096); torch.sin(x) + torch.cos(x)",
         ref=lambda x: [f32(np.sin(f64(x)) + np.cos(f64(x)))], **TRANS),
    dict(id="x_p_softsign", code="torch.nn.functional.softsign(torch.randn(4096))",
         ref=lambda x: [f32(f64(x) / (1 + np.abs(f64(x))))], **TRANS),
    dict(id="x_p_tanhshrink", code="torch.nn.functional.tanhshrink(torch.randn(4096))",
         ref=lambda x: [f32(f64(x) - np.tanh(f64(x)))], rtol=1e-5, atol=1e-6),
    # inputs are always standard normal (constants are not inputs): abs()+0.5 keeps the base positive
    dict(id="x_p_pow_tensor", code="torch.pow(torch.randn(4096).abs() + 0.5, torch.randn(4096))",
         ref=lambda a, b: [f32(f64(np.abs(a) + np.float32(0.5)) ** f64(b))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_neg_pow", code="(torch.randn(4096).abs() + 0.5) ** -2",
         ref=lambda x: [f32(f64(np.abs(x) + np.float32(0.5)) ** -2)], rtol=1e-6, atol=1e-6),
    # a*b+c must round twice like numpy: a fused fma differs in the last bit
    dict(id="x_p_mul_add_exact", code="torch.randn(64, 128) * torch.randn(128) + torch.randn(64, 128)",
         ref=lambda a, b, c: [a * b + c], **EXACT),
    # masks mixed with numbers: emmy's C casts bool <-> float implicitly, PTX does not
    dict(id="x_p_mask_mul", code="x = torch.randn(4096); x * (x > 0)",
         ref=lambda x: [x * (x > 0)], **EXACT),
    dict(id="x_p_mask_float", code="x = torch.randn(4096); (x > 0.5).float() + x",
         ref=lambda x: [(x > 0.5).astype(np.float32) + x], **EXACT),
    dict(id="x_p_mask_and", code="x = torch.randn(4096); torch.where((x > -1.0) & (x < 1.0), x, 0.0)",
         ref=lambda x: [np.where((x > -1) & (x < 1), x, np.float32(0))], **EXACT),
    dict(id="x_p_mask_not", code="x = torch.randn(4096); torch.where(~(x > 0), x * 2.0, x)",
         ref=lambda x: [np.where(~(x > 0), x * np.float32(2), x)], **EXACT),

    # ---- category 2: reductions (pinned like the open set) ----------------
    dict(id="x_r_sum100", code="torch.randn(100).sum()",
         ref=lambda x: [f32(f64(x).sum())], **RED),
    dict(id="x_r_sum1000", code="torch.randn(1000).sum()",
         ref=lambda x: [f32(f64(x).sum())], **RED),
    dict(id="x_r_max", code="torch.randn(4096).amax()",
         ref=lambda x: [x.max()], **RED),
    dict(id="x_r_sumsq", code="x = torch.randn(8192); (x * x).sum()",
         ref=lambda x: [f32((f64(x) ** 2).sum())], **RED),
    dict(id="x_r_mean_rows", code="torch.randn(32, 300).mean(-1)",
         ref=lambda x: [f32(f64(x).mean(-1))], **RED),
    dict(id="x_r_rowsum_wide", code="torch.randn(16, 4096).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], **RED),
    dict(id="x_r_rowmax", code="torch.randn(128, 256).amax(-1)",
         ref=lambda x: [x.max(-1)], **RED),
    dict(id="x_r_colsum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [f32(f64(x).sum(0))], **RED),
    # unpinned: emmy may split across blocks with an atomic add (category-3 concept)
    dict(id="x_r_unpinned_sum", code="torch.randn(65536).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-3, atol=1e-3),

    # ---- category 4: fused (preview, for phase 4) --------------------------
    dict(id="x_f_softmax_wide", code="torch.softmax(torch.randn(32, 1000), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_layernorm_ish", code="x = torch.randn(64, 512); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x) - f64(x).mean(-1, keepdims=True))], rtol=1e-3, atol=1e-4),

    # ---- round 2: the categories the open set covers thinly -----------------
    # matmul: tiny, non-square, odd (non-tile-multiple) sizes, long K (split-K), batched, epilogues
    dict(id="x_m_small", code="torch.randn(32, 32) @ torch.randn(32, 32)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_m_tall", code="torch.randn(1024, 64) @ torch.randn(64, 32)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_m_wide", code="torch.randn(16, 128) @ torch.randn(128, 512)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_m_longk", code="torch.randn(64, 2048) @ torch.randn(2048, 64)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-2),
    dict(id="x_m_odd", code="torch.randn(100, 70) @ torch.randn(70, 50)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_m_256", code="torch.randn(256, 384) @ torch.randn(384, 256)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_m_batched", code="torch.randn(4, 64, 32) @ torch.randn(4, 32, 48)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_m_bias_relu", code="a = torch.randn(128, 64); b = torch.randn(64, 96); c = torch.randn(96); "
                                  "torch.relu(a @ b + c)",
         ref=lambda a, b, c: [f32(np.maximum(f64(a) @ f64(b) + f64(c), 0))], **MM),
    dict(id="x_m_scaled", code="(torch.randn(64, 128) @ torch.randn(128, 64)) * 0.5",
         ref=lambda a, b: [f32((f64(a) @ f64(b)) * 0.5)], **MM),

    # fused: full layernorm, log-softmax, scaled softmax, norm + activation, odd widths
    # Intermediates must be written inline: the tracer turns a named intermediate
    # (`d = x - m`) into a fresh random input.
    dict(id="x_f_layernorm", code="x = torch.randn(32, 768); w = torch.randn(768); b = torch.randn(768); "
                                  "(x - x.mean(-1, keepdim=True)) / torch.sqrt("
                                  "((x - x.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True) + 1e-5) * w + b",
         ref=lambda x, w, b: [f32((f64(x) - f64(x).mean(-1, keepdims=True))
                                  / np.sqrt(f64(x).var(-1, keepdims=True) + 1e-5) * f64(w) + f64(b))],
         rtol=1e-3, atol=1e-4),
    # torch.log_softmax does not trace in emmy 0.3.11; the hand-written form does.
    dict(id="x_f_log_softmax", code="x = torch.randn(64, 200); "
                                    "x - x.amax(-1, keepdim=True) - torch.log(torch.exp("
                                    "x - x.amax(-1, keepdim=True)).sum(-1, keepdim=True))",
         ref=lambda x: [f32(np.log(_softmax(f64(x))))], rtol=1e-3, atol=1e-4),
    dict(id="x_f_softmax_scaled", code="torch.softmax(torch.randn(16, 384) * 0.125, dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x) * 0.125))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_rmsnorm_silu", code="x = torch.randn(48, 256); "
                                     "torch.nn.functional.silu(x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6))",
         ref=lambda x: [f32((lambda y: y / (1 + np.exp(-y)))(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6)))],
         rtol=1e-3, atol=1e-4),
    dict(id="x_f_softmax_odd", code="torch.softmax(torch.randn(7, 33), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_softmax_dim0", code="torch.softmax(torch.randn(64, 32), dim=0)",
         ref=lambda x: [f32(_softmax(f64(x), axis=0))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_max_sub", code="x = torch.randn(64, 300); x - x.amax(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x) - f64(x).max(-1, keepdims=True))], rtol=1e-5, atol=1e-6),

    # attention: other sequence lengths / head dims, non-square Q vs K
    dict(id="x_a_32x16", code="q = torch.randn(32, 16); kt = torch.randn(16, 32); v = torch.randn(32, 16); "
                              "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 4.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_256x64", code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
                               "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 8.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_rect", code="q = torch.randn(64, 32); kt = torch.randn(32, 192); v = torch.randn(192, 48); "
                             "torch.softmax(q @ kt * 0.17677669, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) * 0.17677669) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_batched", code="q = torch.randn(4, 64, 32); kt = torch.randn(4, 32, 64); v = torch.randn(4, 64, 32); "
                               "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 5.656854) @ f64(v))], rtol=1e-3, atol=1e-4),

    # reductions: other ops and axes
    # prod() does not trace in emmy 0.3.11 (no elementwise mapping for aten.prod).
    dict(id="x_r_mean_3d", code="torch.randn(8, 16, 128).mean(-1)",
         ref=lambda x: [f32(f64(x).mean(-1))], **RED),
    dict(id="x_r_sum_dim0", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [f32(f64(x).sum(0))], **RED),
    dict(id="x_r_abs_max", code="torch.randn(4096).abs().amax()",
         ref=lambda x: [f32(np.abs(x).max())], **RED),
    dict(id="x_r_unpinned_rows", code="torch.randn(32, 4096).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-3, atol=1e-3),
    dict(id="x_r_mean_mid_3d", code="torch.randn(4, 64, 8).mean(1)",
         ref=lambda x: [f32(f64(x).mean(1))], **RED),
    # a scan: each output is a running sum, not one value per row
    dict(id="x_r_cumsum", code="torch.cumsum(torch.randn(256), 0)",
         ref=lambda x: [f32(np.cumsum(f64(x)))], rtol=1e-4, atol=1e-4),
    dict(id="x_r_cumsum_rows", code="torch.cumsum(torch.randn(16, 100), -1)",
         ref=lambda x: [f32(np.cumsum(f64(x), -1))], rtol=1e-4, atol=1e-4),

    # ---- other dtypes: f16 (f32 math rounded per op), f64, i32, i64 ------
    dict(id="x_d_f16_add", code=f"torch.randn(1000, {H}) + torch.randn(1000, {H})",
         ref=lambda a, b: [a + b], **EXACT),
    dict(id="x_d_f16_bcast_mul", code=f"torch.randn(64, 128, {H}) * torch.randn(128, {H})",
         ref=lambda a, b: [a * b], **EXACT),
    dict(id="x_d_f16_div_sqrt", code=f"x = torch.randn(2048, {H}); x / torch.sqrt(x.abs() + 1)",
         ref=lambda x: [x / np.sqrt(np.abs(x) + f16(1))], **EXACT),
    dict(id="x_d_f16_exp", code=f"torch.exp(torch.randn(1024, {H}))",
         ref=lambda x: [f16(np.exp(f64(x)))], **HALF_ULP),
    dict(id="x_d_f16_gelu", code=f"torch.nn.functional.gelu(torch.randn(4096, {H}))",
         ref=lambda x: [f16(0.5 * f64(x) * (1 + erf(f64(x) / np.sqrt(2))))], rtol=4e-3, atol=4e-3),
    dict(id="x_d_f16_softmax", code=f"torch.softmax(torch.randn(64, 256, {H}), dim=-1)",
         ref=lambda x: [f16(_softmax(f64(x)))], rtol=1e-2, atol=1e-4),
    dict(id="x_d_f16_sum", code=f"torch.randn(4096, {H}).sum()",
         ref=lambda x: [f16(f64(x).sum())], rtol=2e-3, atol=0.07),
    dict(id="x_d_f16_mean", code=f"torch.randn(64, 512, {H}).mean(dim=-1)",
         ref=lambda x: [f16(f64(x).mean(-1))], rtol=1e-2, atol=2e-3),
    dict(id="x_d_f32_to_f16", code="torch.randn(1024).half() * 2",
         ref=lambda x: [f16(x) * f16(2)], **EXACT),
    dict(id="x_d_f16_to_f32", code=f"torch.randn(1024, {H}).float() * 3.0",
         ref=lambda x: [f32(x) * np.float32(3)], **EXACT),
    dict(id="x_d_f64_add", code=f"torch.randn(1000, {D}) + torch.randn(1000, {D})",
         ref=lambda a, b: [a + b], **F64_EXACT),
    dict(id="x_d_f64_div", code=f"a = torch.randn(2048, {D}); b = torch.randn(2048, {D}); a * b / (b.abs() + 1)",
         ref=lambda a, b: [a * b / (np.abs(b) + 1)], **F64_EXACT),
    dict(id="x_d_f64_exp", code=f"torch.exp(torch.randn(1024, {D}))",
         ref=lambda x: [np.exp(x)], **TRANS),
    dict(id="x_d_f64_sum", code=f"torch.randn(4096, {D}).sum()",
         ref=lambda x: [x.sum()], **RED),
    dict(id="x_d_f64_softmax", code=f"torch.softmax(torch.randn(32, 128, {D}), dim=-1)",
         ref=lambda x: [_softmax(x)], rtol=1e-5, atol=1e-6),
    dict(id="x_d_f64_mm", code=f"torch.randn(64, 64, {D}) @ torch.randn(64, 64, {D})",
         ref=lambda a, b: [a @ b], **MM),
    dict(id="x_d_i32_add", code=f"{I32} + {I32}",
         ref=lambda a, b: [a + b], **EXACT),
    dict(id="x_d_i64_mul_add", code=f"(lambda a, b: a * b + b)({I64}, {I64})",
         ref=lambda a, b: [a * b + b], **EXACT),
    dict(id="x_d_i32_floordiv", code=f"(lambda a, b: a // (b.abs() + 1))({I32}, {I32})",
         ref=lambda a, b: [a // (np.abs(b) + 1)], **EXACT),
    dict(id="x_d_i32_rem", code=f"(lambda a, b: a % (b.abs() + 1))({I32}, {I32})",
         ref=lambda a, b: [a % (np.abs(b) + 1)], **EXACT),
    dict(id="x_d_i32_floordiv_c", code=f"{I32} // 2",
         ref=lambda a: [a // 2], **EXACT),
    dict(id="x_d_i32_bitwise", code=f"(lambda a, b: torch.bitwise_xor(torch.bitwise_and(a, b), torch.bitwise_or(a, b)))({I32}, {I32})",
         ref=lambda a, b: [(a & b) ^ (a | b)], **EXACT),
    dict(id="x_d_i32_rshift", code=f"torch.bitwise_right_shift({I32}, 1)",
         ref=lambda a: [a >> 1], **EXACT),
    dict(id="x_d_i32_cmp", code=f"(lambda a, b: (a > b).float())({I32}, {I32})",
         ref=lambda a, b: [f32(a > b)], **EXACT),
    dict(id="x_d_i32_where", code=f"(lambda a: torch.where(a > 0, a, -a))({I32})",
         ref=lambda a: [np.where(a > 0, a, -a)], **EXACT),
    dict(id="x_d_i32_abs_sub", code=f"{I32}.abs() - 1",
         ref=lambda a: [np.abs(a) - 1], **EXACT),
    dict(id="x_d_i32_mixed", code=f"{I32} * 0.5 + torch.randn(1024)",
         ref=lambda a, b: [f32(a) * np.float32(0.5) + b], **EXACT),
    dict(id="x_d_f32_to_i32", code="(torch.randn(1024) * 10).to(torch.int32)",
         ref=lambda x: [(x * np.float32(10)).astype(np.int32)], **EXACT),
    dict(id="x_d_i32_to_f32", code=f"{I32}.float() / 3",
         ref=lambda a: [f32(a) / np.float32(3)], **TRANS),
    dict(id="x_d_i32_max", code="torch.randint(-50, 50, (64, 256), dtype=torch.int32).amax(dim=-1)",
         ref=lambda x: [x.max(-1)], **EXACT),
    dict(id="x_d_i64_max", code="torch.randint(-50, 50, (64, 256), dtype=torch.int64).amax(dim=-1)",
         ref=lambda x: [x.max(-1)], **EXACT),
    dict(id="x_d_i32_sum", code="torch.randint(-3, 3, (4096,), dtype=torch.int32).sum()",
         ref=lambda x: [x.astype(np.int64).sum()], **EXACT),
    dict(id="x_d_i64_cumsum", code="torch.randint(0, 5, (16, 64), dtype=torch.int64).cumsum(dim=-1)",
         ref=lambda x: [np.cumsum(x, -1)], **EXACT),
    # ---- f16 tensor cores (mma.sync m16n8k16): matmul shapes, epilogues, attention ----
    dict(id="x_h_mm_16", code=f"torch.randn(16, 16, {H}) @ torch.randn(16, 16, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_64", code=f"torch.randn(64, 64, {H}) @ torch.randn(64, 64, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_128", code=f"torch.randn(128, 128, {H}) @ torch.randn(128, 128, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_256", code=f"torch.randn(256, 256, {H}) @ torch.randn(256, 256, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_1024", code=f"torch.randn(1024, 1024, {H}) @ torch.randn(1024, 1024, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_rect", code=f"torch.randn(64, 256, {H}) @ torch.randn(256, 96, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_odd", code=f"torch.randn(100, 72, {H}) @ torch.randn(72, 50, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_vec", code=f"torch.randn(1, 256, {H}) @ torch.randn(256, 256, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_bmm", code=f"torch.randn(4, 64, 64, {H}) @ torch.randn(4, 64, 64, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_mm_t", code=f"torch.randn(128, 64, {H}) @ torch.randn(128, 64, {H}).t()",
         ref=lambda a, b: [_mmh(a, b.T)], **MMH),
    dict(id="x_h_linear", code=f"torch.nn.functional.linear(torch.randn(64, 128, {H}), torch.randn(256, 128, {H}))",
         ref=lambda x, w: [_mmh(x, w.T)], **MMH),
    dict(id="x_h_mm_relu", code=f"torch.relu(torch.randn(128, 128, {H}) @ torch.randn(128, 128, {H}))",
         ref=lambda a, b: [np.maximum(_mmh(a, b), f16(0))], **MMH),
    dict(id="x_h_mm_bias", code=f"torch.randn(128, 128, {H}) @ torch.randn(128, 128, {H}) + torch.randn(128, {H})",
         ref=lambda a, b, c: [_mmh(a, b) + c], **MMH),
    dict(id="x_h_mlp", code=f"x = torch.randn(64, 128, {H}); w1 = torch.randn(128, 256, {H}); "
                            f"w2 = torch.randn(256, 128, {H}); torch.relu(x @ w1) @ w2",
         # an f16 intermediate on a rounding boundary may round the other way under another
         # f32 summation order: one ulp of h (~0.016) times a weight
         ref=lambda x, w1, w2: [_mmh(np.maximum(_mmh(x, w1), f16(0)), w2)], rtol=2e-3, atol=0.06),
    dict(id="x_h_mm_mixed", code=f"torch.randn(64, 64, {H}).float() @ torch.randn(64, 64)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    dict(id="x_h_attn", code=f"q = torch.randn(64, 64, {H}); kt = torch.randn(64, 64, {H}); "
                             f"v = torch.randn(64, 64, {H}); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn_h(q, kt, v, 1 / 8)], **ATTH),
    dict(id="x_h_attn_128", code=f"q = torch.randn(128, 64, {H}); kt = torch.randn(64, 128, {H}); "
                                 f"v = torch.randn(128, 64, {H}); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn_h(q, kt, v, 1 / 8)], **ATTH),
    dict(id="x_h_attn_b", code=f"q = torch.randn(4, 64, 32, {H}); kt = torch.randn(4, 32, 64, {H}); "
                               f"v = torch.randn(4, 64, 32, {H}); torch.softmax(q @ kt / 5.66, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn_h(q, kt, v, 1 / 5.66)], **ATTH),
    dict(id="x_h_sdpa", code=f"torch.nn.functional.scaled_dot_product_attention(torch.randn(1, 4, 64, 32, {H}), "
                             f"torch.randn(1, 4, 64, 32, {H}), torch.randn(1, 4, 64, 32, {H}))",
         ref=lambda q, k, v: [_attn_h(q, np.swapaxes(k, -1, -2), v, 1 / np.sqrt(32))], **ATTH),
    dict(id="x_h_sdpa_causal", code=f"torch.nn.functional.scaled_dot_product_attention(torch.randn(1, 2, 64, 32, {H}), "
                                    f"torch.randn(1, 2, 64, 32, {H}), torch.randn(1, 2, 64, 32, {H}), is_causal=True)",
         ref=lambda q, k, v: [_attn_h(q, np.swapaxes(k, -1, -2), v, 1 / np.sqrt(32), causal=True)], **ATTH),
    dict(id="x_h_sdpa_mask", code=f"torch.nn.functional.scaled_dot_product_attention(torch.randn(1, 2, 64, 32, {H}), "
                                  f"torch.randn(1, 2, 64, 32, {H}), torch.randn(1, 2, 64, 32, {H}), "
                                  f"attn_mask=torch.randn(64, 64, {H}))",
         ref=lambda q, k, v, m: [_attn_h(q, np.swapaxes(k, -1, -2), v, 1 / np.sqrt(32), mask=m)], **ATTH),
    # odd sequence lengths (masked tiles, zero-filled K); the head dim stays a multiple of 16:
    # with 40, emmy's tiling drops k = 32..39 of q @ kt (its own reference kernel too)
    dict(id="x_h_attn_odd", code=f"q = torch.randn(50, 32, {H}); kt = torch.randn(32, 70, {H}); "
                                 f"v = torch.randn(70, 24, {H}); torch.softmax(q @ kt / 6.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn_h(q, kt, v, 1 / 6)], **ATTH),
    dict(id="x_h_mm_odd2", code=f"torch.randn(33, 40, {H}) @ torch.randn(40, 17, {H})",
         ref=lambda a, b: [_mmh(a, b)], **MMH),
    dict(id="x_h_bmm_t", code=f"q = torch.randn(2, 64, 32, {H}); k = torch.randn(2, 96, 32, {H}); q @ k.transpose(-1, -2)",
         ref=lambda q, k: [_mmh(q, np.swapaxes(k, -1, -2))], **MMH),
    dict(id="x_h_linear_bias_gelu", code=f"torch.nn.functional.gelu(torch.nn.functional.linear("
                                         f"torch.randn(128, 64, {H}), torch.randn(96, 64, {H}), torch.randn(96, {H})))",
         ref=lambda x, w, b: [f16(_gelu(f64(_mmh(x, w.T) + b)))], rtol=4e-3, atol=4e-3),
    dict(id="x_h_mm_residual", code=f"a = torch.randn(128, 64, {H}); b = torch.randn(64, 128, {H}); "
                                    f"r = torch.randn(128, 128, {H}); a @ b + r",
         ref=lambda a, b, r: [_mmh(a, b) + r], **MMH),
]
