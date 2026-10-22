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

f64 = lambda a: a.astype(np.float64)
f32 = lambda a: np.asarray(a).astype(np.float32)


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
]
