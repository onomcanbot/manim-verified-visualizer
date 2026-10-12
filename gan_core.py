"""
================================================================================
gan_core.py · GAN 모드 붕괴 데모의 '계산 + 검증' 모듈 (렌더링과 분리)
================================================================================

무엇을 보여주나
  8개 가우시안이 원 위에 놓인 데이터에서
    기본 GAN   : 생성자가 판별자를 쉽게 속이는 소수의 모드에만 몰림 (모드 붕괴)
    PacGAN     : 판별자가 샘플을 m개씩 묶어서 판정 → 묶음의 다양성이 부족하면 들킴
                 (Lin et al., "PacGAN", NeurIPS 2018)
  두 모델을 같은 시드·같은 조건으로 실제 학습시키고, 그 스냅샷을 그대로 영상에 쓴다.

검증 관문 (빠른 것부터, 하나라도 실패하면 즉시 중단 → 렌더링 차단)
  1) 기울기 검사   : 직접 구현한 역전파 vs 중앙 차분 수치미분 (무작위 소형 네트워크 20개)
  2) 지표 검사     : 모드 커버리지를 벡터화 / 순수 파이썬 두 방식으로 계산해 일치 확인
                    + 보정 검사 (진짜 데이터는 반드시 K/K 모드로 측정되어야 함)
                    + 불변식 (0 ≤ 덮은 모드 ≤ K, 비율 합 ≤ 1)
  3) 재현성·출처   : 같은 설정으로 처음부터 다시 학습 → 영상에 쓸 스냅샷과 비트 단위로 같은지
                    (손으로 그린 '그럴듯한 붕괴'나 시드 미고정을 여기서 잡는다)

설정 : 기본값(DEFAULT) 또는 환경변수 GAN_CONFIG 가 가리키는 JSON

실행
  python gan_core.py              → 검증 리포트
  python gan_core.py --inject     → 전형적 오류 주입 실험 (각 오류가 어느 관문에서 잡히는지)
  python gan_core.py --seeds 5    → 여러 시드 요약 (논문 보고용: 한 시드만 고른 결과가 아님을 확인)
================================================================================
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import numpy as np

DEFAULT = {
    "seed": 1,
    "steps": 3000,
    "lr": 3e-3,
    "std": 0.02,          # 각 모드의 표준편차 (좁을수록 붕괴가 잘 일어남)
    "pack": 2,            # PacGAN 묶음 크기
    "K": 8,               # 모드 수
    "R": 2.0,             # 원의 반지름
    "zdim": 16,
    "hidden": 64,
    "batch": 128,
    "slope": 0.2,         # LeakyReLU 기울기
    "radius": 0.25,       # 모드 '근처' 판정 반경
    "min_frac": 0.03,     # 샘플의 3% 이상이 근처에 있으면 그 모드를 덮었다고 판정
    "snap_every": 250,
    "n_eval": 1000,
}
CONFIG_ENV = "GAN_CONFIG"
GRAD_TOL = 1e-6
N_RANDOM = 20


def load_config(cfg: dict | None = None) -> dict:
    c = copy.deepcopy(DEFAULT)
    if cfg is None:
        path = os.environ.get(CONFIG_ENV)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in DEFAULT})
    for k in ("seed", "steps", "pack", "K", "zdim", "hidden", "batch", "snap_every", "n_eval"):
        c[k] = int(c[k])
    for k in ("lr", "std", "R", "slope", "radius", "min_frac"):
        c[k] = float(c[k])
    if c["K"] != 8 or c["pack"] < 2:
        raise ValueError("영상 레이아웃은 모드 8개, PacGAN 묶음 크기 2 이상 기준입니다.")
    return c


# ============================================================================
# 소형 MLP (NumPy, 역전파 직접 구현)
# ============================================================================
def leaky_grad(z, slope):
    return np.where(z > 0, 1.0, slope)


def mlp_init(sizes, rng):
    return [(rng.normal(0, math.sqrt(2.0 / i), (i, o)), np.zeros(o)) for i, o in zip(sizes[:-1], sizes[1:])]


def mlp_fwd(P, x, slope):
    acts, pre, h = [x], [], x
    for k, (W, b) in enumerate(P):
        z = h @ W + b
        pre.append(z)
        h = z if k == len(P) - 1 else np.where(z > 0, z, slope * z)
        acts.append(h)
    return h, (acts, pre)


def mlp_bwd(P, cache, gout, slope, dact=leaky_grad):
    acts, pre = cache
    grads, g = [None] * len(P), gout
    for k in range(len(P) - 1, -1, -1):
        if k != len(P) - 1:
            g = g * dact(pre[k], slope)
        grads[k] = (acts[k].T @ g, g.sum(0))
        g = g @ P[k][0].T
    return grads, g


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# --- 손실과 기울기 (analytic) ------------------------------------------------
def d_loss(D, xr, xf, slope):
    lr_, _ = mlp_fwd(D, xr, slope)
    lf, _ = mlp_fwd(D, xf, slope)
    return float(softplus(-lr_).mean() + softplus(lf).mean())


def d_grads(D, xr, xf, slope, dact=leaky_grad):
    B = len(xr)
    lr_, cr = mlp_fwd(D, xr, slope)
    lf, cf = mlp_fwd(D, xf, slope)
    g1, _ = mlp_bwd(D, cr, -sigmoid(-lr_) / B, slope, dact)
    g2, _ = mlp_bwd(D, cf, sigmoid(lf) / B, slope, dact)
    return [(a[0] + b[0], a[1] + b[1]) for a, b in zip(g1, g2)]


def g_loss(G, D, z, pack, slope):
    xg, _ = mlp_fwd(G, z, slope)
    lf, _ = mlp_fwd(D, xg.reshape(-1, 2 * pack), slope)
    return float(softplus(-lf).mean())          # non-saturating 생성자 손실


def g_grads(G, D, z, pack, slope, dact=leaky_grad, sign=1.0):
    xg, cg = mlp_fwd(G, z, slope)
    xp = xg.reshape(-1, 2 * pack)
    lf, cf = mlp_fwd(D, xp, slope)
    _, gx = mlp_bwd(D, cf, -sigmoid(-lf) / len(xp), slope, dact)
    gG, _ = mlp_bwd(G, cg, gx.reshape(-1, 2), slope, dact)
    return [(sign * a, sign * b) for a, b in gG]


class Adam:
    def __init__(self, P, lr, b1=0.5, b2=0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]
        self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]

    def step(self, P, G):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        out = []
        for k, ((W, b), (gW, gb)) in enumerate(zip(P, G)):
            (mW, mb), (vW, vb) = self.m[k], self.v[k]
            mW, mb = self.b1 * mW + (1 - self.b1) * gW, self.b1 * mb + (1 - self.b1) * gb
            vW, vb = self.b2 * vW + (1 - self.b2) * gW ** 2, self.b2 * vb + (1 - self.b2) * gb ** 2
            self.m[k], self.v[k] = (mW, mb), (vW, vb)
            out.append((W - self.lr * (mW / c1) / (np.sqrt(vW / c2) + 1e-8),
                        b - self.lr * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)))
        return out


# ============================================================================
# 데이터와 지표
# ============================================================================
def mode_centers(c):
    a = 2 * np.pi * np.arange(c["K"]) / c["K"]
    return np.stack([c["R"] * np.cos(a), c["R"] * np.sin(a)], 1)


def sample_real(n, rng, c):
    k = rng.integers(0, c["K"], n)
    return mode_centers(c)[k] + rng.normal(0, c["std"], (n, 2))


def coverage(x, c, min_frac=None):
    """벡터화 구현: (덮은 모드 수, 근처 샘플 비율, 모드별 비율)."""
    mf = c["min_frac"] if min_frac is None else min_frac
    d = np.linalg.norm(x[:, None, :] - mode_centers(c)[None], axis=2)
    j, near = d.argmin(1), d.min(1) < c["radius"]
    share = np.bincount(j[near], minlength=c["K"]) / len(x)
    return int((share >= mf).sum()), float(near.mean()), share


def coverage_loop(x, c):
    """순수 파이썬 구현 (독립 경로)."""
    C = mode_centers(c).tolist()
    cnt, near = [0] * c["K"], 0
    for px, py in np.asarray(x).tolist():
        best, bd = 0, float("inf")
        for k, (cx, cy) in enumerate(C):
            dd = math.hypot(px - cx, py - cy)
            if dd < bd:
                best, bd = k, dd
        if bd < c["radius"]:
            cnt[best] += 1
            near += 1
    share = [v / len(x) for v in cnt]
    return sum(1 for s in share if s >= c["min_frac"]), near / len(x), np.array(share)


# ============================================================================
# 학습
# ============================================================================
def train(c, pack, impl=None):
    """pack=1 → 기본 GAN, pack≥2 → PacGAN. 스냅샷은 고정 잡음 z_eval 로 만든 샘플."""
    impl = impl or {}
    dact = impl.get("dact", leaky_grad)
    rng = np.random.default_rng(c["seed"]) if impl.get("seeded", True) else np.random.default_rng()
    zeval = np.random.default_rng(c["seed"] + 10007).normal(size=(c["n_eval"], c["zdim"]))
    G = mlp_init([c["zdim"], c["hidden"], c["hidden"], 2], rng)
    D = mlp_init([2 * pack, c["hidden"], c["hidden"], 1], rng)
    oG, oD = Adam(G, c["lr"]), Adam(D, c["lr"])
    B, s = c["batch"], c["slope"]
    snaps, steps_at = [], []
    for t in range(c["steps"] + 1):
        if t % c["snap_every"] == 0:
            snaps.append(mlp_fwd(G, zeval, s)[0])
            steps_at.append(t)
        if t == c["steps"]:
            break
        xr = sample_real(B * pack, rng, c).reshape(B, 2 * pack)
        z = rng.normal(size=(B * pack, c["zdim"]))
        xf = mlp_fwd(G, z, s)[0].reshape(B, 2 * pack)
        D = oD.step(D, d_grads(D, xr, xf, s, dact))
        G = oG.step(G, g_grads(G, D, z, pack, s, dact, impl.get("g_sign", 1.0)))
    return dict(snaps=snaps, steps=steps_at)


# ============================================================================
# 검증 관문
# ============================================================================
def _flat(P):
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in P])


def _unflat(v, like):
    out, i = [], 0
    for W, b in like:
        nW, nb = W.size, b.size
        out.append((v[i:i + nW].reshape(W.shape), v[i + nW:i + nW + nb].reshape(b.shape)))
        i += nW + nb
    return out


def _num_grad(f, P, eps=1e-6):
    v = _flat(P)
    g = np.zeros_like(v)
    for k in range(len(v)):
        vp, vm = v.copy(), v.copy()
        vp[k] += eps
        vm[k] -= eps
        g[k] = (f(_unflat(vp, P)) - f(_unflat(vm, P))) / (2 * eps)
    return g


def gate_gradients(impl=None, n=N_RANDOM, seed=0):
    """무작위 소형 네트워크에서 D·G 손실 기울기를 수치미분과 대조 (상대 오차)."""
    impl = impl or {}
    dact, sign = impl.get("dact", leaky_grad), impl.get("g_sign", 1.0)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        pack, zd, h, B, s = int(rng.integers(1, 4)), int(rng.integers(2, 5)), int(rng.integers(3, 7)), 4, 0.2
        G = mlp_init([zd, h, h, 2], rng)
        D = mlp_init([2 * pack, h, h, 1], rng)
        xr, z = rng.normal(size=(B, 2 * pack)), rng.normal(size=(B * pack, zd))
        xf = mlp_fwd(G, z, s)[0].reshape(B, 2 * pack)
        ga = _flat(d_grads(D, xr, xf, s, dact))
        gn = _num_grad(lambda P: d_loss(P, xr, xf, s), D)
        worst = max(worst, float(np.max(np.abs(ga - gn) / np.maximum(1.0, np.abs(gn)))))
        ga = _flat(g_grads(G, D, z, pack, s, dact, sign))
        gn = _num_grad(lambda P: g_loss(P, D, z, pack, s), G)
        worst = max(worst, float(np.max(np.abs(ga - gn) / np.maximum(1.0, np.abs(gn)))))
    return worst


def gate_metric(c, samples, impl=None):
    """지표 두 구현 일치 + 보정(진짜 데이터 = K/K) + 불변식."""
    impl = impl or {}
    mf = impl.get("min_frac")
    mismatch = 0
    for x in samples:
        a = coverage(x, c, mf)
        b = coverage_loop(x, c)
        if a[0] != b[0] or abs(a[1] - b[1]) > 1e-12 or np.abs(a[2] - b[2]).max() > 1e-12:
            mismatch += 1
    real = sample_real(c["n_eval"], np.random.default_rng(424242), c)
    rc, rnear, rshare = coverage(real, c, mf)
    calib = (rc == c["K"]) and (rnear >= 0.99)
    inv = all(0 <= coverage(x, c, mf)[0] <= c["K"] and coverage(x, c, mf)[2].sum() <= 1 + 1e-12
              for x in samples)
    return dict(mismatch=mismatch, calib=calib, real_cov=rc, real_near=rnear, inv=inv)


def gate_repro(c, runs, impl=None):
    """같은 설정으로 처음부터 다시 학습 → 표시할 스냅샷과 비트 단위 비교."""
    impl = dict(impl or {})
    impl.pop("fake", None)                       # 재학습은 항상 실제 학습 코드로
    worst = 0.0
    for pack, shown in runs.items():
        again = train(c, pack, impl)["snaps"]
        worst = max(worst, max(float(np.abs(a - b).max()) for a, b in zip(shown, again)))
    return worst


def _fake_collapse(c, run):
    """'그럴듯한 붕괴'를 손으로 그린 가짜 스냅샷 (오류 주입용)."""
    mu = mode_centers(c)[0]
    rng = np.random.default_rng(0)
    return dict(snaps=[mu + rng.normal(0, c["std"], (c["n_eval"], 2)) for _ in run["snaps"]],
                steps=run["steps"])


def verify(c, runs, impl=None):
    """관문 1 → 2 → 3 순서로, 실패하면 그 자리에서 멈춘다."""
    out = dict(grad_err=None, metric=None, repro=None, stopped_at=None, passed=False)
    t0 = time.time()
    out["grad_err"] = gate_gradients(impl)
    if not out["grad_err"] < GRAD_TOL:
        out["stopped_at"] = "기울기 검사"
        return out
    allsnaps = [s for r in runs.values() for s in r["snaps"]]
    m = gate_metric(c, allsnaps, impl)
    out["metric"] = m
    if m["mismatch"] or not m["calib"] or not m["inv"]:
        out["stopped_at"] = "지표 검사"
        return out
    out["repro"] = gate_repro(c, {p: r["snaps"] for p, r in runs.items()}, impl)
    if out["repro"] != 0.0:
        out["stopped_at"] = "재현성·출처 검사"
        return out
    out["passed"] = True
    out["seconds"] = time.time() - t0
    return out


# ============================================================================
# 실행
# ============================================================================
def run(cfg: dict | None = None, impl=None) -> dict:
    c = load_config(cfg)
    impl = impl or {}
    runs = {1: train(c, 1, impl), c["pack"]: train(c, c["pack"], impl)}
    if impl.get("fake"):
        runs[1] = _fake_collapse(c, runs[1])
    v = verify(c, runs, impl)
    hist = {p: [coverage(s, c)[0] for s in r["snaps"]] for p, r in runs.items()}
    final = {p: coverage(r["snaps"][-1], c) for p, r in runs.items()}
    K = c["K"]
    collapsed = {p: final[p][0] <= K // 2 for p in runs}
    return dict(config=c, centers=mode_centers(c), runs=runs, hist=hist,
                final={p: dict(covered=f[0], near=f[1], share=f[2]) for p, f in final.items()},
                collapsed=collapsed, verify=v, passed=v["passed"],
                real=sample_real(c["n_eval"], np.random.default_rng(424242), c))


def report(r) -> str:
    c, v = r["config"], r["verify"]
    p = c["pack"]
    lines = ["=" * 66, " GAN 모드 붕괴 검증 리포트", "=" * 66,
             f" 설정: 시드 {c['seed']}, {c['steps']}단계, 학습률 {c['lr']:g}, 모드 표준편차 {c['std']:g}, "
             f"PacGAN 묶음 {p}",
             f" 덮은 모드 (반경 {c['radius']:g} 안에 샘플 {c['min_frac']:.0%} 이상):",
             f"   기본 GAN : {r['final'][1]['covered']}/{c['K']}  추이 {r['hist'][1]}",
             f"   PacGAN   : {r['final'][p]['covered']}/{c['K']}  추이 {r['hist'][p]}",
             f"   기본 GAN 최대 모드 점유율 : {r['final'][1]['share'].max():.0%}",
             "-" * 66,
             f" 관문 1 기울기 검사 (무작위 소형망 {N_RANDOM}개) 최대 상대오차 : {v['grad_err']:.2e}"]
    if v["metric"]:
        m = v["metric"]
        lines.append(f" 관문 2 지표 검사 : 두 구현 불일치 {m['mismatch']}건, "
                     f"보정 (진짜 데이터 {m['real_cov']}/{c['K']}, 근처 {m['real_near']:.1%}), "
                     f"불변식 {'OK' if m['inv'] else 'FAIL'}")
    if v["repro"] is not None:
        lines.append(f" 관문 3 재현성·출처 : 재학습 스냅샷과 최대 차이 {v['repro']:.1e}")
    lines.append(" 결과 : " + ("PASS" if v["passed"] else f"FAIL ({v['stopped_at']}에서 차단)"))
    lines.append("=" * 66)
    return "\n".join(lines)


def inject_report(cfg: dict | None = None) -> str:
    c = load_config(cfg)
    c["steps"] = min(c["steps"], 600)            # 오류 주입은 짧게 학습해도 충분히 드러난다
    rows = [("정상 구현", {}),
            ("LeakyReLU 미분 오류 (음수 쪽 기울기 1)", {"dact": lambda z, s: np.ones_like(z)}),
            ("생성자 기울기 부호 반대", {"g_sign": -1.0}),
            ("지표 임계값 오타 (3% → 30%)", {"min_frac": 0.30}),
            ("시드 미고정", {"seeded": False}),
            ("손으로 그린 '그럴듯한 붕괴'", {"fake": True})]
    out = [f" (오류 주입은 {c['steps']}단계 학습으로 실행)",
           f" {'주입한 오류':<30} 결과"]
    for name, impl in rows:
        r = run(c, impl)
        v = r["verify"]
        out.append(f" {name:<30} " + ("PASS" if v["passed"] else f"FAIL → {v['stopped_at']}에서 차단"))
    return "\n".join(out)


def seeds_report(n_seeds: int, cfg: dict | None = None) -> str:
    c = load_config(cfg)
    res = {1: [], c["pack"]: []}
    for s in range(n_seeds):
        c2 = dict(c, seed=s)
        for p in res:
            res[p].append(coverage(train(c2, p)["snaps"][-1], c2)[0])
    out = [f" 시드 0~{n_seeds - 1} 최종 덮은 모드 수 (K={c['K']})"]
    for p, v in res.items():
        name = "기본 GAN" if p == 1 else f"PacGAN({p})"
        out.append(f"   {name:<10}: {v}  평균 {np.mean(v):.1f} ± {np.std(v):.1f}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report(run()))
    if "--inject" in sys.argv:
        print()
        print(inject_report())
    if "--seeds" in sys.argv:
        k = sys.argv.index("--seeds")
        n = int(sys.argv[k + 1]) if len(sys.argv) > k + 1 else 5
        print()
        print(seeds_report(n))
