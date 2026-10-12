"""
================================================================================
backprop_core.py · 역전파 데모의 '계산 + 검증' 모듈 (렌더링과 분리)
================================================================================

파이프라인 원칙  ── "계산 먼저, 그림은 나중"
  1) 이 모듈이 NumPy로 순전파·역전파를 실제로 계산하고
  2) 같은 값을 독립된 참조 구현(중앙 차분 수치미분, 가능하면 PyTorch autograd)과 대조
     - 데모 값 1세트 + 무작위 입력 N세트 (퇴화된 입력이 버그를 가리지 않도록)
  3) 검증을 통과한 숫자만 Manim 씬(backprop_demo.py)이 화면에 그린다.

네트워크 : 2-2-1 MLP, 시그모이드 활성화, 손실 L = ½(ŷ − y)²
  h₁ = σ(w₁x₁ + w₂x₂ + b₁)      h₂ = σ(w₃x₁ + w₄x₂ + b₂)
  ŷ  = σ(w₅h₁ + w₆h₂ + b₃)

설정 : 기본값(DEFAULT) 또는 환경변수 BACKPROP_CONFIG 가 가리키는 JSON 파일
       (웹 화면 app.py 가 이 방식으로 값을 넘긴다)

실행
  python backprop_core.py            → 검증 리포트
  python backprop_core.py --inject   → 전형적 오류 주입 시 검증이 잡아내는지 확인
================================================================================
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------------
DEFAULT = {
    "x": [1.0, 0.5],        # 입력 x₁, x₂
    "y": 1.0,               # 정답 y
    "lr": 1.0,              # 학습률 η
    "params": {
        "w1": 0.5,  "w2": -0.3, "b1": 0.1,     # → h₁
        "w3": 0.2,  "w4": 0.8,  "b2": -0.1,    # → h₂
        "w5": 0.9,  "w6": -0.6, "b3": 0.2,     # → ŷ
    },
}
CONFIG_ENV = "BACKPROP_CONFIG"
PARAM_ORDER = ["w1", "w2", "w3", "w4", "w5", "w6", "b1", "b2", "b3"]
TOL = 1e-8
N_RANDOM = 20


def load_config(cfg: dict | None = None) -> dict:
    """기본값 위에 (인자 cfg 또는 환경변수 JSON) 값을 덮어쓴다."""
    c = copy.deepcopy(DEFAULT)
    if cfg is None:
        path = os.environ.get(CONFIG_ENV)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
    if cfg:
        for k in ("x", "y", "lr"):
            if k in cfg:
                c[k] = cfg[k]
        c["params"].update(cfg.get("params", {}))
    c["x"] = [float(v) for v in c["x"]]
    c["y"], c["lr"] = float(c["y"]), float(c["lr"])
    c["params"] = {k: float(v) for k, v in c["params"].items()}
    return c


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# 순전파 / 역전파
# ----------------------------------------------------------------------------
def forward(p: dict, x, y) -> dict:
    x1, x2 = x
    z1 = p["w1"] * x1 + p["w2"] * x2 + p["b1"]
    z2 = p["w3"] * x1 + p["w4"] * x2 + p["b2"]
    h1, h2 = sigmoid(z1), sigmoid(z2)
    zo = p["w5"] * h1 + p["w6"] * h2 + p["b3"]
    yhat = sigmoid(zo)
    loss = 0.5 * (yhat - y) ** 2
    return dict(x1=x1, x2=x2, y=y, z1=z1, z2=z2, h1=h1, h2=h2,
                zo=zo, yhat=yhat, loss=loss)


def backward(p: dict, f: dict) -> dict:
    """중간값(δ)까지 함께 반환해 영상에서 연쇄법칙 단계를 그대로 보여줄 수 있게 한다."""
    dL_dyhat = f["yhat"] - f["y"]
    dyhat_dzo = f["yhat"] * (1.0 - f["yhat"])
    delta_o = dL_dyhat * dyhat_dzo
    dh1_dz1 = f["h1"] * (1.0 - f["h1"])
    dh2_dz2 = f["h2"] * (1.0 - f["h2"])
    delta_1 = delta_o * p["w5"] * dh1_dz1
    delta_2 = delta_o * p["w6"] * dh2_dz2
    grads = {
        "w5": delta_o * f["h1"], "w6": delta_o * f["h2"], "b3": delta_o,
        "w1": delta_1 * f["x1"], "w2": delta_1 * f["x2"], "b1": delta_1,
        "w3": delta_2 * f["x1"], "w4": delta_2 * f["x2"], "b2": delta_2,
    }
    inter = dict(dL_dyhat=dL_dyhat, dyhat_dzo=dyhat_dzo, delta_o=delta_o,
                 dh1_dz1=dh1_dz1, dh2_dz2=dh2_dz2, delta_1=delta_1, delta_2=delta_2)
    return dict(grads=grads, inter=inter)


# ----------------------------------------------------------------------------
# 참조 구현
# ----------------------------------------------------------------------------
def numerical_grads(p: dict, x, y, eps: float = 1e-6) -> dict:
    g = {}
    for k in PARAM_ORDER:
        plus, minus = dict(p), dict(p)
        plus[k] += eps
        minus[k] -= eps
        g[k] = (forward(plus, x, y)["loss"] - forward(minus, x, y)["loss"]) / (2 * eps)
    return g


def torch_grads(p: dict, x, y):
    try:
        import torch
    except ImportError:
        return None
    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in p.items()}
    x1, x2 = (torch.tensor(float(v), dtype=torch.float64) for v in x)
    h1 = torch.sigmoid(t["w1"] * x1 + t["w2"] * x2 + t["b1"])
    h2 = torch.sigmoid(t["w3"] * x1 + t["w4"] * x2 + t["b2"])
    yhat = torch.sigmoid(t["w5"] * h1 + t["w6"] * h2 + t["b3"])
    (0.5 * (yhat - y) ** 2).backward()
    return {k: t[k].grad.item() for k in PARAM_ORDER}


# ----------------------------------------------------------------------------
# 검증 : 데모 설정 + 무작위 설정
# ----------------------------------------------------------------------------
def random_configs(n=N_RANDOM, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        p = {k: float(rng.normal()) for k in PARAM_ORDER}
        yield p, [float(v) for v in rng.uniform(-2, 2, 2)], float(rng.uniform(0, 1))


def _max_err(g, ref):
    return max(abs(g[k] - ref[k]) for k in PARAM_ORDER)


def verify_impl(backward_fn, c: dict) -> dict:
    p, x, y = c["params"], c["x"], c["y"]
    g = backward_fn(p, forward(p, x, y))["grads"]
    err_demo = _max_err(g, numerical_grads(p, x, y))
    gt = torch_grads(p, x, y)
    err_torch = None if gt is None else _max_err(g, gt)
    worst = err_demo
    for rp, rx, ry in random_configs():
        rg = backward_fn(rp, forward(rp, rx, ry))["grads"]
        worst = max(worst, _max_err(rg, numerical_grads(rp, rx, ry)))
    passed = worst < TOL and (err_torch is None or err_torch < TOL)
    return dict(err_demo=err_demo, err_num=worst, err_torch=err_torch,
                passed=passed, n_random=N_RANDOM)


def sgd_step(p: dict, grads: dict, lr: float) -> dict:
    return {k: p[k] - lr * grads[k] for k in p}


def run(cfg: dict | None = None) -> dict:
    """영상에 필요한 모든 숫자를 계산하고 검증 결과까지 묶어서 반환."""
    c = load_config(cfg)
    p, x, y = dict(c["params"]), c["x"], c["y"]
    f = forward(p, x, y)
    b = backward(p, f)
    v = verify_impl(backward, c)
    p_new = sgd_step(p, b["grads"], c["lr"])
    return dict(config=c, params=p, fwd=f, grads=b["grads"], inter=b["inter"],
                g_num=numerical_grads(p, x, y), params_new=p_new,
                fwd_new=forward(p_new, x, y), lr=c["lr"], **v)


def report(r: dict) -> str:
    lines = ["=" * 64, " 역전파 검증 리포트 (해석적 기울기 vs 참조 구현)", "=" * 64,
             f" 순전파 : h1={r['fwd']['h1']:.4f}, h2={r['fwd']['h2']:.4f}, "
             f"ŷ={r['fwd']['yhat']:.4f}, L={r['fwd']['loss']:.5f}",
             "-" * 64,
             f" {'param':>5} | {'backprop':>12} | {'numeric':>12} | {'|diff|':>9}"]
    for k in PARAM_ORDER:
        a, n = r["grads"][k], r["g_num"][k]
        lines.append(f" {k:>5} | {a:>12.8f} | {n:>12.8f} | {abs(a - n):>9.1e}")
    lines += ["-" * 64,
              f" 수치미분 대조 (데모 값)            : {r['err_demo']:.2e}",
              f" 수치미분 대조 (데모 + 무작위 {r['n_random']}세트) : {r['err_num']:.2e}",
              " autograd 대조                      : "
              + ("PyTorch 미설치 → 생략" if r["err_torch"] is None else f"{r['err_torch']:.2e}"),
              f" 허용 오차 {TOL:.0e} → {'PASS' if r['passed'] else 'FAIL'}",
              "-" * 64,
              f" 1회 갱신 (η={r['lr']:g}) : L {r['fwd']['loss']:.5f} → {r['fwd_new']['loss']:.5f}, "
              f"ŷ {r['fwd']['yhat']:.4f} → {r['fwd_new']['yhat']:.4f}",
              "=" * 64]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 오류 주입 실험 (논문용)
# ----------------------------------------------------------------------------
def _bug_sigma_missing(p, f):           # 출력층 σ'(z_o) 항 누락
    b = backward(p, f)
    g = dict(b["grads"])
    do = f["yhat"] - f["y"]
    g["w5"], g["w6"], g["b3"] = do * f["h1"], do * f["h2"], do
    return dict(grads=g, inter=b["inter"])


def _bug_w5_missing(p, f):              # 은닉층 δ₁ 계산에서 w₅ 곱 누락
    b = backward(p, f)
    g = dict(b["grads"])
    d1 = b["inter"]["delta_o"] * b["inter"]["dh1_dz1"]
    g["w1"], g["w2"], g["b1"] = d1 * f["x1"], d1 * f["x2"], d1
    return dict(grads=g, inter=b["inter"])


def _bug_rounded(p, f):                 # 그럴듯한 반올림 숫자 하드코딩
    b = backward(p, f)
    return dict(grads={k: round(v, 2) for k, v in b["grads"].items()}, inter=b["inter"])


def inject_report(cfg: dict | None = None) -> str:
    c = load_config(cfg)
    rows = [("정상 구현", backward), ("σ'(z_o) 항 누락", _bug_sigma_missing),
            ("은닉층 δ에서 w5 곱 누락", _bug_w5_missing), ("반올림 숫자 하드코딩", _bug_rounded)]
    out = [f" {'주입한 오류':<22} {'최악 오차':>12}  결과"]
    for name, fn in rows:
        v = verify_impl(fn, c)
        out.append(f" {name:<22} {v['err_num']:>12.2e}  "
                   + ("PASS" if v["passed"] else "FAIL → 렌더링 차단"))
    return "\n".join(out)


if __name__ == "__main__":
    print(report(run()))
    if "--inject" in sys.argv:
        print()
        print(inject_report())
