"""
================================================================================
attention_core.py · 셀프 어텐션 데모의 '계산 + 검증' 모듈 (렌더링과 분리)
================================================================================

역전파 데모와 같은 원칙 ── "계산 먼저, 그림은 나중"
  다만 어텐션은 기울기가 아니므로 수치미분으로 검증할 수 없다. 대신
    (1) 독립 구현 대조  : 같은 식을 서로 다른 코드 경로로 계산해 비교
          A. 벡터화 NumPy (영상에 쓰는 구현)
          B. 순수 파이썬 루프 + math.exp (행렬 연산 없이 원소 단위로)
          C. scipy.special.softmax (Manim 설치 시 함께 설치됨)
          D. torch.nn.functional.scaled_dot_product_attention (설치된 경우만)
    (2) 수학적 불변식   : 각 행의 가중치 합 = 1, 모든 가중치 ≥ 0
  을 함께 확인한다. 둘 중 하나만으로는 잡히지 않는 오류가 있다(--inject 참고).
    (3) 무작위 입력 검증 : 데모 값 한 세트만 쓰면 퇴화된 입력(예: W_K = W_V)이 버그를 가린다.
                          크기·값이 다른 무작위 입력 N세트에서도 같은 대조를 반복한다.

장난감 예제 : 3개 토큰, d_model = d_k = 2
  임베딩 축  = [명사성, 동작성]
  W_K, W_V  = 단위행렬 (설명 단순화를 위해 키·값은 임베딩 그대로)
  W_Q       = 동사는 명사(특히 목적어)를, 명사는 동사를 찾도록 설계

실행
  python attention_core.py            → 검증 리포트
  python attention_core.py --inject   → 전형적 오류 4종 주입 시 검증이 잡아내는지 확인
================================================================================
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# 기본 설정 (기본값 또는 환경변수 ATTENTION_CONFIG 가 가리키는 JSON 으로 덮어씀)
#   영상 레이아웃은 토큰 3개, 2차원 임베딩 기준
# ----------------------------------------------------------------------------
DEFAULT = {
    "tokens": ["고양이가", "생선을", "먹었다"],
    "axis_names": ["명사성", "동작성"],
    "X": [[1.0, 0.5],       # 고양이가 : 명사, 살아있는 주체
          [1.0, -0.5],      # 생선을   : 명사, 대상
          [0.0, 1.0]],      # 먹었다   : 동작
    "W_Q": [[-0.75, 1.75],
            [1.50, -0.50]],
    "W_K": [[1.0, 0.0], [0.0, 1.0]],
    "W_V": [[1.0, 0.0], [0.0, 1.0]],
    "focus": 2,             # 영상에서 자세히 풀어볼 질문 단어 (먹었다)
}
CONFIG_ENV = "ATTENTION_CONFIG"
TOL = 1e-10
N_RANDOM = 20      # 무작위 검증 입력 개수


def load_config(cfg: dict | None = None) -> dict:
    c = copy.deepcopy(DEFAULT)
    if cfg is None:
        path = os.environ.get(CONFIG_ENV)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in DEFAULT})
    for k in ("X", "W_Q", "W_K", "W_V"):
        c[k] = np.asarray(c[k], dtype=float)
    c["focus"] = int(c["focus"])
    if len(c["tokens"]) != 3 or c["X"].shape != (3, 2) or c["W_Q"].shape != (2, 2):
        raise ValueError("영상 레이아웃은 토큰 3개, 2차원 임베딩, 2×2 W_Q 기준입니다.")
    return c


# ----------------------------------------------------------------------------
# A. 벡터화 구현 (영상에 표시되는 값)
# ----------------------------------------------------------------------------
def softmax_rows(S):
    E = np.exp(S - S.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def attention(X, WQ, WK, WV):
    Q, K, V = X @ WQ, X @ WK, X @ WV
    dk = K.shape[1]
    S = Q @ K.T / np.sqrt(dk)
    A = softmax_rows(S)
    Z = A @ V
    return dict(Q=Q, K=K, V=V, S=S, A=A, Z=Z, dk=dk)


# ----------------------------------------------------------------------------
# B. 순수 파이썬 루프 구현 (NumPy 행렬 연산을 쓰지 않는 독립 경로)
# ----------------------------------------------------------------------------
def attention_loop(X, WQ, WK, WV):
    X, WQ, WK, WV = (np.asarray(m).tolist() for m in (X, WQ, WK, WV))
    n, d = len(X), len(WQ[0])

    def matvec(row, W):
        return [sum(row[k] * W[k][j] for k in range(len(row))) for j in range(len(W[0]))]

    Q = [matvec(r, WQ) for r in X]
    K = [matvec(r, WK) for r in X]
    V = [matvec(r, WV) for r in X]
    A, Z = [], []
    for i in range(n):
        s = [sum(Q[i][t] * K[j][t] for t in range(d)) / math.sqrt(d) for j in range(n)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        tot = sum(e)
        a = [v / tot for v in e]
        A.append(a)
        Z.append([sum(a[j] * V[j][t] for j in range(n)) for t in range(len(V[0]))])
    return dict(A=np.array(A), Z=np.array(Z))


# ----------------------------------------------------------------------------
# C. scipy 구현
# ----------------------------------------------------------------------------
def attention_scipy(X, WQ, WK, WV):
    try:
        from scipy.special import softmax
    except ImportError:
        return None
    Q, K, V = np.einsum("nd,de->ne", X, WQ), np.einsum("nd,de->ne", X, WK), np.einsum("nd,de->ne", X, WV)
    S = np.einsum("ie,je->ij", Q, K) / math.sqrt(K.shape[1])
    A = softmax(S, axis=1)
    return dict(A=A, Z=np.einsum("ij,je->ie", A, V))


# ----------------------------------------------------------------------------
# D. PyTorch 구현 (선택)
# ----------------------------------------------------------------------------
def attention_torch(X, WQ, WK, WV):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None
    t = lambda m: torch.tensor(np.asarray(m), dtype=torch.float64)
    Xt = t(X)
    Q, K, V = Xt @ t(WQ), Xt @ t(WK), Xt @ t(WV)
    S = Q @ K.T / math.sqrt(K.shape[1])
    A = torch.softmax(S, dim=1)
    Z = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    return dict(A=A.numpy(), Z=Z.numpy())


# ----------------------------------------------------------------------------
# 검증
# ----------------------------------------------------------------------------
def _check(res, X, WQ, WK, WV):
    refs = {"loop": attention_loop(X, WQ, WK, WV),
            "scipy": attention_scipy(X, WQ, WK, WV),
            "torch": attention_torch(X, WQ, WK, WV)}
    diffs = {}
    for name, ref in refs.items():
        if ref is None:
            diffs[name] = None
            continue
        diffs[name] = max(np.abs(res["A"] - ref["A"]).max(), np.abs(res["Z"] - ref["Z"]).max())
    row_sum_err = float(np.abs(res["A"].sum(axis=1) - 1.0).max())
    nonneg = bool((res["A"] >= 0).all())
    used = [d for d in diffs.values() if d is not None]
    return dict(diffs=diffs, max_ref=max(used), row_sum_err=row_sum_err, nonneg=nonneg)


def random_configs(n=N_RANDOM, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        n_tok, d_model, d_k, d_v = (int(rng.integers(lo, hi)) for lo, hi in
                                    [(3, 7), (2, 6), (2, 5), (2, 5)])
        yield (rng.normal(size=(n_tok, d_model)), rng.normal(size=(d_model, d_k)),
               rng.normal(size=(d_model, d_k)), rng.normal(size=(d_model, d_v)))


def verify_impl(impl, c: dict | None = None):
    """impl(X, WQ, WK, WV) -> {'A','Z'} 를 데모 설정 + 무작위 설정에서 모두 대조."""
    c = load_config() if c is None else c
    X, W_Q, W_K, W_V = c["X"], c["W_Q"], c["W_K"], c["W_V"]
    demo = _check(impl(X, W_Q, W_K, W_V), X, W_Q, W_K, W_V)
    worst_ref, worst_row, all_nonneg = demo["max_ref"], demo["row_sum_err"], demo["nonneg"]
    for cfg in random_configs():
        c = _check(impl(*cfg), *cfg)
        worst_ref = max(worst_ref, c["max_ref"])
        worst_row = max(worst_row, c["row_sum_err"])
        all_nonneg &= c["nonneg"]
    passed = worst_ref < TOL and worst_row < TOL and all_nonneg
    return dict(diffs=demo["diffs"], max_ref=worst_ref, row_sum_err=worst_row,
                nonneg=all_nonneg, passed=passed, n_random=N_RANDOM)


def run(cfg: dict | None = None):
    c = load_config(cfg)
    res = attention(c["X"], c["W_Q"], c["W_K"], c["W_V"])
    v = verify_impl(attention, c)
    return dict(tokens=c["tokens"], axis_names=c["axis_names"], X=c["X"],
                WQ=c["W_Q"], WK=c["W_K"], WV=c["W_V"], focus=c["focus"], **res, **v)


def report(r) -> str:
    np.set_printoptions(precision=4, suppress=True)
    lines = ["=" * 64, " 셀프 어텐션 검증 리포트", "=" * 64,
             " 어텐션 가중치 A (행 = 질문 단어, 열 = 참고 단어)"]
    for tok, row in zip(r["tokens"], r["A"]):
        lines.append(f"   {tok:<6} " + "  ".join(f"{v:.4f}" for v in row))
    lines.append("-" * 64)
    for name, d in r["diffs"].items():
        lines.append(f" vs {name:<6}(데모 값): " + ("미설치 → 생략" if d is None else f"max|diff| = {d:.2e}"))
    lines.append(f" 데모 + 무작위 {r['n_random']}세트 최악 오차 : {r['max_ref']:.2e}")
    lines.append(f" 행 합 = 1 오차   : {r['row_sum_err']:.2e}")
    lines.append(f" 가중치 ≥ 0       : {r['nonneg']}")
    lines.append(f" 허용 오차 {TOL:.0e} → {'PASS' if r['passed'] else 'FAIL'}")
    lines.append("=" * 64)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 오류 주입 실험 (논문용) : LLM이 흔히 내는 실수를 넣었을 때 검증이 잡아내는가
# ----------------------------------------------------------------------------
def _faulty(kind, X, WQ, WK, WV):
    Q, K, V = X @ WQ, X @ WK, X @ WV
    S = Q @ K.T / np.sqrt(K.shape[1])
    if kind == "no_scaling":                    # √d_k 나누기 누락
        A = softmax_rows(Q @ K.T)
        Z = A @ V
    elif kind == "wrong_axis":                  # softmax를 열 방향으로
        E = np.exp(S)
        A = E / E.sum(axis=0, keepdims=True)
        Z = A @ V
    elif kind == "uses_K_not_V":                # 가중합에 V 대신 K 사용
        A = softmax_rows(S)
        Z = A @ K[:, :V.shape[1]] if K.shape[1] >= V.shape[1] else A @ np.pad(K, ((0, 0), (0, V.shape[1] - K.shape[1])))
    elif kind == "rounded_hardcode":            # 그럴듯한 반올림 숫자 하드코딩
        A = np.round(softmax_rows(S), 2)
        Z = A @ V
    return dict(A=A, Z=Z)


def inject_report(cfg: dict | None = None):
    from functools import partial
    c = load_config(cfg)
    X, W_Q, W_K, W_V = c["X"], c["W_Q"], c["W_K"], c["W_V"]
    impls = [("정상 구현", attention),
             ("√d_k 스케일링 누락", partial(_faulty, "no_scaling")),
             ("softmax 축 오류(열 방향)", partial(_faulty, "wrong_axis")),
             ("V 대신 K로 가중합", partial(_faulty, "uses_K_not_V")),
             ("반올림 숫자 하드코딩", partial(_faulty, "rounded_hardcode"))]
    out = [f" {'주입한 오류':<22} {'데모 값만':>12} {'데모+무작위':>12}  결과"]
    for name, impl in impls:
        demo_only = _check(impl(X, W_Q, W_K, W_V), X, W_Q, W_K, W_V)
        demo_pass = (demo_only["max_ref"] < TOL and demo_only["row_sum_err"] < TOL
                     and demo_only["nonneg"])
        v = verify_impl(impl, c)
        out.append(f" {name:<22} {'PASS' if demo_pass else 'FAIL':>12} "
                   f"{'PASS' if v['passed'] else 'FAIL':>12}  "
                   + ("" if v["passed"] else "→ 렌더링 차단"))
    return "\n".join(out)


if __name__ == "__main__":
    print(report(run()))
    if "--inject" in sys.argv:
        print()
        print(inject_report())
