"""
================================================================================
gcn_core.py · GCN 메시지 패싱 데모의 '계산 + 검증' 모듈 (렌더링과 분리)
================================================================================

GCN 한 층 (Kipf & Welling, 2017)
  Ã = A + I,   d̃_i = Σ_j Ã_ij,   Â = D̃^{-1/2} Ã D̃^{-1/2}
  H' = ReLU(Â X W)
  노드 관점 : h_i' = ReLU( Σ_{j ∈ N(i) ∪ {i}}  x_j W / √(d̃_i d̃_j) )

검증 ── 앞의 두 데모와 같은 원칙에 그래프 고유의 성질을 더함
  (1) 독립 구현 대조 : A. 밀집 NumPy (영상에 쓰는 구현)
                       B. 순수 파이썬 메시지 패싱 (간선 목록을 돌며 노드별로 합산)
                       C. scipy.sparse 희소 행렬
                       D. PyTorch 밀집 연산 (설치된 경우만)
  (2) 그래프 불변식   : Â 대칭 · Â√d̃ = √d̃ (고유값 1) · 모든 고유값 |λ| ≤ 1
                       · 순열 등변성 GCN(PX, PAPᵀ) = P·GCN(X, A)
                       · 수용 영역 : (Â^k)_ij ≠ 0  ⇔  j 가 i 의 k-hop 이내
  (3) 무작위 그래프 N개 (노드 수·간선·특징 차원이 모두 다름)

설정 : 기본값(DEFAULT) 또는 환경변수 GCN_CONFIG 가 가리키는 JSON
       영상 레이아웃은 노드 5개, 2차원 특징 기준

실행
  python gcn_core.py            → 검증 리포트
  python gcn_core.py --inject   → 전형적 오류 주입 실험
================================================================================
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from collections import deque

import numpy as np

DEFAULT = {
    "nodes": ["A", "B", "C", "D", "E"],
    "axis_names": ["축구", "독서"],
    "X": [[1.0, 0.0],      # A : 축구 성향
          [0.8, 0.2],      # B
          [0.5, 0.5],      # C : 두 무리를 잇는 허브
          [0.2, 0.8],      # D
          [0.0, 1.0]],     # E : 독서 성향
    "edges": [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]],
    "W": [[1.0, -0.5],
          [-0.5, 1.0]],    # 자기 성향은 키우고 반대 성향은 억제
    "focus": 2,            # 메시지 패싱을 자세히 풀어볼 노드 (C)
}
CONFIG_ENV = "GCN_CONFIG"
TOL = 1e-10
N_RANDOM = 20


# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
def load_config(cfg: dict | None = None) -> dict:
    c = copy.deepcopy(DEFAULT)
    if cfg is None:
        path = os.environ.get(CONFIG_ENV)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in DEFAULT})
    c["X"] = np.asarray(c["X"], dtype=float)
    c["W"] = np.asarray(c["W"], dtype=float)
    n = len(c["nodes"])
    edges = sorted({tuple(sorted((int(i), int(j)))) for i, j in c["edges"] if int(i) != int(j)})
    if n != 5 or c["X"].shape != (5, 2) or c["W"].shape != (2, 2):
        raise ValueError("영상 레이아웃은 노드 5개, 2차원 특징, 2×2 W 기준입니다.")
    if any(not (0 <= i < n and 0 <= j < n) for i, j in edges):
        raise ValueError("간선의 노드 번호가 범위를 벗어났습니다.")
    c["edges"] = [list(e) for e in edges]
    c["focus"] = int(c["focus"])
    return c


def adjacency(n, edges):
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A


# ----------------------------------------------------------------------------
# A. 밀집 NumPy 구현 (영상에 표시되는 값)
# ----------------------------------------------------------------------------
def gcn_dense(X, edges, n, W):
    A = adjacency(n, edges)
    At = A + np.eye(n)
    deg = At.sum(axis=1)
    Dm = np.diag(1.0 / np.sqrt(deg))
    Ahat = Dm @ At @ Dm
    AX = Ahat @ X
    Z = AX @ W
    return dict(A=A, deg=deg, Ahat=Ahat, AX=AX, Z=Z, H=np.maximum(Z, 0.0))


# ----------------------------------------------------------------------------
# B. 순수 파이썬 메시지 패싱 (행렬 없이 간선 목록만으로)
# ----------------------------------------------------------------------------
def gcn_loop(X, edges, n, W):
    X, W = np.asarray(X).tolist(), np.asarray(W).tolist()
    nbr = {i: {i} for i in range(n)}                    # 자기 자신 포함
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    deg = {i: len(nbr[i]) for i in range(n)}
    AX, H = [], []
    for i in range(n):
        agg = [0.0] * len(X[0])
        for j in nbr[i]:
            c = 1.0 / math.sqrt(deg[i] * deg[j])        # 메시지 가중치
            for t in range(len(agg)):
                agg[t] += c * X[j][t]
        out = [max(0.0, sum(agg[k] * W[k][m] for k in range(len(agg)))) for m in range(len(W[0]))]
        AX.append(agg)
        H.append(out)
    return dict(AX=np.array(AX), H=np.array(H))


# ----------------------------------------------------------------------------
# C. scipy.sparse
# ----------------------------------------------------------------------------
def gcn_sparse(X, edges, n, W):
    try:
        import scipy.sparse as sp
    except ImportError:
        return None
    if edges:
        r = [i for i, j in edges] + [j for i, j in edges]
        c = [j for i, j in edges] + [i for i, j in edges]
        A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    else:
        A = sp.coo_matrix((n, n))
    At = (A + sp.identity(n)).tocsr()
    d = np.asarray(At.sum(axis=1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    AX = Dm @ At @ Dm @ np.asarray(X)
    return dict(AX=np.asarray(AX), H=np.maximum(np.asarray(AX) @ np.asarray(W), 0.0))


# ----------------------------------------------------------------------------
# D. PyTorch (선택)
# ----------------------------------------------------------------------------
def gcn_torch(X, edges, n, W):
    try:
        import torch
    except ImportError:
        return None
    A = torch.zeros(n, n, dtype=torch.float64)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    At = A + torch.eye(n, dtype=torch.float64)
    Dm = torch.diag(At.sum(1).rsqrt())
    AX = Dm @ At @ Dm @ torch.tensor(np.asarray(X), dtype=torch.float64)
    H = torch.relu(AX @ torch.tensor(np.asarray(W), dtype=torch.float64))
    return dict(AX=AX.numpy(), H=H.numpy())


# ----------------------------------------------------------------------------
# 그래프 유틸
# ----------------------------------------------------------------------------
def hops_from(n, edges, src):
    """BFS 거리 (도달 불가 = None)."""
    nbr = {i: set() for i in range(n)}
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in nbr[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return [dist.get(i) for i in range(n)]


# ----------------------------------------------------------------------------
# 검증
# ----------------------------------------------------------------------------
def _check(impl, X, edges, n, W, rng):
    with np.errstate(all="ignore"):
        res = impl(X, edges, n, W)
        refs = {"loop": gcn_loop(X, edges, n, W), "sparse": gcn_sparse(X, edges, n, W),
                "torch": gcn_torch(X, edges, n, W)}
        diffs = {}
        for k, ref in refs.items():
            if ref is None:
                diffs[k] = None
                continue
            d = max(np.abs(res["AX"] - ref["AX"]).max(), np.abs(res["H"] - ref["H"]).max())
            diffs[k] = float(d) if np.isfinite(d) else float("inf")

        Ah = res["Ahat"]
        P = np.eye(n)[rng.permutation(n)]                         # 순열 (난수 소비 순서 고정)
        if not np.all(np.isfinite(Ah)):                           # 0으로 나누기 등 → 그 자체로 실패
            sym = eigvec = spec = perm = rf = float("inf")
        else:
            A = adjacency(n, edges)
            sq = np.sqrt((A + np.eye(n)).sum(1))                  # 독립 계산한 √d̃
            sym = float(np.abs(Ah - Ah.T).max())
            eigvec = float(np.abs(Ah @ sq - sq).max())
            lam = np.linalg.eigvals(Ah)
            spec = float(max(0.0, np.abs(lam).max() - 1.0))
            Pe = [(int(np.argmax(P[:, i])), int(np.argmax(P[:, j]))) for i, j in edges]
            perm = float(np.abs(impl(P @ X, Pe, n, W)["H"] - P @ res["H"]).max())
            rf = 0.0                                              # 수용 영역 (k = 1, 2)
            for s in range(n):
                hop = hops_from(n, edges, s)
                for k in (1, 2):
                    support = np.abs(np.linalg.matrix_power(Ah, k)[s]) > 1e-12
                    expect = np.array([h is not None and h <= k for h in hop])
                    rf = max(rf, float((support != expect).sum()))
    inv = dict(symmetric=sym, eigvec=eigvec, spectrum=spec, permutation=perm, receptive=rf)
    inv = {k: (v if np.isfinite(v) else float("inf")) for k, v in inv.items()}
    used = [d for d in diffs.values() if d is not None]
    return dict(diffs=diffs, max_ref=max(used), inv=inv)


def random_graphs(n=N_RANDOM, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        nn = int(rng.integers(4, 9))
        din, dout = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        edges = [[i, j] for i in range(nn) for j in range(i + 1, nn) if rng.random() < 0.4]
        yield rng.normal(size=(nn, din)), edges, nn, rng.normal(size=(din, dout))


def _passed(c):
    return c["max_ref"] < TOL and all(v < TOL for v in c["inv"].values())


def verify_impl(impl, c: dict | None = None) -> dict:
    c = load_config() if c is None else c
    rng = np.random.default_rng(123)
    n = len(c["nodes"])
    demo = _check(impl, c["X"], c["edges"], n, c["W"], rng)
    worst_ref, worst_inv = demo["max_ref"], dict(demo["inv"])
    for X, e, nn, W in random_graphs():
        r = _check(impl, X, e, nn, W, rng)
        worst_ref = max(worst_ref, r["max_ref"])
        worst_inv = {k: max(worst_inv[k], r["inv"][k]) for k in worst_inv}
    passed = worst_ref < TOL and all(v < TOL for v in worst_inv.values())
    return dict(diffs=demo["diffs"], max_ref=worst_ref, inv=worst_inv,
                demo_passed=_passed(demo), passed=passed, n_random=N_RANDOM)


def run(cfg: dict | None = None) -> dict:
    c = load_config(cfg)
    n = len(c["nodes"])
    res = gcn_dense(c["X"], c["edges"], n, c["W"])
    v = verify_impl(gcn_dense, c)
    f = c["focus"]
    nbrs = sorted({j for i, j in c["edges"] if i == f} | {i for i, j in c["edges"] if j == f})
    hop = hops_from(n, c["edges"], f)
    # 수용 영역 장면용 : 이심률(가장 먼 거리)이 큰 노드를 자동 선택 → 층이 늘 때 범위가 넓어지는 모습
    ecc = [max([h for h in hops_from(n, c["edges"], s) if h is not None]) for s in range(n)]
    rf_node = int(np.argmax(ecc))
    return dict(nodes=c["nodes"], axis_names=c["axis_names"], X=c["X"], W=c["W"],
                edges=c["edges"], focus=f, neighbors=nbrs, hop_focus=hop,
                rf_node=rf_node, rf_hops=hops_from(n, c["edges"], rf_node),
                A2=np.linalg.matrix_power(res["Ahat"], 2), **res, **v)


def report(r) -> str:
    nm = r["nodes"]
    lines = ["=" * 66, " GCN 메시지 패싱 검증 리포트", "=" * 66,
             " 간선 : " + ", ".join(f"{nm[i]}-{nm[j]}" for i, j in r["edges"]),
             " 노드 | d̃ |   입력 x     |  집계 Âx      |  출력 ReLU(ÂxW)"]
    for i in range(len(nm)):
        lines.append(f"  {nm[i]:<3} | {int(r['deg'][i])}  | "
                     f"({r['X'][i][0]:5.2f},{r['X'][i][1]:5.2f}) | "
                     f"({r['AX'][i][0]:5.3f},{r['AX'][i][1]:5.3f}) | "
                     f"({r['H'][i][0]:5.3f},{r['H'][i][1]:5.3f})")
    lines.append("-" * 66)
    for k, d in r["diffs"].items():
        lines.append(f" vs {k:<6}(데모 값) : " + ("미설치 → 생략" if d is None else f"{d:.2e}"))
    lines.append(f" 독립 구현 대조 (데모 + 무작위 그래프 {r['n_random']}개) 최악 : {r['max_ref']:.2e}")
    names = {"symmetric": "Â 대칭", "eigvec": "Â√d̃ = √d̃", "spectrum": "|λ| ≤ 1 초과분",
             "permutation": "순열 등변성", "receptive": "수용 영역 불일치 수"}
    for k, v in r["inv"].items():
        lines.append(f" 불변식 {names[k]:<14}: {v:.2e}")
    lines.append(f" 허용 오차 {TOL:.0e} → {'PASS' if r['passed'] else 'FAIL'}")
    lines.append("=" * 66)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 오류 주입 실험 (논문용)
# ----------------------------------------------------------------------------
def _bug(kind):
    def impl(X, edges, n, W):
        A = adjacency(n, edges)
        if kind == "one_direction":                 # 간선을 한 방향으로만 넣음
            A = np.zeros((n, n))
            for i, j in edges:
                A[i, j] = 1.0
        At = A if kind == "no_self_loop" else A + np.eye(n)
        deg = (A.sum(1) if kind == "deg_without_self" else At.sum(1))
        with np.errstate(all="ignore"):
            if kind == "row_norm":                  # 대칭 정규화 대신 행 정규화
                Ahat = np.diag(1.0 / deg) @ At
            else:
                Dm = np.diag(1.0 / np.sqrt(deg))
                Ahat = Dm @ At @ Dm
        AX = Ahat @ X
        Z = AX @ W
        H = Z if kind == "no_relu" else np.maximum(Z, 0.0)
        if kind == "rounded":
            AX, H = np.round(AX, 2), np.round(H, 2)
        return dict(Ahat=Ahat, AX=AX, H=H)
    return impl


def inject_report(cfg: dict | None = None) -> str:
    c = load_config(cfg)
    rows = [("정상 구현", gcn_dense),
            ("자기 루프(+I) 누락", _bug("no_self_loop")),
            ("차수에서 자기 루프 빠뜨림", _bug("deg_without_self")),
            ("행 정규화 D̃⁻¹Ã 사용", _bug("row_norm")),
            ("간선을 한 방향으로만", _bug("one_direction")),
            ("ReLU 누락", _bug("no_relu")),
            ("반올림 숫자 하드코딩", _bug("rounded"))]
    out = [f" {'주입한 오류':<24} {'데모 값만':>8} {'데모+무작위':>10}  결과"]
    for name, impl in rows:
        v = verify_impl(impl, c)
        out.append(f" {name:<24} {'PASS' if v['demo_passed'] else 'FAIL':>8} "
                   f"{'PASS' if v['passed'] else 'FAIL':>10}  " + ("" if v["passed"] else "→ 렌더링 차단"))
    # 퇴화 입력 실험 : 모든 노드의 차수가 같은 '정규 그래프'에서는 행 정규화 = 대칭 정규화
    reg = dict(c)
    reg["edges"] = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]          # 5-사이클
    v = verify_impl(_bug("row_norm"), reg)
    out.append("-" * 60)
    out.append(" [퇴화 입력] 5-사이클(정규 그래프)에서 행 정규화 오류")
    out.append(f"   데모 값만 : {'PASS (놓침)' if v['demo_passed'] else 'FAIL'}   "
               f"데모+무작위 : {'PASS' if v['passed'] else 'FAIL → 렌더링 차단'}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report(run()))
    if "--inject" in sys.argv:
        print()
        print(inject_report())
