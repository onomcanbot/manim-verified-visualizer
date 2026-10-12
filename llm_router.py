"""
================================================================================
llm_router.py · 자연어 요청 → 로컬 LLM(Ollama) → 검증된 템플릿 설정
================================================================================

LLM은 코드를 쓰지 않는다. "어느 템플릿을 어떤 설정값으로" JSON만 낸다.

  요청 ─▶ LLM(JSON) ─▶ ① 형식·범위 검사 ─▶ ② 계산 모듈 검증(*_core.run) ─▶ 통과
              ▲                │ 실패                 │ 실패
              └──── 오류 내용을 되돌려 스스로 고치게 함 (최대 3회) ────┘

  - LLM이 쓴 '설명(reason)'은 화면에 참고로만 보여 주며 검증 대상이 아니다.
    영상 속 숫자와 문장은 여전히 계산 결과에서만 만들어진다.
  - 모든 시도는 기록되어 논문용 지표(첫 시도 성공률, 평균 수정 횟수, 소요 시간)를 계산할 수 있다.

실행
  python llm_router.py "정답을 0으로 바꾼 역전파 보여줘"
  python llm_router.py --models                          # 설치된 모델 목록
  python llm_router.py --bench --model gemma4:26b        # 벤치마크 (논문용)
  python llm_router.py --bench --model gemma4:26b,hermes3:8b   # 여러 모델 연속 비교 + 비교표
  python llm_router.py --bench --model hermes3:8b --prompt v1,v2  # 프롬프트 개선 전후 비교 (개발셋+보류셋)
  옵션 : --set dev|holdout|all (기본 all)  --prompt v1|v2|v1,v2 (기본 v2)
  옵션 : --url http://127.0.0.1:11434  --model <이름>
================================================================================
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attention_core as at  # noqa: E402
import backprop_core as bp   # noqa: E402
import gan_core as gn        # noqa: E402
import gcn_core as gc        # noqa: E402

DEFAULT_URL = "http://127.0.0.1:11434"
MAX_REPAIR = 3
TEMPLATES = ["backprop", "attention", "gcn", "gan"]
CORES = {"backprop": bp, "attention": at, "gcn": gc, "gan": gn}

# ============================================================================
# 시스템 프롬프트 (템플릿 설명 + 설정 규칙 + 예시)
# ============================================================================
SYSTEM_PROMPT_V1 = """너는 공학 개념 시각화 도구의 '요청 해석기'다. 사용자의 한국어 요청을 읽고,
아래 4개 템플릿 중 하나와 그 설정값을 JSON 하나로만 답한다. 다른 글은 절대 쓰지 않는다.

출력 형식: {"template": "...", "config": {...}, "reason": "한국어 한 문장"}
- config에는 사용자가 바꾸길 원하는 항목만 넣는다. 언급이 없으면 빈 객체 {}.
- 템플릿에 맞지 않는 요청이면 {"template": "none", "config": {}, "reason": "지원하지 않는 이유"}.

[backprop] 역전파: 2-2-1 신경망(시그모이드, MSE)에서 기울기를 연쇄법칙으로 거꾸로 계산하고 1회 갱신
  x: [x1, x2] 각 -2~2 | y: 정답 0~1 | lr: 학습률 0.1~5
  params: {"w1","w2","w3","w4","w5","w6","b1","b2","b3"} 중 바꿀 것만, 각 -2~2
  (w1,w2→h1, w3,w4→h2, w5,w6→출력)

[attention] 셀프 어텐션: 단어 3개가 서로를 얼마나 참고하는지 (Q·K·V, softmax)
  tokens: 정확히 3개 단어 (긴 문장이면 핵심 단어 3개만) | axis_names: 임베딩 축 이름 2개
  X: 단어별 2차원 임베딩 [[a,b],[c,d],[e,f]] 각 -2~2 | W_Q, W_K, W_V: 2x2 행렬 각 -2~2
  focus: 자세히 볼 단어 번호 0~2 (0부터 시작)

[gcn] GCN 메시지 패싱: 노드 5개 그래프에서 이웃 정보를 모아 갱신, 층을 쌓으면 수용 영역 확대
  nodes: 정확히 5개 이름 | axis_names: 특징 축 이름 2개 | X: 노드별 2차원 특징 5개 각 -2~2
  W: 2x2 행렬 각 -2~2 | edges: [[i,j], ...] 노드 번호 0~4, i≠j | focus: 자세히 볼 노드 0~4

[gan] GAN 모드 붕괴: 8개 모드 데이터에서 기본 GAN과 PacGAN을 실제 학습시켜 비교
  seed: 0~9 정수 | steps: 1000~4000 정수 | lr: 0.0005~0.005 | std: 모드 폭 0.01~0.1
  pack: PacGAN 묶음 크기 2~4 정수

예시
요청: 정답이 0일 때 역전파가 어떻게 달라지는지 보여줘
{"template": "backprop", "config": {"y": 0.0}, "reason": "정답 y를 0으로 바꿔 기울기 부호 변화를 본다"}
요청: '나는 밥을 먹었다'에서 '먹었다'가 어디에 집중하는지
{"template": "attention", "config": {"tokens": ["나는", "밥을", "먹었다"], "focus": 2}, "reason": "세 단어 문장에서 동사의 어텐션을 본다"}
요청: 가운데 노드 하나에 나머지 넷이 연결된 별 모양 그래프
{"template": "gcn", "config": {"nodes": ["중심", "A", "B", "C", "D"], "edges": [[0,1],[0,2],[0,3],[0,4]], "focus": 0}, "reason": "별 그래프에서 허브로 메시지가 모이는 과정"}
요청: 모드 붕괴를 시드 3으로, 모드를 조금 넓게 해서
{"template": "gan", "config": {"seed": 3, "std": 0.05}, "reason": "시드 3, 모드 폭을 넓혀 붕괴 정도를 비교"}
요청: 트랜스포머 디코더 전체 구조를 그려줘
{"template": "none", "config": {}, "reason": "지원하는 템플릿은 역전파·셀프 어텐션·GCN·GAN 모드 붕괴뿐이다"}
"""

# v2 : hermes3:8b 개발셋 오류 분석(2026-10-09)에 따른 일반 규칙 추가
#   과잉 지정 → '언급한 것만' 강화 / 순서 착오 → 0부터 세는 대응표 / 형태소 분리 → 띄어쓰기 단위
#   개념 오해 → 그래프·행렬 이름(프리셋)을 코드가 펼침 / 생략 → 고립 노드는 edges 직접 명시
#   형식 오류 → params 예시
SYSTEM_PROMPT_V2 = """너는 공학 개념 시각화 도구의 '요청 해석기'다. 사용자의 한국어 요청을 읽고,
아래 4개 템플릿 중 하나와 그 설정값을 JSON 하나로만 답한다. 다른 글은 절대 쓰지 않는다.

출력 형식: {"template": "...", "config": {...}, "reason": "한국어 한 문장"}
- template은 "backprop", "attention", "gcn", "gan" 중 하나. 맞는 것이 없으면 "none".

반드시 지킬 규칙
1. config에는 요청에 직접 나온 항목만 넣는다. 요청에 없는 X, W, axis_names, nodes 등은 절대 넣지 않는다
   (넣지 않으면 잘 설계된 기본값이 쓰인다). 언급이 전혀 없으면 config는 {}.
2. 번호는 0부터 센다: 첫 번째 = 0, 두 번째 = 1, 세 번째 = 2, 네 번째 = 3, 다섯 번째 = 4.
   "3번 노드"처럼 1부터 센 번호를 말하면 1을 빼서 쓴다 (3번 노드 → 2).
3. 문장의 단어는 띄어쓰기 단위 그대로 쓰고 조사를 떼지 않는다.
   '고양이가 쥐를 잡았다' → ["고양이가", "쥐를", "잡았다"]  (["고양이", "가", ...] 는 틀림)
4. 숫자를 지어내지 말고 요청에 나온 값을 그대로 쓴다.
5. 행렬 이름: 단위행렬은 "identity" 라고 쓴다 (= [[1,0],[0,1]]). 그 밖의 행렬은 숫자로 쓴다.
6. 그래프 모양 이름은 edges에 문자열로 쓴다:
   "path" = 한 줄로 이어짐 (0-1-2-3-4) | "cycle" = 원형 (0-1-2-3-4-0)
   "star" = 0번 중심 별 모양 | "star:2" = 2번 중심 별 모양 | "complete" = 모두 서로 연결 | "empty" = 간선 없음
   이 밖의 모양(예: 고립 노드가 있는 그래프)은 [[i,j], ...]로 직접 쓰고, 고립시킬 노드는 어떤 간선에도 넣지 않는다.

[backprop] 역전파: 2-2-1 신경망(시그모이드, MSE)에서 기울기를 연쇄법칙으로 거꾸로 계산하고 1회 갱신
  x: [x1, x2] 각 -2~2 | y: 정답 0~1 | lr: 학습률 0.1~5
  params: 이름→값 객체. {"w1","w2","w3","w4","w5","w6","b1","b2","b3"} 중 바꿀 것만, 각 -2~2
  (w1,w2→h1, w3,w4→h2, w5,w6→출력)

[attention] 셀프 어텐션: 단어 3개가 서로를 얼마나 참고하는지 (Q·K·V, softmax)
  tokens: 정확히 3개 단어 (긴 문장이면 핵심 단어 3개만) | axis_names: 임베딩 축 이름 2개
  X: 단어별 2차원 임베딩 [[a,b],[c,d],[e,f]] 각 -2~2 | W_Q, W_K, W_V: 2x2 행렬 각 -2~2 또는 "identity"
  focus: 자세히 볼 단어 번호 0~2

[gcn] GCN 메시지 패싱: 노드 5개 그래프에서 이웃 정보를 모아 갱신, 층을 쌓으면 수용 영역 확대
  nodes: 정확히 5개 이름 | axis_names: 특징 축 이름 2개 | X: 노드별 2차원 특징 5개 각 -2~2
  W: 2x2 행렬 각 -2~2 또는 "identity" | edges: 그래프 모양 이름 또는 [[i,j], ...] (0~4, i≠j)
  focus: 자세히 볼 노드 0~4

[gan] GAN 모드 붕괴: 8개 모드 데이터에서 기본 GAN과 PacGAN을 실제 학습시켜 비교
  seed: 0~9 정수 | steps: 1000~4000 정수 | lr: 0.0005~0.005 | std: 모드 폭 0.01~0.1
  pack: PacGAN 묶음 크기 2~4 정수

예시
요청: 정답이 0일 때 역전파가 어떻게 달라지는지 보여줘
{"template": "backprop", "config": {"y": 0.0}, "reason": "정답 y를 0으로 바꿔 기울기 부호 변화를 본다"}
요청: w3을 1.5로 바꾸면 h2 쪽 기울기가 어떻게 되는지
{"template": "backprop", "config": {"params": {"w3": 1.5}}, "reason": "w3만 1.5로 바꿔 h2 경로를 본다"}
요청: '나는 밥을 먹었다'에서 '먹었다'가 어디에 집중하는지
{"template": "attention", "config": {"tokens": ["나는", "밥을", "먹었다"], "focus": 2}, "reason": "세 단어 문장에서 동사의 어텐션을 본다"}
요청: 가운데 노드 하나에 나머지 넷이 연결된 별 모양 그래프
{"template": "gcn", "config": {"edges": "star", "focus": 0}, "reason": "별 그래프에서 허브로 메시지가 모이는 과정"}
요청: 2번 노드만 혼자 떨어져 있는 그래프
{"template": "gcn", "config": {"edges": [[0,2],[2,3],[3,4]], "focus": 1}, "reason": "2번 노드(번호 1)를 간선에서 빼서 고립시킨다"}
요청: 모드 붕괴를 시드 3으로, 모드를 조금 넓게 해서
{"template": "gan", "config": {"seed": 3, "std": 0.05}, "reason": "시드 3, 모드 폭을 넓혀 붕괴 정도를 비교"}
요청: 트랜스포머 디코더 전체 구조를 그려줘
{"template": "none", "config": {}, "reason": "지원하는 템플릿은 역전파·셀프 어텐션·GCN·GAN 모드 붕괴뿐이다"}
"""

PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2}
DEFAULT_PROMPT = "v2"
SYSTEM_PROMPT = SYSTEM_PROMPT_V2          # 하위 호환

REPAIR_MSG = {
    "v1": "방금 답에 다음 문제가 있다. 같은 형식의 JSON만 고쳐서 다시 답하라.\n- ",
    "v2": ("방금 답에 다음 문제가 있다. 같은 형식의 JSON만 고쳐서 다시 답하라.\n"
           "형식만 고치고, 요청에 나온 단어·이름·값은 자르거나 바꾸지 마라. "
           "개수가 맞지 않으면 요청에서 핵심을 다시 골라라. 같은 답을 반복하지 마라.\n- "),
}

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "template": {"type": "string", "enum": TEMPLATES + ["none"]},
        "config": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["template", "config", "reason"],
}


# ============================================================================
# ① 형식·범위 검사
# ============================================================================
def _num(v, lo, hi, name, errs):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errs.append(f"{name}: 숫자여야 함 (받은 값 {v!r})")
        return None
    if not lo <= v <= hi:
        errs.append(f"{name}: {lo}~{hi} 범위여야 함 (받은 값 {v})")
        return None
    return float(v)


def _int(v, lo, hi, name, errs):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int):
        errs.append(f"{name}: 정수여야 함 (받은 값 {v!r})")
        return None
    if not lo <= v <= hi:
        errs.append(f"{name}: {lo}~{hi} 범위여야 함 (받은 값 {v})")
        return None
    return v


def _strs(v, n, name, errs, maxlen=8):
    if not isinstance(v, list) or len(v) != n or not all(isinstance(s, str) and s.strip() for s in v):
        errs.append(f"{name}: 비어 있지 않은 문자열 정확히 {n}개의 목록이어야 함 (받은 값 {v!r})")
        return None
    long = [s for s in v if len(s.strip()) > maxlen]
    if long:
        errs.append(f"{name}: 각 이름은 {maxlen}자 이하 (너무 긴 값 {long})")
        return None
    return [s.strip() for s in v]


IDENTITY_NAMES = {"identity", "eye", "i", "단위행렬"}


def _mat(v, r, c, lo, hi, name, errs):
    if isinstance(v, str) and v.strip().lower() in IDENTITY_NAMES and r == c:    # 프리셋: 단위행렬
        return [[1.0 if i == j else 0.0 for j in range(c)] for i in range(r)]
    ok = isinstance(v, list) and len(v) == r and all(isinstance(row, list) and len(row) == c for row in v)
    if not ok:
        errs.append(f"{name}: {r}x{c} 숫자 행렬이어야 함 (받은 값 {v!r})")
        return None
    out = [[_num(x, lo, hi, f"{name}[{i}][{j}]", errs) for j, x in enumerate(row)] for i, row in enumerate(v)]
    return None if any(x is None for row in out for x in row) else out


GRAPH_PRESETS = "path, cycle, star, star:<중심 번호>, complete, empty"


def graph_preset(name: str, n: int = 5):
    """그래프 모양 이름 → 간선 목록 (결정론적으로 펼침). 모르는 이름이면 None."""
    name = name.strip().lower()
    if name == "path":
        return [[i, i + 1] for i in range(n - 1)]
    if name == "cycle":
        return [[i, (i + 1) % n] for i in range(n)]
    if name == "complete":
        return [[i, j] for i in range(n) for j in range(i + 1, n)]
    if name == "empty":
        return []
    if name == "star" or name.startswith("star:"):
        try:
            c = int(name.split(":")[1]) if ":" in name else 0
        except ValueError:
            return None
        return [[min(c, j), max(c, j)] for j in range(n) if j != c] if 0 <= c < n else None
    return None


def validate(template: str, cfg) -> tuple[dict, list[str]]:
    errs: list[str] = []
    if template not in TEMPLATES:
        return {}, [f"template: {TEMPLATES} 중 하나여야 함 (받은 값 {template!r})"]
    if not isinstance(cfg, dict):
        return {}, [f"config: 객체여야 함 (받은 값 {cfg!r})"]
    out = {}
    allowed = {
        "backprop": {"x", "y", "lr", "params"},
        "attention": {"tokens", "axis_names", "X", "W_Q", "W_K", "W_V", "focus"},
        "gcn": {"nodes", "axis_names", "X", "W", "edges", "focus"},
        "gan": {"seed", "steps", "lr", "std", "pack"},
    }[template]
    unknown = set(cfg) - allowed
    if unknown:
        hint = ""
        if template == "backprop" and unknown & {"w1", "w2", "w3", "w4", "w5", "w6", "b1", "b2", "b3"}:
            hint = ' — 가중치·편향은 params 안에 넣어야 함. 예: {"params": {"w5": 0.5}}'
        errs.append(f"config에 이 템플릿이 모르는 항목이 있음: {sorted(unknown)} (허용: {sorted(allowed)}){hint}")

    if template == "backprop":
        if "x" in cfg:
            v = cfg["x"]
            if isinstance(v, list) and len(v) == 2:
                out["x"] = [_num(a, -2, 2, f"x[{i}]", errs) for i, a in enumerate(v)]
            else:
                errs.append(f"x: 숫자 2개 목록이어야 함 (받은 값 {v!r})")
        if "y" in cfg:
            out["y"] = _num(cfg["y"], 0, 1, "y", errs)
        if "lr" in cfg:
            out["lr"] = _num(cfg["lr"], 0.1, 5, "lr", errs)
        if "params" in cfg:
            p = cfg["params"]
            keys = ["w1", "w2", "w3", "w4", "w5", "w6", "b1", "b2", "b3"]
            if not isinstance(p, dict):
                errs.append(f'params: 이름→값 객체여야 함. 예: {{"w5": 0.5, "w6": 0.5}} (받은 값 {p!r})')
            elif set(p) - set(keys):
                errs.append(f"params: 이름은 {keys} 중에서만 (모르는 이름 {sorted(set(p) - set(keys))})")
            else:
                out["params"] = {k: _num(v, -2, 2, f"params.{k}", errs) for k, v in p.items()}
    elif template == "attention":
        if "tokens" in cfg:
            out["tokens"] = _strs(cfg["tokens"], 3, "tokens", errs)
        if "axis_names" in cfg:
            out["axis_names"] = _strs(cfg["axis_names"], 2, "axis_names", errs)
        if "X" in cfg:
            out["X"] = _mat(cfg["X"], 3, 2, -2, 2, "X", errs)
        for k in ("W_Q", "W_K", "W_V"):
            if k in cfg:
                out[k] = _mat(cfg[k], 2, 2, -2, 2, k, errs)
        if "focus" in cfg:
            out["focus"] = _int(cfg["focus"], 0, 2, "focus", errs)
    elif template == "gcn":
        if "nodes" in cfg:
            out["nodes"] = _strs(cfg["nodes"], 5, "nodes", errs)
        if "axis_names" in cfg:
            out["axis_names"] = _strs(cfg["axis_names"], 2, "axis_names", errs)
        if "X" in cfg:
            out["X"] = _mat(cfg["X"], 5, 2, -2, 2, "X", errs)
        if "W" in cfg:
            out["W"] = _mat(cfg["W"], 2, 2, -2, 2, "W", errs)
        if "edges" in cfg and isinstance(cfg["edges"], str):
            e = graph_preset(cfg["edges"])
            if e is None:
                errs.append(f"edges: 그래프 모양 이름은 {GRAPH_PRESETS} 중 하나 (받은 값 {cfg['edges']!r})")
            else:
                out["edges"] = e
        elif "edges" in cfg:
            e = cfg["edges"]
            good = isinstance(e, list) and all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) and not isinstance(i, bool)
                                                            and 0 <= i <= 4 for i in p) and p[0] != p[1]
                for p in e)
            if not good:
                errs.append(f"edges: [[i,j], ...] 형식, i·j는 0~4 정수이고 i≠j (받은 값 {e!r})")
            else:
                out["edges"] = e
        if "focus" in cfg:
            out["focus"] = _int(cfg["focus"], 0, 4, "focus", errs)
    elif template == "gan":
        if "seed" in cfg:
            out["seed"] = _int(cfg["seed"], 0, 9, "seed", errs)
        if "steps" in cfg:
            out["steps"] = _int(cfg["steps"], 1000, 4000, "steps", errs)
        if "lr" in cfg:
            out["lr"] = _num(cfg["lr"], 0.0005, 0.005, "lr", errs)
        if "std" in cfg:
            out["std"] = _num(cfg["std"], 0.01, 0.1, "std", errs)
        if "pack" in cfg:
            out["pack"] = _int(cfg["pack"], 2, 4, "pack", errs)
    return out, errs


# ============================================================================
# ①-b 요청 근거 검사 (v2) : 요청 문장에서 결정론적으로 확인할 수 있는 것만
# ============================================================================
QUOTE_RE = re.compile(r"['‘\"“]([^'’\"”]+)['’\"”]")


def grounding_check(request: str, template: str, cfg: dict) -> list[str]:
    """요청에 따옴표로 묶인 세 단어 문장이 있으면 tokens는 그 띄어쓰기 단어와 같아야 한다.
    (수정 과정에서 단어를 쪼개거나 잘라 개수만 맞추는 '의도 이탈'을 막는다)"""
    errs = []
    if template == "attention" and "tokens" in cfg and cfg["tokens"]:
        for q in QUOTE_RE.findall(request):
            words = q.split()
            if len(words) == 3 and cfg["tokens"] != words:
                errs.append(f"tokens: 요청의 문장 '{q}'를 띄어쓰기 그대로 써야 함 → {words} "
                            f"(받은 값 {cfg['tokens']})")
                break
    return errs


# ============================================================================
# ② 계산 모듈 검증 (GAN은 학습이 포함되어 무거우므로 영상 만들기 단계에서 실행)
# ============================================================================
def core_check(template: str, cfg: dict) -> list[str]:
    core = CORES[template]
    try:
        if template == "gan":
            core.load_config(cfg)
            return []
        r = core.run(cfg)
    except Exception as e:                       # 설정 구조 오류 등
        return [f"계산 모듈이 설정을 거부함: {e}"]
    return [] if r["passed"] else ["계산 모듈 검증 실패 (수치 대조 또는 불변식 불일치)"]


# ============================================================================
# Ollama 호출
# ============================================================================
def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(url: str = DEFAULT_URL, timeout: float = 5) -> list[str]:
    with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return sorted(m["name"] for m in data.get("models", []))


def is_cloud(name: str) -> bool:
    """Ollama 클라우드 모델 (요청이 PC 밖 서버로 전송됨)."""
    return "cloud" in name


def pick_default(models: list[str]) -> str | None:
    """로컬 모델 우선. 사용자 정의 모델(gemma4-kw 등)보다 공식 태그(gemma4:26b 등)를 우선."""
    local = [m for m in models if not is_cloud(m)]
    for key in ("gemma4:26b", "gemma4:", "qwen3", "gemma", "qwen"):
        for m in local:
            if key in m:
                return m
    return local[0] if local else (models[0] if models else None)


def chat(url, model, messages, timeout=300):
    """구조화 출력(JSON 스키마) → 안 되면 'json' 모드로, think 옵션 미지원이면 빼고 재시도."""
    base = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0, "seed": 42}}
    attempts = [dict(base, format=FORMAT_SCHEMA, think=False), dict(base, format=FORMAT_SCHEMA),
                dict(base, format="json")]
    last = None
    for payload in attempts:
        try:
            data = _post(url.rstrip("/") + "/api/chat", payload, timeout)
            text = data.get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except urllib.error.HTTPError as e:     # 옵션 미지원 등 → 다음 방식으로
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
    raise RuntimeError(f"Ollama 호출 실패: {last}")


def parse_json(text: str):
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)      # 앞뒤에 군말이 붙은 경우
        if m:
            try:
                return json.loads(m.group(0)), None
            except json.JSONDecodeError as e:
                return None, f"JSON 해석 실패: {e}"
        return None, "JSON 객체를 찾지 못함"


# ============================================================================
# 해석 + 자동 수정 루프
# ============================================================================
def route(request: str, model: str, url: str = DEFAULT_URL, max_repair: int = MAX_REPAIR,
          prompt: str = DEFAULT_PROMPT) -> dict:
    """prompt='v1' : 처음 버전 (논문의 기준선) / 'v2' : 오류 분석 후 개선 버전."""
    messages = [{"role": "system", "content": PROMPTS[prompt]},
                {"role": "user", "content": f"요청: {request}"}]
    log, t_all = [], time.time()
    for k in range(max_repair + 1):
        t0 = time.time()
        raw = chat(url, model, messages)
        obj, perr = parse_json(raw)
        errs = [perr] if perr else []
        template, cfg, reason = None, {}, ""
        if obj is not None:
            if not isinstance(obj, dict):
                errs.append("최상위는 JSON 객체여야 함")
            else:
                template, reason = obj.get("template"), str(obj.get("reason", ""))
                if template == "none":
                    log.append(dict(attempt=k, raw=raw, errors=[], seconds=time.time() - t0))
                    return dict(ok=False, unsupported=True, template="none", config={}, reason=reason,
                                attempts=log, repairs=k, seconds=time.time() - t_all, model=model,
                                prompt=prompt)
                cfg, verrs = validate(template, obj.get("config", {}))
                errs += verrs
                if not errs and prompt != "v1":
                    errs += grounding_check(request, template, cfg)
                if not errs:
                    errs += core_check(template, cfg)
        log.append(dict(attempt=k, raw=raw, errors=errs, seconds=time.time() - t0))
        if not errs:
            return dict(ok=True, unsupported=False, template=template, config=cfg, reason=reason,
                        attempts=log, repairs=k, seconds=time.time() - t_all, model=model, prompt=prompt)
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": REPAIR_MSG[prompt] + "\n- ".join(errs)}]
    return dict(ok=False, unsupported=False, template=template, config=cfg, reason=reason,
                attempts=log, repairs=max_repair, seconds=time.time() - t_all, model=model, prompt=prompt)


# ============================================================================
# 벤치마크 (논문용 : 모델별 정확도 · 첫 시도 성공률 · 평균 수정 횟수 · 시간)
# ============================================================================
def merged(template: str, cfg: dict) -> dict:
    """LLM이 바꾼 항목 + 기본값 = 실제로 영상에 쓰일 설정."""
    return CORES[template].load_config(cfg)


def _deg(c):
    d = [0] * len(c["nodes"])
    for i, j in c["edges"]:
        d[i] += 1
        d[j] += 1
    return d


def _is_path(c):
    d, e = _deg(c), c["edges"]
    if len(e) != len(c["nodes"]) - 1 or max(d) > 2:
        return False
    seen, stack = {0}, [0]                          # 연결 여부
    while stack:
        u = stack.pop()
        for i, j in e:
            for a, b in ((i, j), (j, i)):
                if a == u and b not in seen:
                    seen.add(b)
                    stack.append(b)
    return len(seen) == len(c["nodes"])


# (요청, 기대 템플릿, 의도 검사) — 의도 검사는 실제 쓰일 설정(기본값 병합 후)에 대해 수행.
# None = 이 요청은 특정 값을 요구하지 않음.
BENCH = [
    ("정답을 0으로 바꾼 역전파 과정을 보여줘", "backprop",
     lambda c: c["y"] == 0),
    ("학습률을 3으로 크게 했을 때 가중치 갱신이 어떻게 되는지", "backprop",
     lambda c: c["lr"] == 3),
    ("w5와 w6을 둘 다 양수로 해서 은닉층 기울기 부호를 비교하고 싶어", "backprop",
     lambda c: c["params"]["w5"] > 0 and c["params"]["w6"] > 0),
    ("입력을 x1=-1, x2=2로 바꿔서 역전파를 보여줘", "backprop",
     lambda c: c["x"] == [-1.0, 2.0]),
    ("'학생이 인공지능을 배운다'에서 '배운다'가 어디를 보는지", "attention",
     lambda c: c["tokens"] == ["학생이", "인공지능을", "배운다"] and c["focus"] == 2),
    ("어텐션에서 질문 행렬을 단위행렬로 하면 패턴이 어떻게 바뀌어?", "attention",
     lambda c: c["W_Q"].tolist() == [[1.0, 0.0], [0.0, 1.0]]),
    ("'고양이가 쥐를 잡았다' 문장의 셀프 어텐션, 두 번째 단어 중심으로", "attention",
     lambda c: c["tokens"] == ["고양이가", "쥐를", "잡았다"] and c["focus"] == 1),
    ("Q, K, V가 뭔지 세 단어 문장으로 보여줘", "attention", None),
    ("다섯 명이 한 줄로 손잡고 있는 경로 그래프에서 GCN", "gcn", _is_path),
    ("노드 하나가 아무와도 연결되지 않은 그래프에서 메시지 패싱", "gcn",
     lambda c: _deg(c).count(0) == 1),
    ("완전 그래프에서 GCN을 하면 특징이 어떻게 섞이는지", "gcn",
     lambda c: len(c["edges"]) == 10),
    ("민수, 지영, 현우, 서연, 도윤의 친구 관계 그래프에서 도윤이 받는 메시지", "gcn",
     lambda c: sorted(c["nodes"]) == sorted(["민수", "지영", "현우", "서연", "도윤"])
     and c["nodes"][c["focus"]] == "도윤"),
    ("GAN 모드 붕괴를 보여줘", "gan", None),
    ("PacGAN 묶음을 4로 늘리면 모드 붕괴가 줄어드는지", "gan",
     lambda c: c["pack"] == 4),
    ("시드 7로 GAN 학습을 2000단계만 해서 비교해줘", "gan",
     lambda c: c["seed"] == 7 and c["steps"] == 2000),
    ("모드 폭을 넓게 하면 붕괴가 약해지는지 GAN으로", "gan",
     lambda c: c["std"] > gn.DEFAULT["std"]),
    ("CNN 합성곱 과정을 시각화해줘", "none", None),
    ("오늘 서울 날씨 알려줘", "none", None),
]


def _edgeset(c):
    return {tuple(sorted(e)) for e in c["edges"]}


# 보류(held-out) 세트 — v2 개선 작업에 쓰지 않은 문항. 개선 효과의 일반화는 이 세트로 주장한다.
# (v2 규칙을 정한 뒤, v2를 한 번도 돌려 보기 전에 작성. 일부 문항은 v2의 프리셋 이름과 겹치는
#  개념을 다른 표현으로 묻고, 일부는 프리셋과 무관한 값·순서·문장 처리를 묻는다)
HOLDOUT = [
    ("타깃 값을 0.2로 두고 역전파 계산 과정을 보고 싶어", "backprop",
     lambda c: c["y"] == 0.2),
    ("학습률 0.5에서 한 번 업데이트하면 손실이 얼마나 줄어?", "backprop",
     lambda c: c["lr"] == 0.5),
    ("출력층 편향 b3를 -1로 바꾸면 기울기가 어떻게 변하는지", "backprop",
     lambda c: c["params"]["b3"] == -1),
    ("w1을 음수로 만들어서 역전파 해줘", "backprop",
     lambda c: c["params"]["w1"] < 0),
    ("'나는 커피를 마신다' 문장에서 첫 번째 단어의 어텐션", "attention",
     lambda c: c["tokens"] == ["나는", "커피를", "마신다"] and c["focus"] == 0),
    ("키 가중치 W_K를 [[0,1],[1,0]]으로 바꿔서 셀프 어텐션을 보여줘", "attention",
     lambda c: c["W_K"].tolist() == [[0.0, 1.0], [1.0, 0.0]]),
    ("'비가 와서 우산을' 세 단어로 셀프 어텐션, 마지막 단어 기준", "attention",
     lambda c: c["tokens"] == ["비가", "와서", "우산을"] and c["focus"] == 2),
    ("'바람이 나무를 흔든다'에서 가운데 단어가 다른 단어를 얼마나 보는지", "attention",
     lambda c: c["tokens"] == ["바람이", "나무를", "흔든다"] and c["focus"] == 1),
    ("3번 노드가 허브인 별 모양 그래프에서 메시지 패싱", "gcn",
     lambda c: _edgeset(c) == {(0, 2), (1, 2), (2, 3), (2, 4)}),
    ("다섯 노드가 원형으로 이어진 사이클 그래프에서 GCN", "gcn",
     lambda c: len(c["edges"]) == 5 and all(d == 2 for d in _deg(c))),
    ("간선이 하나도 없는 그래프에서 GCN을 하면 어떻게 되는지", "gcn",
     lambda c: len(c["edges"]) == 0),
    ("노드 이름을 가 나 다 라 마로 하고 세 번째 노드에 집중해서 GCN", "gcn",
     lambda c: c["nodes"] == ["가", "나", "다", "라", "마"] and c["focus"] == 2),
    ("1·2·3번 노드끼리 삼각형, 4·5번 노드는 둘만 연결된 두 무리 그래프", "gcn",
     lambda c: _edgeset(c) == {(0, 1), (0, 2), (1, 2), (3, 4)}),
    ("PacGAN 묶음 크기를 3으로 해서 모드 붕괴 비교", "gan",
     lambda c: c["pack"] == 3),
    ("학습률 0.001로 GAN 모드 붕괴 실험", "gan",
     lambda c: c["lr"] == 0.001),
    ("1500단계만 짧게 학습한 GAN 비교", "gan",
     lambda c: c["steps"] == 1500),
    ("모드를 아주 좁게, 표준편차 0.01로 해서 붕괴를 더 심하게", "gan",
     lambda c: c["std"] == 0.01),
    ("시드 0으로 모드 붕괴 보여줘", "gan",
     lambda c: c["seed"] == 0),
    ("RNN이 시퀀스를 처리하는 과정 보여줘", "none", None),
    ("파이썬으로 웹 크롤러 만들어줘", "none", None),
]
SETS = {"dev": BENCH, "holdout": HOLDOUT}


def bench_iter(model: str, url: str = DEFAULT_URL, set_name: str = "dev", prompt: str = DEFAULT_PROMPT):
    """문항마다 ('row', 결과)를 내보내고 마지막에 ('summary', 요약)을 내보낸다.

    지표
      routing   : 템플릿을 맞게 고르고 검증을 통과 (none 문항은 거절)
      intent    : routing + 실제 쓰일 설정이 요청의 구체적 값과 일치
                  (검증기는 '계산이 맞는지'를 보장하지만 '요청과 같은지'는 보장하지 못한다 → 따로 측정)
      first_try : 수정 없이 첫 응답으로 통과
    """
    rows = []
    for text, expect, check in SETS[set_name]:
        try:
            r = route(text, model, url, prompt=prompt)
            got = r["template"]
            routing = (got == expect) and (r["ok"] or expect == "none")
            intent, note = routing, ""
            if routing and check is not None:
                try:
                    intent = bool(check(merged(got, r["config"])))
                except Exception as e:
                    intent, note = False, f"의도 검사 오류: {e}"
            row = dict(request=text, expect=expect, got=got, routing=routing, intent=intent,
                       first_try=(r["repairs"] == 0) and (r["ok"] or r["unsupported"]),
                       repairs=r["repairs"], seconds=r["seconds"], ok=r["ok"],
                       config=r["config"], note=note,
                       attempts=[dict(raw=a["raw"], errors=a["errors"]) for a in r["attempts"]])
        except Exception as e:
            row = dict(request=text, expect=expect, got=f"ERROR {e}", routing=False, intent=False,
                       first_try=False, repairs=None, seconds=None, ok=False, config={}, note="",
                       attempts=[])
        rows.append(row)
        yield "row", row
    n = len(rows)
    done = [r for r in rows if r["seconds"] is not None]
    summ = dict(model=model, set=set_name, prompt=prompt, n=n,
                routing=sum(r["routing"] for r in rows) / n,
                intent=sum(r["intent"] for r in rows) / n,
                first_try=sum(r["first_try"] for r in rows) / n,
                mean_repairs=(sum(r["repairs"] for r in done) / len(done)) if done else None,
                mean_seconds=(sum(r["seconds"] for r in done) / len(done)) if done else None)
    out = HERE / "web_runs" / (f"bench_{model.replace(':', '_').replace('/', '_')}_{set_name}_{prompt}_"
                               f"{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(summary=summ, rows=rows), ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    summ["saved"] = str(out)
    yield "summary", summ


def row_line(k, v, n=None):
    mark = "O" if v["intent"] else ("△" if v["routing"] else "X")
    return (f"  [{k:2d}/{n or len(BENCH)}] {mark} 기대={v['expect']:<9} 결과={str(v['got'])[:20]:<9} "
            f"수정={v['repairs']}  {v['request']}")


def failure_lines(rows):
    """의도 불일치·실패 문항의 마지막 시도 원문과 문제를 보여 준다 (오류 유형 분석용)."""
    out = []
    for v in rows:
        if v["intent"]:
            continue
        out.append(f"\n  ▶ {v['request']}")
        out.append(f"    최종 설정: {json.dumps(v['config'], ensure_ascii=False)}")
        for k, a in enumerate(v["attempts"]):
            raw = a["raw"].replace("\n", " ")
            out.append(f"    시도 {k}: {raw[:160]}{'…' if len(raw) > 160 else ''}")
            if a["errors"]:
                out.append(f"           문제: {' / '.join(a['errors'])[:200]}")
        if v["routing"] and not v["intent"]:
            out.append("    → 검증은 통과했지만 요청한 값과 다름 (의도 불일치)" + (f" · {v['note']}" if v["note"] else ""))
    return out


def bench(model: str, url: str = DEFAULT_URL, set_name: str = "dev", prompt: str = DEFAULT_PROMPT) -> dict:
    k, rows = 0, []
    for kind, v in bench_iter(model, url, set_name, prompt):
        if kind == "row":
            k += 1
            rows.append(v)
            print(row_line(k, v, len(SETS[set_name])))
        else:
            fl = failure_lines(rows)
            if fl:
                print("\n  [틀린 문항 상세]" + "\n".join(fl))
            return v


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    url = _arg("--url", DEFAULT_URL)
    if "--models" in sys.argv:
        print("\n".join(list_models(url)))
        sys.exit()
    model = _arg("--model") or pick_default(list_models(url))
    if "--bench" in sys.argv:
        f = lambda v, p: "-" if v is None else format(v, p)
        sets = ["dev", "holdout"] if _arg("--set", "all") == "all" else [_arg("--set")]
        prompts = [p.strip() for p in _arg("--prompt", DEFAULT_PROMPT).split(",")]
        table = []
        for m in [x.strip() for x in (model or "").split(",") if x.strip()]:
            for p in prompts:
                for st in sets:
                    print(f"\n벤치마크: {m} · 세트 {st} ({len(SETS[st])}문항) · 프롬프트 {p}"
                          + ("  ※ 클라우드 모델" if is_cloud(m) else ""))
                    table.append(bench(m, url, st, p))
                    print(f"  → 저장: {table[-1]['saved']}")
        print("\n  O = 의도까지 일치 · △ = 템플릿·검증은 통과했지만 요청한 값과 다름 · X = 실패")
        print(f"\n{'모델':<18} {'프롬프트':>5} {'세트':>8} {'라우팅':>6} {'의도 일치':>8} {'첫 시도':>7} "
              f"{'평균 수정':>8} {'평균 초':>7}")
        for t in table:
            print(f"{t['model']:<18} {t['prompt']:>5} {t['set']:>8} {t['routing']:>6.0%} {t['intent']:>8.0%} "
                  f"{t['first_try']:>7.0%} {f(t['mean_repairs'], '.2f'):>8} {f(t['mean_seconds'], '.1f'):>7}")
        if all(t["mean_seconds"] is None for t in table):
            print("모든 문항이 실패했습니다. Ollama가 켜져 있는지, 주소(--url)와 모델 이름(--model)을 확인하세요.")
        sys.exit()
    words, skip = [], False
    for a in sys.argv[1:]:                       # 옵션과 옵션 값은 요청 문장에서 뺀다
        if skip:
            skip = False
        elif a in ("--url", "--model", "--prompt", "--set"):
            skip = True
        elif not a.startswith("--"):
            words.append(a)
    req = " ".join(words)
    r = route(req, model, url, prompt=_arg("--prompt", DEFAULT_PROMPT))
    print(json.dumps({k: r[k] for k in ("ok", "template", "config", "reason", "repairs")},
                     ensure_ascii=False, indent=2))
    for a in r["attempts"]:
        print(f"  시도 {a['attempt']}: {'통과' if not a['errors'] else ' / '.join(a['errors'])}")
