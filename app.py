"""
================================================================================
app.py · 검증형 Manim 시각화 생성기 — 웹 화면
================================================================================

코드를 몰라도 슬라이더로 값을 바꾸고, 검증을 통과한 경우에만 영상을 만든다.

  [슬라이더] → 설정(JSON) → *_core.py 검증 ─(통과)→ manim 별도 프로세스 렌더링 → 영상
                                           └(실패)→ 생성 차단 + 사유 표시

검증 관문은 세 곳에 있다
  1) 웹 화면  : '영상 만들기'를 눌러도 먼저 검증을 다시 돌린다
  2) 서버     : 검증 결과가 FAIL이면 manim을 아예 실행하지 않는다
  3) Manim 씬 : 씬 내부에서도 검증을 다시 하고, 실패하면 렌더링을 중단한다

실행 (모든 파일을 같은 폴더에)
  conda activate 3Blue1Brown
  pip install gradio                 # 최초 1회
  python app.py                      # 브라우저에서 http://127.0.0.1:7860

필요 파일 : app.py, backprop_core.py, backprop_demo.py, attention_core.py, attention_demo.py,
            gcn_core.py, gcn_demo.py, gan_core.py, gan_demo.py, llm_router.py
결과 저장 : 같은 폴더의 web_runs/ (실행마다 설정 JSON + 영상 + 로그를 남김 → 재현 가능)
================================================================================
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attention_core as at  # noqa: E402
import backprop_core as bp   # noqa: E402
import gcn_core as gc        # noqa: E402
import gan_core as gn        # noqa: E402
import llm_router as lr      # noqa: E402

# 같은 폴더에 예전 버전 *_core.py 가 남아 있으면 웹 화면용 설정 구조가 없다
_old = [m.__name__ + ".py" for m in (bp, at, gc, gn) if not hasattr(m, "DEFAULT")]
if _old:
    sys.exit("[안내] 예전 버전 파일이 섞여 있습니다: " + ", ".join(_old) + "\n"
             f"  위치: {HERE}\n"
             "  manim_web_demo.zip 의 파일을 모두 같은 폴더에 두고 다시 실행하세요.")

WORK = HERE / "web_runs"
WORK.mkdir(exist_ok=True)

DEMOS = {
    "backprop":  dict(core=bp, file="backprop_demo.py",  scene="BackpropChainRule", label="역전파"),
    "attention": dict(core=at, file="attention_demo.py", scene="SelfAttention",     label="셀프 어텐션"),
    "gcn":       dict(core=gc, file="gcn_demo.py",       scene="GCNMessagePassing", label="GCN 메시지 패싱"),
    "gan":       dict(core=gn, file="gan_demo.py",       scene="GANModeCollapse",   label="GAN 모드 붕괴"),
}
QUALITY = {"480p 미리보기": "l", "720p 보통": "m", "1080p 고화질": "h"}
RENDER_TIMEOUT = 1800   # 초


# ============================================================================
# 공통 : 검증 상태 표시 / 렌더링
# ============================================================================
def status_md(passed: bool, detail: str) -> str:
    if passed:
        return f"### ✅ 검증 통과\n{detail}\n\n이제 **영상 만들기**를 누를 수 있습니다."
    return f"### ❌ 검증 실패 — 영상 생성 차단\n{detail}"


def render(demo_key: str, cfg: dict, quality_label: str):
    """generator : (상태, 영상 경로, 로그) 를 단계별로 내보낸다."""
    d = DEMOS[demo_key]
    core = d["core"]

    # --- 관문 2 : 서버에서 검증을 다시 돌린다 ---
    yield ("### ⏳ 검증 중..." + ("  \n(GAN은 실제 학습과 재학습을 포함해 수십 초 걸립니다)"
                                  if demo_key == "gan" else "")), None, ""
    try:
        r = core.run(cfg)
    except Exception as e:                                   # 잘못된 설정 등
        yield status_md(False, f"설정 오류: `{e}`"), None, ""
        return
    if not r["passed"]:
        yield status_md(False, "검증을 통과하지 못해 렌더링을 시작하지 않았습니다."), None, ""
        return

    # --- 실행마다 별도 폴더 : 설정·영상·로그를 함께 보관 (재현성) ---
    run_dir = WORK / f"{demo_key}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env[core.CONFIG_ENV] = str(cfg_path)
    env["PYTHONIOENCODING"] = "utf-8"       # Windows cp949 로그 깨짐 방지
    env["PYTHONUTF8"] = "1"
    q = QUALITY.get(quality_label, "l")
    cmd = [sys.executable, "-m", "manim", f"-q{q}", "--disable_caching",
           "--media_dir", str(run_dir / "media"), str(HERE / d["file"]), d["scene"]]

    yield (f"### ⏳ 렌더링 중... ({quality_label})\n검증은 통과했습니다. "
           "영상이 완성될 때까지 이 창을 닫지 마세요."), None, " ".join(cmd)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=RENDER_TIMEOUT)
    except subprocess.TimeoutExpired:
        yield "### ❌ 시간 초과 — 더 낮은 화질로 다시 시도해 보세요.", None, ""
        return
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (run_dir / "render.log").write_text(log, encoding="utf-8")

    videos = sorted(run_dir.glob(f"media/videos/**/{d['scene']}.mp4"))
    if proc.returncode != 0 or not videos:
        yield ("### ❌ 렌더링 실패\n아래 로그의 마지막 부분을 확인하세요. "
               "(관문 3: 씬 내부 검증 실패일 수도 있습니다)"), None, log[-4000:]
        return
    yield (f"### ✅ 완성 ({time.time() - t0:.0f}초)\n"
           f"설정과 로그는 `{run_dir.relative_to(HERE)}` 폴더에 함께 저장했습니다."), \
        str(videos[-1]), log[-2000:]


# ============================================================================
# 역전파 탭
# ============================================================================
BP_KEYS = ["x1", "x2", "y", "lr", "w1", "w2", "w3", "w4", "w5", "w6", "b1", "b2", "b3"]


def bp_defaults():
    D = bp.DEFAULT
    return [D["x"][0], D["x"][1], D["y"], D["lr"]] + [D["params"][k] for k in BP_KEYS[4:]]


def bp_cfg(*v):
    v = [round(float(a), 2) for a in v]
    return {"x": v[0:2], "y": v[2], "lr": v[3],
            "params": dict(zip(BP_KEYS[4:], v[4:]))}


def bp_verify(*v):
    try:
        r = bp.run(bp_cfg(*v))
    except Exception as e:
        return status_md(False, f"설정 오류: `{e}`"), ""
    detail = (f"수치미분 대조 최대 오차: 현재 설정 **{r['err_demo']:.1e}** · "
              f"무작위 {r['n_random']}세트 포함 최악 {r['err_num']:.1e} (허용 {bp.TOL:.0e})  \n"
              f"손실 L: {r['fwd']['loss']:.4f} → 1회 갱신 후 {r['fwd_new']['loss']:.4f}")
    return status_md(r["passed"], detail), bp.report(r)


def bp_inject(*v):
    return bp.inject_report(bp_cfg(*v))


def bp_render(quality, *v):
    yield from render("backprop", bp_cfg(*v), quality)


# ============================================================================
# 어텐션 탭
# ============================================================================
def at_defaults():
    D = at.DEFAULT
    flat = lambda M: [M[i][j] for i in range(len(M)) for j in range(len(M[0]))]
    return (list(D["tokens"]) + list(D["axis_names"]) + flat(D["X"]) + flat(D["W_Q"])
            + flat(D["W_K"]) + flat(D["W_V"]) + [D["focus"]])


def at_cfg(*v):
    v = list(v)
    tokens = [str(t).strip() or f"단어{i + 1}" for i, t in enumerate(v[0:3])]
    axes = [str(a).strip() or f"축{i + 1}" for i, a in enumerate(v[3:5])]
    nums = [round(float(a), 2) for a in v[5:23]]
    mat = lambda a, r, c: [nums[a + i * c: a + (i + 1) * c] for i in range(r)]
    return {"tokens": tokens, "axis_names": axes,
            "X": mat(0, 3, 2), "W_Q": mat(6, 2, 2), "W_K": mat(10, 2, 2), "W_V": mat(14, 2, 2),
            "focus": int(v[23])}


def at_verify(*v):
    try:
        cfg = at_cfg(*v)
        r = at.run(cfg)
    except Exception as e:
        return status_md(False, f"설정 오류: `{e}`"), ""
    f = r["focus"]
    best = int(r["A"][f].argmax())
    detail = (f"독립 구현 대조 최악 오차 **{r['max_ref']:.1e}**, 행 합 오차 {r['row_sum_err']:.1e} "
              f"(데모 값 + 무작위 {r['n_random']}세트, 허용 {at.TOL:.0e})  \n"
              f"'{r['tokens'][f]}' → '{r['tokens'][best]}' 가중치가 가장 큼 ({r['A'][f][best]:.2f})")
    return status_md(r["passed"], detail), at.report(r)


def at_inject(*v):
    return at.inject_report(at_cfg(*v))


def at_render(quality, *v):
    yield from render("attention", at_cfg(*v), quality)


# ============================================================================
# GCN 탭
# ============================================================================
GC_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
GC_PAIR_LABELS = [f"{i + 1}–{j + 1}" for i, j in GC_PAIRS]


def gc_defaults():
    D = gc.DEFAULT
    flat = lambda M: [M[i][j] for i in range(len(M)) for j in range(len(M[0]))]
    edges = [GC_PAIR_LABELS[GC_PAIRS.index(tuple(sorted(e)))] for e in D["edges"]]
    return (list(D["nodes"]) + list(D["axis_names"]) + flat(D["X"]) + flat(D["W"])
            + [edges, f"{D['focus'] + 1}번 노드"])


def gc_cfg(*v):
    v = list(v)
    nodes = [str(t).strip() or f"N{i + 1}" for i, t in enumerate(v[0:5])]
    axes = [str(a).strip() or f"축{i + 1}" for i, a in enumerate(v[5:7])]
    nums = [round(float(a), 2) for a in v[7:21]]
    X = [nums[2 * i:2 * i + 2] for i in range(5)]
    W = [nums[10:12], nums[12:14]]
    edges = [list(GC_PAIRS[GC_PAIR_LABELS.index(lbl)]) for lbl in (v[21] or [])]
    focus = v[22] if isinstance(v[22], int) else int(str(v[22])[0]) - 1
    return {"nodes": nodes, "axis_names": axes, "X": X, "W": W, "edges": edges, "focus": focus}


def gc_verify(*v):
    try:
        r = gc.run(gc_cfg(*v))
    except Exception as e:
        return status_md(False, f"설정 오류: `{e}`"), ""
    nm = r["nodes"]
    nb = ", ".join(nm[j] for j in r["neighbors"]) or "없음"
    detail = (f"독립 구현 대조 최악 오차 **{r['max_ref']:.1e}**, 불변식 최대 오차 {max(r['inv'].values()):.1e} "
              f"(데모 그래프 + 무작위 그래프 {r['n_random']}개, 허용 {gc.TOL:.0e})  \n"
              f"'{nm[r['focus']]}'의 이웃: {nb}")
    return status_md(r["passed"], detail), gc.report(r)


def gc_inject(*v):
    return gc.inject_report(gc_cfg(*v))


def gc_render(quality, *v):
    yield from render("gcn", gc_cfg(*v), quality)


GC_HELP = """
**쉽게** : 각 노드가 이웃의 정보를 조금씩 받아 자기 정보를 새로 만드는 과정을 보여줍니다. 층을 쌓을수록 더 먼 노드의 정보까지 닿습니다.

**전문적으로** : GCN 한 층 H' = ReLU(ÂXW), Â = D̃^(-1/2)(A+I)D̃^(-1/2) (Kipf & Welling, 2017)을 밀집 NumPy로 계산하고,
순수 파이썬 메시지 패싱·scipy.sparse 구현과 대조합니다. 대칭성, 고유값 1의 고유벡터 √d̃, |λ| ≤ 1, 순열 등변성,
수용 영역(Â^k의 0이 아닌 칸 = k-hop)을 데모 그래프 + 무작위 그래프 20개에서 확인합니다(허용 오차 1e-10).

**해볼 만한 실험** : 간선을 끊어 고립 노드 만들기 · 허브 노드에 연결 몰아주기 · W를 단위행렬로 바꿔 순수한 스무딩만 보기
"""


# ============================================================================
# GAN 탭
# ============================================================================
def gn_cfg(seed, steps, lr, std, pack):
    return {"seed": int(seed), "steps": int(steps), "lr": float(lr), "std": float(std), "pack": int(pack)}


def gn_verify(*v):
    yield "### ⏳ 학습 · 검증 중... (기본 GAN과 PacGAN 학습 + 재학습, 수십 초)", ""
    try:
        r = gn.run(gn_cfg(*v))
    except Exception as e:
        yield status_md(False, f"설정 오류: `{e}`"), ""
        return
    c, f = r["config"], r["final"]
    p = c["pack"]
    vv = r["verify"]
    detail = (f"덮은 모드: 기본 GAN **{f[1]['covered']}/{c['K']}**, PacGAN **{f[p]['covered']}/{c['K']}**  \n"
              + ("관문 3개 모두 통과 (기울기 · 지표 · 재현성)" if vv["passed"]
                 else f"{vv['stopped_at']}에서 차단"))
    yield status_md(r["passed"], detail), gn.report(r)


def gn_inject(*v):
    yield "⏳ 오류 6종을 주입해 짧게 학습하며 확인 중... (1~2분)"
    yield gn.inject_report(gn_cfg(*v))


def gn_seeds(*v):
    yield "⏳ 시드 0~4로 두 모델을 각각 학습 중... (1~2분)"
    yield gn.seeds_report(5, gn_cfg(*v))


def gn_render(quality, *v):
    yield from render("gan", gn_cfg(*v), quality)


GN_HELP = """
**쉽게** : 생성 모델이 다양한 답을 만들지 못하고 몇 가지 답만 반복하는 '모드 붕괴'를, 실제로 학습시킨 두 모델로 비교합니다.

**전문적으로** : 8-가우시안 링에서 기본 GAN(non-saturating loss)과 PacGAN(Lin et al., 2018)을 NumPy로 같은 시드·조건에서 학습합니다.
기울기 검사(수치미분 대조) → 지표 검사(두 구현 일치 + 진짜 데이터 보정) → 재현성·출처 검사(재학습 스냅샷과 비트 단위 일치)를
통과해야만 렌더링합니다. 영상의 모든 점은 실제 학습 스냅샷입니다.

**해볼 만한 실험** : 시드 바꾸기 · 모드 표준편차를 0.05로 넓히기(붕괴가 약해지는지) · PacGAN 묶음 4로 늘리기 · **여러 시드 요약**으로 한 시드만 고른 결과가 아님을 확인
"""


# ============================================================================
# 자연어 요청 탭 (로컬 Ollama)
# ============================================================================
LLM_HELP = """
**자연어로 요청하면** 로컬 LLM이 4개 템플릿(역전파 · 셀프 어텐션 · GCN · GAN 모드 붕괴) 중 하나와 설정값을 골라 JSON으로 냅니다.
LLM은 코드를 쓰지 않으며, 그 JSON도 **형식·범위 검사 → 계산 모듈 검증**을 통과해야 합니다. 실패하면 오류를 LLM에 되돌려 최대 3번 스스로 고치게 합니다.

LLM이 쓴 설명은 참고용이고, 영상 속 숫자와 문장은 언제나 계산 결과에서 만들어집니다.
"""

LLM_EXAMPLES = [
    "정답을 0으로 바꾼 역전파 과정을 보여줘",
    "'학생이 인공지능을 배운다'에서 '배운다'가 어디를 보는지",
    "다섯 명이 한 줄로 손잡고 있는 경로 그래프에서 가운데 사람이 받는 메시지",
    "PacGAN 묶음을 4로 늘리면 모드 붕괴가 줄어드는지 시드 3으로",
]


def ollama_models(url):
    try:
        ms = lr.list_models(url)
    except Exception as e:
        return (gr.update(choices=[], value=None),
                f"⚠️ Ollama에 연결하지 못했습니다 (`{e}`)  \n"
                "Ollama 실행 여부, 주소·포트(기본 11434), 다른 프로그램의 포트 점유 여부를 확인하세요.")
    if not ms:
        return gr.update(choices=[], value=None), "⚠️ 연결은 됐지만 설치된 모델이 없습니다. `ollama pull <모델>`로 받아 주세요."
    labels = [(f"{m}  (클라우드 · 요청이 PC 밖으로 전송)" if lr.is_cloud(m) else m, m) for m in ms]
    n_cloud = sum(lr.is_cloud(m) for m in ms)
    note = f" (로컬 {len(ms) - n_cloud}개, 클라우드 {n_cloud}개)" if n_cloud else ""
    return gr.update(choices=labels, value=lr.pick_default(ms)), f"✅ 연결됨 · 모델 {len(ms)}개{note}"


def llm_parse(request, model, url):
    if not (request or "").strip():
        yield "요청을 입력하세요.", "", "", None
        return
    if not model:
        yield "먼저 **모델 목록 불러오기**로 모델을 고르세요.", "", "", None
        return
    yield f"### ⏳ 해석 중... (모델: {model} · 첫 호출은 모델을 GPU에 올리느라 더 걸립니다)", "", "", None
    try:
        r = lr.route(request, model, url)
    except Exception as e:
        yield f"### ❌ LLM 호출 실패\n`{e}`", "", "", None
        return
    log_dir = WORK / f"llm_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "route.json").write_text(json.dumps(dict(request=request, **r), ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    attempts = "\n\n".join(
        f"[시도 {a['attempt']}] {a['seconds']:.1f}초\n{a['raw']}\n→ "
        + ("통과" if not a["errors"] else "문제: " + " / ".join(a["errors"])) for a in r["attempts"])
    if r["unsupported"]:
        yield (f"### ℹ️ 지원하지 않는 요청\nLLM 판단: {r['reason']}  \n"
               "지원 템플릿: 역전파 · 셀프 어텐션 · GCN 메시지 패싱 · GAN 모드 붕괴"), "", attempts, None
        return
    if not r["ok"]:
        last = r["attempts"][-1]["errors"]
        yield (f"### ❌ {lr.MAX_REPAIR}번 고쳐도 통과하지 못했습니다\n마지막 문제: " + " / ".join(last)), "", attempts, None
        return
    name = DEMOS[r["template"]]["label"]
    extra = "  \nGAN은 **영상 만들기** 단계에서 실제 학습과 3중 검증을 합니다." if r["template"] == "gan" else ""
    yield (f"### ✅ 해석 완료 → **{name}**\n"
           f"자동 수정 {r['repairs']}회 · {r['seconds']:.1f}초 · 형식·범위 검사와 계산 모듈 검증 통과{extra}  \n"
           f"LLM 설명 (참고용, 검증 대상 아님): {r['reason']}"), \
        json.dumps(dict(template=r["template"], config=r["config"]), ensure_ascii=False, indent=2), \
        attempts, dict(template=r["template"], config=r["config"])


def llm_render(quality, state):
    if not state:
        yield "먼저 **① 해석하기**로 검증을 통과한 설정을 만드세요.", None, ""
        return
    yield from render(state["template"], state["config"], quality)


def llm_bench(model, url, set_name="dev", prompt=lr.DEFAULT_PROMPT):
    if not model:
        yield "먼저 모델을 고르세요."
        return
    n = len(lr.SETS[set_name])
    lines = [f"{model} · 세트 {set_name} ({n}문항) · {lr.PROMPT_LABEL.get(prompt, prompt)} 프롬프트 "
             "(문항마다 수 초~수십 초)",
             "O = 의도까지 일치 · △ = 검증은 통과했지만 요청한 값과 다름 · X = 실패"]
    yield "\n".join(lines)
    rows = []
    for kind, v in lr.bench_iter(model, url, set_name, prompt):
        if kind == "row":
            rows.append(v)
            lines.append(lr.row_line(len(rows), v, n))
        else:
            f = lambda x, s: "-" if x is None else format(x, s)
            lines += lr.failure_lines(rows)
            lines.append(f"\n라우팅 {v['routing']:.0%} · 의도 일치 {v['intent']:.0%} · 첫 시도 성공 {v['first_try']:.0%} · "
                         f"평균 수정 {f(v['mean_repairs'], '.2f')}회 · 평균 {f(v['mean_seconds'], '.1f')}초")
            lines.append(f"저장: {v['saved']}")
        yield "\n".join(lines)


# ============================================================================
# 화면 구성
# ============================================================================
INTRO = """
# 검증형 Manim 시각화 생성기
**사용 순서** : ① 왼쪽에서 값을 바꾼다 → ② **검증하기** → ③ **영상 만들기**

검증을 통과하지 못하면 영상이 만들어지지 않습니다. 화면의 숫자와 설명 문장은 모두 실제 계산 결과에서 만들어집니다.
"""

BP_HELP = """
**쉽게** : 신경망이 틀린 만큼 각 가중치를 얼마나 고쳐야 하는지, 출력에서 입력 쪽으로 거꾸로 계산하는 과정을 보여줍니다.

**전문적으로** : 2-2-1 MLP(시그모이드, MSE)의 해석적 기울기를 연쇄법칙으로 계산하고, 중앙 차분 수치미분과
데모 값 + 무작위 입력 20세트에서 대조합니다(허용 오차 1e-8). 통과한 값으로만 1회 경사하강 갱신까지 렌더링합니다.

**해볼 만한 실험** : 정답 y를 0으로 바꾸기 · w5와 w6의 부호를 같게 만들기 · 학습률을 크게 하기
"""

AT_HELP = """
**쉽게** : 문장 속 각 단어가 다른 단어를 얼마나 참고하는지를 선의 굵기로 보여줍니다.

**전문적으로** : softmax(QKᵀ/√d_k)V 를 벡터화 NumPy로 계산하고, 순수 파이썬 루프·scipy 구현과 대조하며
행 합 = 1, 가중치 ≥ 0 불변식을 데모 값 + 무작위 입력 20세트에서 확인합니다(허용 오차 1e-10).

**해볼 만한 실험** : 단어를 바꾸기 · 집중해서 볼 단어 바꾸기 · W_Q를 단위행렬로 만들어 패턴 비교하기
"""


def slider(label, value, lo=-2.0, hi=2.0, step=0.05):
    return gr.Slider(lo, hi, value=value, step=step, label=label)


def build_ui():
    with gr.Blocks(title="검증형 Manim 시각화 생성기") as demo:
        gr.Markdown(INTRO)

        # -------------------------------------------------- 자연어 요청 (Ollama)
        with gr.Tab("자연어로 요청 (Ollama)"):
            gr.Markdown(LLM_HELP)
            with gr.Row():
                o_url = gr.Textbox(value=lr.DEFAULT_URL, label="Ollama 주소", scale=2)
                o_model = gr.Dropdown(choices=[], value=None, label="모델", allow_custom_value=True, scale=2)
                o_refresh = gr.Button("모델 목록 불러오기", scale=1)
            o_conn = gr.Markdown("")
            o_req = gr.Textbox(lines=2, label="무엇을 보고 싶으세요?",
                               placeholder="예: '학생이 인공지능을 배운다'에서 '배운다'가 어디를 보는지")
            gr.Examples(LLM_EXAMPLES, inputs=o_req, label="예시 요청")
            with gr.Row():
                o_parse = gr.Button("① 해석하기", variant="secondary")
                o_q = gr.Dropdown(list(QUALITY), value=list(QUALITY)[0], label="화질")
                o_render = gr.Button("② 영상 만들기", variant="primary")
            o_state = gr.State(None)
            o_status = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=1):
                    o_json = gr.Code(language="json", label="LLM이 고른 템플릿과 설정 (검증 통과본)")
                    with gr.Accordion("LLM 시도 기록 (원문 · 문제 · 수정)", open=False):
                        o_attempts = gr.Textbox(lines=12, show_label=False)
                with gr.Column(scale=1):
                    o_video = gr.Video(label="결과 영상")
                    with gr.Accordion("렌더링 로그", open=False):
                        o_log = gr.Textbox(lines=8, show_label=False)
            with gr.Accordion("모델 벤치마크 (논문 표 4: 라우팅 · 의도 일치, 최종 프롬프트)", open=False):
                with gr.Row():
                    o_bset = gr.Radio(["dev", "holdout"], value="dev", label="문항 세트 (개발 18 · 보류 20)")
                    o_bbtn = gr.Button("선택한 모델로 실행", size="sm")
                o_bench = gr.Textbox(lines=24, show_label=False)

            o_refresh.click(ollama_models, o_url, [o_model, o_conn])
            demo.load(ollama_models, o_url, [o_model, o_conn])
            o_parse.click(llm_parse, [o_req, o_model, o_url], [o_status, o_json, o_attempts, o_state])
            o_render.click(llm_render, [o_q, o_state], [o_status, o_video, o_log])
            o_bbtn.click(llm_bench, [o_model, o_url, o_bset], o_bench)

        # -------------------------------------------------- 역전파
        with gr.Tab("역전파 (Backpropagation)"):
            gr.Markdown(BP_HELP)
            D = bp_defaults()
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**입력 · 정답 · 학습률**")
                    with gr.Row():
                        bx1 = slider("x1", D[0], step=0.1)
                        bx2 = slider("x2", D[1], step=0.1)
                    with gr.Row():
                        by = slider("정답 y", D[2], 0.0, 1.0)
                        blr = slider("학습률 η", D[3], 0.1, 5.0, step=0.1)
                    gr.Markdown("**가중치 · 편향**")
                    with gr.Row():
                        bw = [slider(k, D[4 + i]) for i, k in enumerate(["w1", "w2", "w3"])]
                    with gr.Row():
                        bw += [slider(k, D[7 + i]) for i, k in enumerate(["w4", "w5", "w6"])]
                    with gr.Row():
                        bb = [slider(k, D[10 + i]) for i, k in enumerate(["b1", "b2", "b3"])]
                    bp_inputs = [bx1, bx2, by, blr] + bw + bb
                    bp_reset = gr.Button("기본값으로 되돌리기", size="sm")
                with gr.Column(scale=1):
                    bp_status = gr.Markdown("값을 정한 뒤 **검증하기**를 누르세요.")
                    with gr.Row():
                        bp_vbtn = gr.Button("① 검증하기", variant="secondary")
                        bp_q = gr.Dropdown(list(QUALITY), value=list(QUALITY)[0], label="화질")
                        bp_rbtn = gr.Button("② 영상 만들기", variant="primary")
                    bp_video = gr.Video(label="결과 영상")
                    with gr.Accordion("검증 리포트", open=False):
                        bp_report = gr.Textbox(lines=18, show_label=False)
                    with gr.Accordion("오류 주입 실험 (검증 장치가 실수를 잡아내는지)", open=False):
                        bp_ibtn = gr.Button("실험 실행", size="sm")
                        bp_inj = gr.Textbox(lines=6, show_label=False)
                    with gr.Accordion("렌더링 로그", open=False):
                        bp_log = gr.Textbox(lines=8, show_label=False)

            bp_vbtn.click(bp_verify, bp_inputs, [bp_status, bp_report])
            bp_ibtn.click(bp_inject, bp_inputs, bp_inj)
            bp_rbtn.click(bp_render, [bp_q] + bp_inputs, [bp_status, bp_video, bp_log])
            bp_reset.click(lambda: bp_defaults(), None, bp_inputs)

        # -------------------------------------------------- 어텐션
        with gr.Tab("셀프 어텐션 (Self-Attention)"):
            gr.Markdown(AT_HELP)
            D = at_defaults()
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**문장 (단어 3개)**")
                    with gr.Row():
                        toks = [gr.Textbox(value=D[i], label=f"{i + 1}번 단어", max_lines=1)
                                for i in range(3)]
                    with gr.Row():
                        axes = [gr.Textbox(value=D[3 + i], label=f"임베딩 축 {i + 1} 이름", max_lines=1)
                                for i in range(2)]
                    gr.Markdown("**단어별 임베딩 (축1, 축2)**")
                    xs = []
                    for i in range(3):
                        with gr.Row():
                            xs += [slider(f"{i + 1}번 단어 · 축{j + 1}", D[5 + 2 * i + j], step=0.1)
                                   for j in range(2)]
                    gr.Markdown("**질문 가중치 W_Q (2×2)**")
                    wq = []
                    for i in range(2):
                        with gr.Row():
                            wq += [slider(f"W_Q[{i + 1},{j + 1}]", D[11 + 2 * i + j]) for j in range(2)]
                    with gr.Accordion("고급: W_K, W_V (기본값은 단위행렬)", open=False):
                        wk, wv = [], []
                        for i in range(2):
                            with gr.Row():
                                wk += [slider(f"W_K[{i + 1},{j + 1}]", D[15 + 2 * i + j]) for j in range(2)]
                        for i in range(2):
                            with gr.Row():
                                wv += [slider(f"W_V[{i + 1},{j + 1}]", D[19 + 2 * i + j]) for j in range(2)]
                    focus = gr.Radio(["1번 단어", "2번 단어", "3번 단어"], value=f"{D[23] + 1}번 단어",
                                     type="index", label="영상에서 자세히 풀어볼 단어")
                    at_inputs = toks + axes + xs + wq + wk + wv + [focus]
                    at_reset = gr.Button("기본값으로 되돌리기", size="sm")
                with gr.Column(scale=1):
                    at_status = gr.Markdown("값을 정한 뒤 **검증하기**를 누르세요.")
                    with gr.Row():
                        at_vbtn = gr.Button("① 검증하기", variant="secondary")
                        at_q = gr.Dropdown(list(QUALITY), value=list(QUALITY)[0], label="화질")
                        at_rbtn = gr.Button("② 영상 만들기", variant="primary")
                    at_video = gr.Video(label="결과 영상")
                    with gr.Accordion("검증 리포트", open=False):
                        at_report = gr.Textbox(lines=16, show_label=False)
                    with gr.Accordion("오류 주입 실험 (검증 장치가 실수를 잡아내는지)", open=False):
                        at_ibtn = gr.Button("실험 실행", size="sm")
                        at_inj = gr.Textbox(lines=7, show_label=False)
                    with gr.Accordion("렌더링 로그", open=False):
                        at_log = gr.Textbox(lines=8, show_label=False)

            at_vbtn.click(at_verify, at_inputs, [at_status, at_report])
            at_ibtn.click(at_inject, at_inputs, at_inj)
            at_rbtn.click(at_render, [at_q] + at_inputs, [at_status, at_video, at_log])
            at_reset.click(lambda: at_defaults()[:-1] + [f"{at.DEFAULT['focus'] + 1}번 단어"],
                           None, at_inputs)

        # -------------------------------------------------- GCN
        with gr.Tab("GCN 메시지 패싱"):
            gr.Markdown(GC_HELP)
            D = gc_defaults()
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**노드 이름 (5개) · 특징 축 이름**")
                    with gr.Row():
                        gnodes = [gr.Textbox(value=D[i], label=f"{i + 1}번 노드", max_lines=1)
                                  for i in range(5)]
                    with gr.Row():
                        gaxes = [gr.Textbox(value=D[5 + i], label=f"특징 축 {i + 1}", max_lines=1)
                                 for i in range(2)]
                    gr.Markdown("**연결 (간선)** — 번호는 위 노드 순서")
                    gedges = gr.CheckboxGroup(GC_PAIR_LABELS, value=D[21], show_label=False)
                    gr.Markdown("**노드별 특징 (축1, 축2)**")
                    gx = []
                    for i in range(5):
                        with gr.Row():
                            gx += [slider(f"{i + 1}번 · 축{j + 1}", D[7 + 2 * i + j], step=0.1)
                                   for j in range(2)]
                    gr.Markdown("**공유 가중치 W (2×2)**")
                    gw = []
                    for i in range(2):
                        with gr.Row():
                            gw += [slider(f"W[{i + 1},{j + 1}]", D[17 + 2 * i + j]) for j in range(2)]
                    gfocus = gr.Radio([f"{i + 1}번 노드" for i in range(5)], value=D[22], type="index",
                                      label="메시지 패싱을 자세히 풀어볼 노드")
                    gc_inputs = gnodes + gaxes + gx + gw + [gedges, gfocus]
                    gc_reset = gr.Button("기본값으로 되돌리기", size="sm")
                with gr.Column(scale=1):
                    gc_status = gr.Markdown("값을 정한 뒤 **검증하기**를 누르세요.")
                    with gr.Row():
                        gc_vbtn = gr.Button("① 검증하기", variant="secondary")
                        gc_q = gr.Dropdown(list(QUALITY), value=list(QUALITY)[0], label="화질")
                        gc_rbtn = gr.Button("② 영상 만들기", variant="primary")
                    gc_video = gr.Video(label="결과 영상")
                    with gr.Accordion("검증 리포트", open=False):
                        gc_report = gr.Textbox(lines=20, show_label=False)
                    with gr.Accordion("오류 주입 실험 (검증 장치가 실수를 잡아내는지)", open=False):
                        gc_ibtn = gr.Button("실험 실행", size="sm")
                        gc_inj = gr.Textbox(lines=12, show_label=False)
                    with gr.Accordion("렌더링 로그", open=False):
                        gc_log = gr.Textbox(lines=8, show_label=False)

            gc_vbtn.click(gc_verify, gc_inputs, [gc_status, gc_report])
            gc_ibtn.click(gc_inject, gc_inputs, gc_inj)
            gc_rbtn.click(gc_render, [gc_q] + gc_inputs, [gc_status, gc_video, gc_log])
            gc_reset.click(lambda: gc_defaults(), None, gc_inputs)

        # -------------------------------------------------- GAN
        with gr.Tab("GAN 모드 붕괴"):
            gr.Markdown(GN_HELP)
            D = gn.DEFAULT
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**학습 조건** (기본 GAN과 PacGAN에 똑같이 적용)")
                    nseed = gr.Slider(0, 9, value=D["seed"], step=1, label="시드")
                    nsteps = gr.Slider(1000, 4000, value=D["steps"], step=500, label="학습 단계 수")
                    with gr.Row():
                        nlr = gr.Dropdown([0.001, 0.003], value=D["lr"], label="학습률")
                        nstd = gr.Dropdown([0.02, 0.05], value=D["std"], label="모드 표준편차 (좁을수록 붕괴 쉬움)")
                    npack = gr.Radio([2, 4], value=D["pack"], label="PacGAN 묶음 크기")
                    gn_inputs = [nseed, nsteps, nlr, nstd, npack]
                    gn_reset = gr.Button("기본값으로 되돌리기", size="sm")
                with gr.Column(scale=1):
                    gn_status = gr.Markdown("조건을 정한 뒤 **검증하기**를 누르세요. (학습이 포함되어 수십 초 걸립니다)")
                    with gr.Row():
                        gn_vbtn = gr.Button("① 검증하기", variant="secondary")
                        gn_q = gr.Dropdown(list(QUALITY), value=list(QUALITY)[0], label="화질")
                        gn_rbtn = gr.Button("② 영상 만들기", variant="primary")
                    gn_video = gr.Video(label="결과 영상")
                    with gr.Accordion("검증 리포트", open=False):
                        gn_report = gr.Textbox(lines=16, show_label=False)
                    with gr.Accordion("여러 시드 요약 (논문 보고용)", open=False):
                        gn_sbtn = gr.Button("시드 0~4 요약 실행", size="sm")
                        gn_seed_out = gr.Textbox(lines=4, show_label=False)
                    with gr.Accordion("오류 주입 실험 (각 오류가 어느 관문에서 잡히는지)", open=False):
                        gn_ibtn = gr.Button("실험 실행", size="sm")
                        gn_inj = gr.Textbox(lines=9, show_label=False)
                    with gr.Accordion("렌더링 로그", open=False):
                        gn_log = gr.Textbox(lines=8, show_label=False)

            gn_vbtn.click(gn_verify, gn_inputs, [gn_status, gn_report])
            gn_sbtn.click(gn_seeds, gn_inputs, gn_seed_out)
            gn_ibtn.click(gn_inject, gn_inputs, gn_inj)
            gn_rbtn.click(gn_render, [gn_q] + gn_inputs, [gn_status, gn_video, gn_log])
            gn_reset.click(lambda D=D: [D["seed"], D["steps"], D["lr"], D["std"], D["pack"]], None, gn_inputs)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860,
                              allowed_paths=[str(WORK)], inbrowser=True)
