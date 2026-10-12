"""
layout_check.py · Manim 장면의 글자 겹침·화면 이탈 자동 검사기
================================================================
장면을 렌더링 없이(dry run) 끝까지 실행하면서, self.play / self.wait 가 끝날 때마다
화면에 보이는 글자 객체(Text, MathTex, 숫자 등)의 경계 상자를 재서 다음을 보고한다.

  OVERLAP  : 글자 두 개가 서로 겹침 (작은 쪽 면적의 3% 이상)
  CROSS    : 글자가 상자·원의 테두리에 걸침 (완전히 안에 있거나 완전히 밖이면 정상)
  OFFSCREEN: 글자가 화면(가로 ±7.11, 세로 ±4) 밖으로 나감
  ON_CURVE : 선·곡선이 글자를 관통 (글자에 배경 상자가 있으면 의도된 것으로 보고 제외)

사용법
  python layout_check.py <파일.py> <장면 이름> [<장면 이름> ...]
  환경변수 LAYOUT_CFG_ENV="GCN_CONFIG=path.json" 처럼 설정 파일도 줄 수 있다.
"""
import importlib.util
import os
import sys

import numpy as np
from manim import (Circle, DecimalNumber, Dot, Integer, MarkupText, MathTex, Rectangle,
                   Scene, SingleStringMathTex, Tex, Text, VMobject, config, tempconfig)

TEXT_TYPES = (Text, MarkupText, MathTex, SingleStringMathTex, Tex, DecimalNumber, Integer)
SHAPE_TYPES = (Circle, Rectangle)              # Square, RoundedRectangle 포함
FRAME_X, FRAME_Y = 7.111, 4.0
MIN_OVERLAP = 0.03


def label(m):
    for attr in ("text", "tex_string"):
        v = getattr(m, attr, None)
        if isinstance(v, str) and v.strip():
            return v.replace("\n", " ")[:40]
    if isinstance(m, (DecimalNumber, Integer)):
        return f"<num {m.get_value():g}>"
    return type(m).__name__


def visible(m):
    ops = []
    for s in m.get_family():
        try:
            ops.append(max(float(np.max(s.get_fill_opacity())), float(np.max(s.get_stroke_opacity()))))
        except Exception:
            pass
    return bool(ops) and max(ops) > 0.05


def box(m):
    (x0, y0, _), (x1, y1, _) = m.get_corner([-1, -1, 0]), m.get_corner([1, 1, 0])
    return x0, y0, x1, y1


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def inside(a, b, tol=0.01):          # a 가 b 안에 완전히
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def collect(mobs, texts, shapes, strokes=None):
    for m in mobs:
        if isinstance(m, TEXT_TYPES):
            if visible(m) and m.width > 1e-3:
                texts.append(m)
            continue                  # 글자 객체 내부(글리프)는 내려가지 않음
        if isinstance(m, Dot):
            continue                  # 움직이는 점·산점도는 제외
        if isinstance(m, SHAPE_TYPES) and visible(m) and m.width > 0.05:
            shapes.append(m)
            continue
        if strokes is not None and isinstance(m, VMobject) and not m.submobjects and len(m.points) >= 4:
            try:
                if float(np.max(m.get_stroke_opacity())) > 0.05 and float(np.max(m.get_stroke_width())) > 0:
                    strokes.append(m)
            except Exception:
                pass
        collect(m.submobjects, texts, shapes, strokes)


def stroke_samples(m):
    """3차 베지어 곡선을 길이에 맞춰 촘촘히 샘플링."""
    P = m.points
    k = len(P) // 4
    out = []
    for i in range(k):
        p0, p1, p2, p3 = P[4 * i:4 * i + 4]
        n = int(np.clip(np.linalg.norm(p3 - p0) / 0.04, 6, 400))
        t = np.linspace(0, 1, n)[:, None]
        out.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
    return np.vstack(out) if out else np.zeros((0, 3))


class Checker:
    def __init__(self, scene_name, fixed_only=False):
        self.scene_name, self.fixed_only = scene_name, fixed_only
        self.step, self.issues = 0, {}

    def check(self, scene, what):
        self.step += 1
        snaps = [int(x) for x in os.environ.get("SNAP_STEPS", "").split(",") if x.strip()]
        if self.step in snaps:                       # 지정한 단계의 정지 화면 저장
            from manim.camera.camera import Camera
            cam = Camera()
            cam.capture_mobjects(scene.mobjects)
            out = f"snap_{self.scene_name}_{self.step:03d}.png"
            cam.get_image().save(out)
        mobs = scene.mobjects
        if self.fixed_only:                          # ThreeDScene: 화면 고정 객체만 2D로 검사
            fixed = getattr(scene.camera, "fixed_in_frame_mobjects", set())
            mobs = [m for m in mobs if m in fixed]
        texts, shapes, strokes = [], [], []
        collect(mobs, texts, shapes, strokes)
        tb = [(t, box(t)) for t in texts]
        samples = [(s, stroke_samples(s)) for s in strokes]
        for i in range(len(tb)):
            a, ba = tb[i]
            if ba[0] < -FRAME_X - 0.02 or ba[2] > FRAME_X + 0.02 or ba[1] < -FRAME_Y - 0.02 or ba[3] > FRAME_Y + 0.02:
                self.add("OFFSCREEN", (label(a),), what,
                         f"x[{ba[0]:.2f},{ba[2]:.2f}] y[{ba[1]:.2f},{ba[3]:.2f}]")
            for j in range(i + 1, len(tb)):
                b, bb = tb[j]
                if a in b.get_family() or b in a.get_family():
                    continue
                ov = inter(ba, bb)
                small = min(area(ba), area(bb))
                if small > 0 and ov / small > MIN_OVERLAP:
                    self.add("OVERLAP", tuple(sorted((label(a), label(b)))), what, f"{ov / small:.0%}")
            if getattr(a, "background_rectangle", None) is None:
                m_ = 0.03                                   # 테두리 여유
                for s, pts in samples:
                    if s in a.get_family() or len(pts) == 0:
                        continue
                    hit = ((pts[:, 0] > ba[0] + m_) & (pts[:, 0] < ba[2] - m_) &
                           (pts[:, 1] > ba[1] + m_) & (pts[:, 1] < ba[3] - m_)).sum()
                    if hit >= 2:
                        self.add("ON_CURVE", (label(a), type(s).__name__), what, f"{hit}점")
            for s in shapes:
                sb = box(s)
                if s in a.get_family():
                    continue
                if inter(ba, sb) > 0 and not inside(ba, sb) and not inside(sb, ba):
                    frac = inter(ba, sb) / area(ba)
                    if 0.03 < frac < 0.97:                 # 테두리에 걸친 경우만
                        self.add("CROSS", (label(a), type(s).__name__), what, f"{frac:.0%}")

    def add(self, kind, key, what, detail):
        k = (kind, key)
        if k not in self.issues:
            self.issues[k] = dict(first_step=self.step, what=what, detail=detail, count=0)
        self.issues[k]["count"] += 1

    def report(self):
        out = [f"[{self.scene_name}] 검사 단계 {self.step}개, 문제 {len(self.issues)}건"]
        for (kind, key), v in sorted(self.issues.items(), key=lambda kv: kv[1]["first_step"]):
            out.append(f"  {kind:<9} 단계 {v['first_step']:>3} ({v['what']}) × {v['count']}  "
                       f"{' | '.join(key)}  [{v['detail']}]")
        return "\n".join(out)


def run(path, scene_names):
    spec = importlib.util.spec_from_file_location("demo_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    orig_play, orig_wait = Scene.play, Scene.wait
    reports = []
    for name in scene_names:
        cls = getattr(mod, name)
        chk = Checker(name, fixed_only=any(c.__name__ == "ThreeDScene" for c in cls.__mro__))

        def play(self, *anims, **kw):
            orig_play(self, *anims, **kw)
            chk.check(self, "play: " + ",".join(type(a).__name__ for a in anims)[:40])

        def wait(self, *a, **kw):
            orig_wait(self, *a, **kw)
            chk.check(self, "wait")

        Scene.play, Scene.wait = play, wait
        try:
            with tempconfig({"dry_run": True, "quality": "low_quality", "disable_caching": True,
                             "verbosity": "ERROR", "progress_bar": "none"}):
                cls().render()
        except Exception as e:
            reports.append(f"[{name}] 실행 오류: {type(e).__name__}: {e}")
            continue
        finally:
            Scene.play, Scene.wait = orig_play, orig_wait
        reports.append(chk.report())
    return reports


if __name__ == "__main__":
    for line in run(sys.argv[1], sys.argv[2:]):
        print(line)
