"""
================================================================================
gcn_demo.py · GCN 메시지 패싱 시각화 — 특집호 Demo 9
================================================================================

구성
  gcn_core.py  : 계산 + 검증 (독립 구현 3종, 그래프 불변식 5종, 무작위 그래프 20개)
  gcn_demo.py  : (이 파일) 검증을 통과한 숫자만 받아서 그림

장면 흐름 (약 90초)
  ① 그래프와 특징      : 노드 색 = 두 성향의 비율
  ② 자기 루프          : Ã = A + I, 차수 d̃
  ③ 메시지 패싱        : 한 노드로 이웃의 메시지가 계수 1/√(d̃ᵢd̃ⱼ)만큼 줄어 모임
  ④ 변환과 활성화      : 공유 가중치 W, ReLU
  ⑤ 모든 노드 동시에   : 한 층 = 모든 노드가 같은 계산
  ⑥ 행렬 형태          : H' = ReLU(ÂXW)
  ⑦ 층을 쌓으면        : Â² 의 0이 아닌 칸 = 2-hop 이웃 (수용 영역)
  ⑧ 검증              : 통과하지 못하면 렌더링 자체를 중단

실행
  python gcn_core.py --inject
  manim -pql gcn_demo.py GCNMessagePassing

한글 규칙 : 노드 이름·축 이름 등 한글이 들어갈 수 있는 값은 Text 로만, 수식은 MathTex 로만
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manim import *
from gcn_core import run

KO_FONT = os.environ.get("KO_FONT", "Malgun Gothic")
C1, C2 = BLUE, GOLD                 # 축1 ↔ 축2 성향 색
MSG = ORANGE
HM = TEAL
OK_C = GREEN
HOP_COLORS = {0: WHITE, 1: GREEN, 2: YELLOW}


# ---------------------------------------------------------------------------
# 한국어 조사 자동 선택 : 사용자가 바꾼 이름 뒤에 '(으)로'처럼 두 형태를 병기하지 않도록
#   받침 없음·ㄹ받침 → '로', 그 밖의 받침 → '으로' (영문·숫자는 읽는 소리의 끝으로 판단)
# ---------------------------------------------------------------------------
_LATIN_FINAL = {"L": "ㄹ", "M": "ㅁ", "N": "ㄴ", "R": "ㄹ"}          # 엘·엠·엔·알
_DIGIT_FINAL = {"0": "ㅇ", "1": "ㄹ", "3": "ㅁ", "6": "ㄱ", "7": "ㄹ", "8": "ㄹ"}   # 영·일·삼·육·칠·팔


def _final_sound(word):
    for ch in reversed(str(word).strip()):
        if "가" <= ch <= "힣":
            j = (ord(ch) - 0xAC00) % 28
            return None if j == 0 else ("ㄹ" if j == 8 else "받침")
        if ch.isdigit():
            return _DIGIT_FINAL.get(ch)
        if ch.isascii() and ch.isalpha():
            return _LATIN_FINAL.get(ch.upper())
    return None


def with_ro(word):
    """'도윤' → '도윤으로', '민수' → '민수로', '서울' → '서울로', 'C' → 'C로', 'N' → 'N으로'."""
    return f"{word}{'로' if _final_sound(word) in (None, 'ㄹ') else '으로'}"


def sci(v):
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def vec(v, d=3):
    return rf"({v[0]:.{d}f},\ {v[1]:.{d}f})"


def bmat(M):
    return (r"\begin{bmatrix}" + rf"{M[0][0]:.2f} & {M[0][1]:.2f}\\ {M[1][0]:.2f} & {M[1][1]:.2f}"
            + r"\end{bmatrix}")


class GCNMessagePassing(Scene):

    R = 0.36
    CENTER = np.array([-4.1, 0.7, 0.0])
    RX, RY = 2.0, 1.4
    ANGLES = [162, 234, 90, 18, 306]                     # 노드 0~4 (볼록 배치 → 간선이 노드를 지나지 않음)
    LED_X = [0.3, 1.45, 3.05, 5.55]     # 첫 열(이름)을 넓게
    LED_Y0, LED_DY = 2.35, 0.45
    HM_X = [1.45, 2.05, 2.65, 3.25, 3.85]
    HM_Y = [1.95, 1.37, 0.79, 0.21, -0.37]
    CELL = 0.56
    OUT_X = 5.75

    def pos(self, i):
        a = np.deg2rad(self.ANGLES[i])
        return self.CENTER + np.array([self.RX * np.cos(a), self.RY * np.sin(a), 0.0])

    @staticmethod
    def tone(v):
        return interpolate_color(C1, C2, 0.5 + 0.5 * np.tanh(v[1] - v[0]))

    def construct(self):
        # ============ 0. 계산 + 검증 ============
        r = run()
        if not r["passed"]:
            raise RuntimeError(f"GCN 검증 실패 (max diff = {r['max_ref']:.2e}, "
                               f"invariants = {r['inv']}) — 렌더링하지 않습니다.")
        self.r = r
        nm, X, AX, Z, H = r["nodes"], r["X"], r["AX"], r["Z"], r["H"]
        # [표시 수정] 긴 이름을 원 안에 억지로 줄이면 읽을 수 없다 → 한 노드라도 그렇게 되면
        #   원 안에는 번호(1~5)를 쓰고 이름은 표·장부·설명문에서 보여 준다
        probe = [Text(x, font=KO_FONT, font_size=24, weight=BOLD).width for x in nm]
        self.use_numbers = any(w * 0.72 > 2 * self.R * 0.8 for w in probe)
        self.nm = nm
        deg, Ahat, W = r["deg"], r["Ahat"], r["W"]
        n = len(nm)
        self.panel = None

        title = Text("GCN 메시지 패싱: 이웃의 정보를 모아 나를 갱신하기",
                     font=KO_FONT, font_size=30).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ======================= ① 그래프와 특징 =======================
        self.circles, names = [], []
        for i in range(n):
            c = Circle(radius=self.R, stroke_color=GREY_A, stroke_width=3,
                       fill_color=self.tone(X[i]), fill_opacity=0.9).move_to(self.pos(i))
            t = Text(self.node_tag(i), font=KO_FONT, font_size=24, color=BLACK, weight=BOLD)
            if t.width > 2 * self.R * 0.8:
                t.scale_to_fit_width(2 * self.R * 0.8)
            t.move_to(c)
            self.circles.append(c)
            names.append(t)
        edges = VGroup(*[Line(self.pos(i), self.pos(j), buff=self.R, color=GREY_B, stroke_width=3)
                         for i, j in r["edges"]])
        self.feat = [self.feat_label(i, X[i], GREY_A) for i in range(n)]
        legend = self.legend(r["axis_names"])

        self.play(Create(edges), LaggedStart(*[GrowFromCenter(VGroup(c, t))
                                               for c, t in zip(self.circles, names)], lag_ratio=0.15))
        self.play(*[FadeIn(f) for f in self.feat], FadeIn(legend))
        self.show_panel(
            "① 그래프와 특징: 각 노드는 2차원 특징 벡터를 가진다",
            [r"X^{\top}=\begin{bmatrix}" + " & ".join(f"{X[i][0]:.2f}" for i in range(n)) + r"\\ "
             + " & ".join(f"{X[i][1]:.2f}" for i in range(n)) + r"\end{bmatrix}"],
            note=((f"노드 번호(= 열 순서): " + ", ".join(f"{i + 1} {x}" for i, x in enumerate(nm)))
                  if self.use_numbers else f"열 순서: {', '.join(nm)}")
                 + f"   ·   행: {r['axis_names'][0]}, {r['axis_names'][1]}   ·   노드 색 = 두 성향의 비율")

        # ======================= ② 자기 루프 =======================
        self.rings = VGroup(*[DashedVMobject(Circle(radius=self.R + 0.1, color=GREY_B, stroke_width=2)
                                             .move_to(self.pos(i)), num_dashes=18) for i in range(n)])
        self.show_panel(
            "② 자기 루프 추가: 자기 자신의 정보도 메시지에 포함한다",
            [r"\tilde A=A+I,\qquad \tilde d_i=\sum_j \tilde A_{ij}"],
            note="차수(자기 자신 포함): " + ",  ".join(f"{nm[i]} = {int(deg[i])}" for i in range(n)))
        self.play(Create(self.rings), run_time=1.0)

        # ======================= ③ 한 노드로 메시지 수집 =======================
        f = r["focus"]
        nb = r["neighbors"]
        srcs = nb + [f]
        self.play(self.circles[f].animate.set_stroke(YELLOW, width=6))
        self.show_panel(
            f"③ 메시지 패싱: 이웃의 특징이 계수만큼 줄어들어 {with_ro(nm[f])} 모인다",
            [r"a_i=\sum_{j\in N(i)\cup\{i\}}\frac{x_j}{\sqrt{\tilde d_i\,\tilde d_j}}"],
            note=f"i = {nm[f]}   ·   이웃 = {', '.join(nm[j] for j in nb) if nb else '없음'} + 자기 자신")

        lx, y0, dy = self.LED_X, self.LED_Y0, self.LED_DY
        led_title = Text(f"{with_ro(nm[f])} 들어오는 메시지", font=KO_FONT, font_size=20, color=YELLOW
                         ).move_to([3.2, 2.85, 0])
        header = VGroup(
            Text("보낸 노드", font=KO_FONT, font_size=17).move_to([lx[0], y0, 0]),
            MathTex(r"\tilde d_j", font_size=26).move_to([lx[1], y0, 0]),
            VGroup(Text("계수", font=KO_FONT, font_size=17),
                   MathTex(r"c_{ij}=\tfrac{1}{\sqrt{\tilde d_i\tilde d_j}}", font_size=24)
                   ).arrange(RIGHT, buff=0.1).move_to([lx[2], y0, 0]),
            VGroup(Text("메시지", font=KO_FONT, font_size=17),
                   MathTex(r"c_{ij}\,x_j", font_size=26, color=MSG)).arrange(RIGHT, buff=0.1
                                                                            ).move_to([lx[3], y0, 0]),
        )
        rows_bottom = y0 - len(srcs) * dy
        led_bg = RoundedRectangle(width=7.4, height=(y0 - rows_bottom) + 1.25, corner_radius=0.15,
                                  stroke_color=GREY_D, stroke_width=1.5, fill_color=GREY_E,
                                  fill_opacity=0.35)
        led_bg.move_to([3.15, (y0 + 0.3 + rows_bottom - 0.95) / 2, 0])
        rule = Line([-0.5, y0 - 0.24, 0], [6.8, y0 - 0.24, 0], color=GREY_D, stroke_width=1.5)
        self.play(FadeIn(led_bg), FadeIn(led_title), FadeIn(header), Create(rule))

        ledger = VGroup()
        for k, j in enumerate(srcs):
            y = y0 - (k + 1) * dy
            c = Ahat[f][j]
            row = VGroup(
                self.fit(Text(nm[j] + (" (자신)" if j == f else ""), font=KO_FONT, font_size=19),
                         1.45).move_to([lx[0], y, 0]),       # [위치 수정] 긴 이름이 장부 밖으로 나가지 않게
                MathTex(f"{int(deg[j])}", font_size=26).move_to([lx[1], y, 0]),
                MathTex(rf"\tfrac{{1}}{{\sqrt{{{int(deg[f])}\cdot{int(deg[j])}}}}}={c:.3f}",
                        font_size=26).move_to([lx[2], y, 0]),
                MathTex(vec(c * X[j]), font_size=26, color=MSG).move_to([lx[3], y, 0]),
            )
            dot = Dot(self.pos(j), radius=0.1, color=self.tone(X[j]))
            path = (Line(self.pos(j), self.pos(f)) if j != f
                    else Circle(radius=self.R + 0.1).move_to(self.pos(f)).rotate(PI / 2))
            self.play(MoveAlongPath(dot, path), FadeIn(row, shift=LEFT * 0.15), run_time=1.0)
            self.play(FadeOut(dot), run_time=0.2)
            ledger.add(row)

        sum_rule = Line([-0.5, rows_bottom - 0.24, 0], [6.8, rows_bottom - 0.24, 0],
                        color=GREY_B, stroke_width=1.5)
        sum_row = VGroup(
            Text("합 (집계)", font=KO_FONT, font_size=19, color=YELLOW).move_to([lx[0], rows_bottom - 0.6, 0]),
            MathTex(r"a_i=" + vec(AX[f]), font_size=28, color=YELLOW).move_to([lx[3] - 0.25, rows_bottom - 0.6, 0]),
        )
        self.play(Create(sum_rule), FadeIn(sum_row),
                  self.circles[f].animate.set_fill(self.tone(AX[f])))
        self.wait(1.0)

        # ======================= ④ 변환과 활성화 =======================
        neg = (Z[f] < 0).any()
        self.show_panel(
            "④ 변환과 활성화: 모든 노드가 같은 가중치 W를 공유한다",
            [r"h_i'=\mathrm{ReLU}(a_iW)=\mathrm{ReLU}\Big(" + vec(AX[f]) + bmat(W) + r"\Big)"
             r"=\mathrm{ReLU}\big(" + vec(Z[f]) + r"\big)=" + vec(H[f])],
            note=("음수 성분은 ReLU가 0으로 바꾼다" if neg
                  else "이 노드는 음수 성분이 없어 ReLU 전후 값이 같다"))
        self.wait(0.8)

        # ======================= ⑤ 모든 노드 동시에 =======================
        self.play(FadeOut(VGroup(led_bg, led_title, header, rule, ledger, sum_rule, sum_row)),
                  self.circles[f].animate.set_stroke(GREY_A, width=3).set_fill(self.tone(X[f])))
        d0 = np.std(X[:, 1] - X[:, 0])
        d1 = np.std(AX[:, 1] - AX[:, 0])
        smooth = (f"이웃과 섞이면서 노드 사이 성향 차이가 줄어든다 (표준편차 {d0:.2f} → {d1:.2f})"
                  if d1 < d0 else f"이 설정에서는 성향 차이가 줄지 않는다 (표준편차 {d0:.2f} → {d1:.2f})")
        self.show_panel("⑤ 모든 노드가 동시에: 한 층(layer)의 메시지 패싱",
                        [r"a_i=\sum_{j\in N(i)\cup\{i\}}\frac{x_j}{\sqrt{\tilde d_i\,\tilde d_j}}"
                         r"\quad\text{for all } i"], note=smooth)
        dots, paths = [], []
        for i, j in r["edges"]:
            for a, b in [(i, j), (j, i)]:
                dots.append(Dot(self.pos(a), radius=0.08, color=self.tone(X[a])))
                paths.append(Line(self.pos(a), self.pos(b)))
        for i in range(n):
            dots.append(Dot(self.pos(i), radius=0.08, color=self.tone(X[i])))
            paths.append(Circle(radius=self.R + 0.1).move_to(self.pos(i)).rotate(PI / 2))
        self.play(*[MoveAlongPath(d, p) for d, p in zip(dots, paths)], run_time=1.6)
        self.play(*[FadeOut(d) for d in dots],
                  *[self.circles[i].animate.set_fill(self.tone(AX[i])) for i in range(n)],
                  *[Transform(self.feat[i], self.feat_label(i, AX[i], WHITE)) for i in range(n)])
        self.wait(1.2)

        zeroed = [nm[i] for i in range(n) if (Z[i] < 0).any()]
        self.show_panel("⑤ 모든 노드가 동시에: 한 층(layer)의 메시지 패싱",
                        [r"h_i'=\mathrm{ReLU}(a_iW)\quad\text{for all } i"],
                        note=(f"ReLU가 음수 성분을 0으로 바꾼 노드: {', '.join(zeroed)}" if zeroed
                              else "음수 성분이 없어 ReLU가 바꾼 값이 없다"))
        self.play(*[Transform(self.feat[i], self.feat_label(i, H[i], OK_C)) for i in range(n)],
                  *[self.circles[i].animate.set_fill(self.tone(H[i])) for i in range(n)])
        self.wait(0.8)

        # ======================= ⑥ 행렬 형태 =======================
        self.show_panel(
            "⑥ 행렬로 쓰면 한 줄: 모든 노드의 계산을 한 번에",
            [r"H'=\mathrm{ReLU}(\hat A X W),\qquad \hat A=\tilde D^{-1/2}\,\tilde A\,\tilde D^{-1/2}"],
            note="Â에서 0이 아닌 칸 = 이웃이거나 자기 자신 · 칸의 값 = 메시지 계수")
        hm_title = VGroup(Text("정규화 인접행렬", font=KO_FONT, font_size=18, color=GREY_A),
                          MathTex(r"\hat A", font_size=30, color=HM)).arrange(RIGHT, buff=0.12)
        hm_title.move_to([self.HM_X[2], 2.85, 0])
        col_lbl = VGroup(*[self.fit(Text(nm[j], font=KO_FONT, font_size=18), self.CELL)
                           .move_to([self.HM_X[j], self.HM_Y[0] + 0.45, 0]) for j in range(n)])
        row_lbl = VGroup(*[self.fit(Text(nm[i], font=KO_FONT, font_size=18), 0.6)
                           .move_to([self.HM_X[0] - 0.62, self.HM_Y[i], 0]) for i in range(n)])
        self.hm = self.heatmap(Ahat)
        out_head = VGroup(Text("출력", font=KO_FONT, font_size=18), MathTex("h'", font_size=26)
                          ).arrange(RIGHT, buff=0.08).move_to([self.OUT_X, self.HM_Y[0] + 0.45, 0])
        out_vals = VGroup(*[MathTex(vec(H[i]), font_size=24, color=OK_C)
                            .move_to([self.OUT_X, self.HM_Y[i], 0]) for i in range(n)])
        self.play(FadeIn(hm_title), FadeIn(col_lbl), FadeIn(row_lbl), FadeIn(self.hm),
                  FadeIn(out_head), LaggedStart(*[FadeIn(v) for v in out_vals], lag_ratio=0.1))
        self.wait(1.2)

        # ======================= ⑦ 수용 영역 =======================
        s, hops = r["rf_node"], r["rf_hops"]
        groups = {}
        for i, h in enumerate(hops):
            groups.setdefault(h, []).append(nm[i])
        parts = [f"{k}층 → {', '.join(groups[k])}" for k in sorted(h for h in groups if h is not None and h > 0)]
        if None in groups:
            parts.append(f"닿지 않음 → {', '.join(groups[None])}")
        self.show_panel(
            f"⑦ 층을 쌓으면: 한 층마다 정보가 한 칸씩 더 멀리 퍼진다 ({nm[s]} 기준)",
            [r"(\hat A^{k})_{ij}\neq 0\ \Longleftrightarrow\ \mathrm{dist}(i,j)\le k"],
            note="   ".join(parts) + "   ·   흰색 = 기준, 초록 = 1칸, 노랑 = 2칸, 빨강 = 3칸 이상")
        self.play(*[self.circles[i].animate.set_stroke(HOP_COLORS.get(h, RED) if h is not None else GREY_D,
                                                       width=6) for i, h in enumerate(hops)])
        row_box = SurroundingRectangle(VGroup(*[self.hm[s * n + j] for j in range(n)]),
                                       color=WHITE, buff=0.03)
        self.play(Create(row_box))
        self.wait(0.6)
        A2 = r["A2"]
        new_title = VGroup(Text("2층의 연결", font=KO_FONT, font_size=18, color=GREY_A),
                           MathTex(r"\hat A^{2}", font_size=30, color=HM)).arrange(RIGHT, buff=0.12
                                                                                ).move_to(hm_title)
        self.play(Transform(self.hm, self.heatmap(A2)), Transform(hm_title, new_title), run_time=1.2)
        new_cells = [j for j in range(n) if abs(A2[s][j]) > 1e-12 and abs(Ahat[s][j]) <= 1e-12]
        if new_cells:
            boxes = VGroup(*[SurroundingRectangle(self.hm[s * n + j], color=YELLOW, buff=0.02,
                                                  stroke_width=5) for j in new_cells])
            self.play(Create(boxes))
            self.play(*[Indicate(self.circles[j], color=YELLOW) for j in new_cells])
        self.wait(1.2)

        # ======================= ⑧ 검증 =======================
        inv_max = max(r["inv"].values())
        self.show_panel(
            f"⑧ 검증: 독립 구현 대조 + 그래프 불변식 + 무작위 그래프 {r['n_random']}개",
            [r"\max(|\hat AX-\mathrm{ref}|,\ |H'-\mathrm{ref}|)=" + sci(r["max_ref"]),
             r"\hat A=\hat A^{\top},\ \ \hat A\sqrt{\tilde d}=\sqrt{\tilde d},\ \ |\lambda|\le 1,"
             r"\ \ \mathrm{GCN}(PX,PAP^{\top})=P\,\mathrm{GCN}(X,A)"],
            note=f"불변식 최대 오차 {inv_max:.1e}  ·  수용 영역 불일치 0건", gap=0.7)
        refs = {"loop": "순수 파이썬", "sparse": "scipy.sparse", "torch": "PyTorch"}
        used = [refs[k] for k, v in r["diffs"].items() if v is not None]
        badge = VGroup(Text("검증 통과", font=KO_FONT, font_size=20, color=OK_C, weight=BOLD),
                       Text("· " + " · ".join(used) + " 구현과 일치 · 불변식 5종",
                            font=KO_FONT, font_size=15, color=GREY_B)).arrange(RIGHT, buff=0.12)
        box = SurroundingRectangle(badge, color=OK_C, buff=0.1, corner_radius=0.1)
        VGroup(badge, box).move_to([3.6, -1.2, 0])
        self.play(FadeIn(badge, scale=0.9), Create(box))
        self.wait(2.5)

    # ------------------------------------------------------------------ helpers
    def node_tag(self, i):
        return str(i + 1) if self.use_numbers else self.nm[i]

    @staticmethod
    def fit(m, w):
        return m.scale_to_fit_width(w) if m.width > w else m

    def feat_label(self, i, v, color):
        lab = MathTex(vec(v, 2), font_size=22, color=color)
        p = self.pos(i)
        if p[1] >= self.CENTER[1]:
            lab.next_to(p + UP * (self.R + 0.1), UP, buff=0.08)
        else:
            lab.next_to(p + DOWN * (self.R + 0.1), DOWN, buff=0.08)
        return lab

    def legend(self, axes):
        bar = VGroup(*[Rectangle(width=0.16, height=0.16, stroke_width=0,
                                 fill_color=interpolate_color(C1, C2, k / 9), fill_opacity=1)
                       for k in range(10)]).arrange(RIGHT, buff=0)
        g = VGroup(Text(axes[0], font=KO_FONT, font_size=16, color=C1), bar,
                   Text(axes[1], font=KO_FONT, font_size=16, color=C2)).arrange(RIGHT, buff=0.12)
        return g.move_to([self.CENTER[0], -1.45, 0])

    def heatmap(self, M):
        n = len(M)
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                v = float(M[i][j])
                rect = Square(side_length=self.CELL, stroke_color=GREY_D, stroke_width=1.2,
                              fill_color=interpolate_color(GREY_E, HM, min(1.0, abs(v) / 0.6)),
                              fill_opacity=0.25 + 0.75 * min(1.0, abs(v) / 0.6) if abs(v) > 1e-12 else 0.15)
                rect.move_to([self.HM_X[j], self.HM_Y[i], 0])
                txt = MathTex("0" if abs(v) <= 1e-12 else f"{v:.2f}", font_size=18,
                              color=GREY_D if abs(v) <= 1e-12 else WHITE).move_to(rect)
                cells.add(VGroup(rect, txt))
        return cells

    def show_panel(self, caption, tex_lines, note=None, gap=0.62):
        cap = Text(caption, font=KO_FONT, font_size=22, color=GREY_A).move_to([0, -1.85, 0])
        if cap.width > 13.4:
            cap.scale_to_fit_width(13.4)
        lines = VGroup(*[MathTex(t, font_size=28) for t in tex_lines])
        lines.arrange(DOWN, buff=gap - 0.45, aligned_edge=LEFT)
        lines.next_to(cap, DOWN, buff=0.22)
        if lines.width > 13.4:
            lines.scale_to_fit_width(13.4)
        parts = [cap, lines]
        if note:
            nt = Text(note, font=KO_FONT, font_size=19, color=YELLOW_B)
            if nt.width > 13.4:
                nt.scale_to_fit_width(13.4)
            parts.append(nt.next_to(lines, DOWN, buff=0.16))
        new = VGroup(*parts)
        if self.panel is None:
            self.play(FadeIn(new, shift=UP * 0.1), run_time=0.7)
        else:
            self.play(FadeOut(self.panel), FadeIn(new, shift=UP * 0.1), run_time=0.7)
        self.panel = new
        self.wait(1.2)
