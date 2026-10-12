"""
================================================================================
attention_demo.py · 셀프 어텐션(Self-Attention) 시각화 — 특집호 Demo 8
================================================================================

구성
  attention_core.py  : 계산 + 검증 (독립 구현 대조, 불변식, 무작위 입력 20세트)
  attention_demo.py  : (이 파일) 검증을 통과한 숫자만 받아서 그림

장면 흐름 (약 80초)
  ① 임베딩      : 단어 → 숫자 벡터
  ② Q · K · V   : 같은 단어를 세 가지 역할로 투영
  ③ 유사도 점수 : '먹었다'의 질문 벡터와 각 키의 내적 / √d_k
  ④ softmax     : 합이 1인 가중치 → 연결선 굵기 = 가중치
  ⑤ 출력        : 가중치로 값(Value)을 섞어 새 표현 생성
  ⑥ 전체 행렬   : 모든 단어가 같은 계산을 한 번에
  ⑦ 검증        : 통과하지 못하면 렌더링 자체를 중단

실행 (두 파일을 같은 폴더에 두고)
  conda activate 3Blue1Brown
  python attention_core.py --inject              # 검증 리포트 + 오류 주입 실험
  manim -pql attention_demo.py SelfAttention     # 저화질 미리보기

한글 규칙 : 한글은 항상 Text(font=KO_FONT), 수식은 MathTex — MathTex 안에 한글 금지
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manim import *
from attention_core import run

KO_FONT = os.environ.get("KO_FONT", "Malgun Gothic")

Q_C, K_C, V_C = BLUE, GREEN, PURPLE_B
ATT = ORANGE
OK_C = GREEN


def sci(v):
    """0.0 → '0', 그 외 → 'a\\times10^{b}' (LaTeX)."""
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


class SelfAttention(Scene):

    TOK_X = [-5.4, -3.4, -1.4]
    KEY_Y, QRY_Y = 1.85, -0.45
    BOX_W, BOX_H = 1.55, 0.6
    CELL_W, CELL_H = 0.85, 0.57
    HM_X = [2.35, 3.20, 4.05]
    HM_Y = [0.30, -0.27, -0.84]

    def construct(self):
        # ============ 0. 계산 + 검증 ============
        r = run()
        if not r["passed"]:
            raise RuntimeError(
                f"어텐션 검증 실패 (max diff = {r['max_ref']:.2e}, "
                f"row-sum err = {r['row_sum_err']:.2e}) — 렌더링하지 않습니다."
            )
        self.r = r
        tok, A, Z, Q, K, V = r["tokens"], r["A"], r["Z"], r["Q"], r["K"], r["V"]
        fq = r["focus"]
        n = len(tok)
        self.panel = None

        # ============ 1. 제목 ============
        title = Text("셀프 어텐션: 각 단어가 다른 단어를 얼마나 참고하는가",
                     font=KO_FONT, font_size=30).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ============ 2. 토큰 (아래 = 질문 Q, 위 = 참고 대상 K) ============
        def token_box(i, y, color):
            box = RoundedRectangle(width=self.BOX_W, height=self.BOX_H, corner_radius=0.1,
                                   color=color, stroke_width=2.5)
            label = Text(tok[i], font=KO_FONT, font_size=22)
            if label.width > self.BOX_W - 0.2:          # 사용자가 긴 단어를 넣어도 상자 안에
                label.scale_to_fit_width(self.BOX_W - 0.2)
            return VGroup(box, label).move_to([self.TOK_X[i], y, 0])

        q_boxes = VGroup(*[token_box(i, self.QRY_Y, Q_C) for i in range(n)])
        k_boxes = VGroup(*[token_box(i, self.KEY_Y, K_C) for i in range(n)])
        q_tag = MathTex("Q", font_size=30, color=Q_C).move_to([-6.75, self.QRY_Y, 0])
        k_tag = MathTex("K", font_size=30, color=K_C).move_to([-6.75, self.KEY_Y, 0])
        emb = VGroup(*[
            MathTex(rf"({r['X'][i][0]:.1f},\ {r['X'][i][1]:.1f})", font_size=22, color=GREY_A)
            .next_to(q_boxes[i], DOWN, buff=0.12) for i in range(n)])

        cap1 = "① 임베딩: 각 단어를 숫자 벡터로 표현 (2차원 장난감 예제)"
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in q_boxes], lag_ratio=0.2),
                  FadeIn(q_tag))
        self.show_panel(cap1, [
            ",\\quad ".join(rf"x_{i + 1}=({r['X'][i][0]:.1f},\ {r['X'][i][1]:.1f})"
                            for i in range(n))],
            note=f"축의 의미 = ({r['axis_names'][0]}, {r['axis_names'][1]})   ·   " +
                 "   ".join(f"{i + 1} {t}" for i, t in enumerate(tok)))
        self.play(LaggedStart(*[FadeIn(e) for e in emb], lag_ratio=0.2))
        self.play(*[TransformFromCopy(q_boxes[i], k_boxes[i]) for i in range(n)], FadeIn(k_tag))
        self.wait(0.5)

        # ============ ② Q · K · V ============
        def mat(M, label, color, d=2):
            m = Matrix([[f"{v:.{d}f}" for v in row] for row in M],
                       element_to_mobject_config={"font_size": 28},
                       h_buff=0.88 if d == 2 else 1.15,   # 소수 3자리·음수가 붙지 않도록
                       v_buff=0.5, bracket_h_buff=0.1)
            lab = MathTex(label + "=", font_size=28, color=color)
            g = VGroup(lab, m).arrange(RIGHT, buff=0.1)
            if g.width > 2.15:                       # 화면 폭을 넘지 않도록 상한
                g.scale_to_fit_width(2.15)
            return g

        mQ, mK, mV = mat(Q, "Q", Q_C), mat(K, "K", K_C), mat(V, "V", V_C)
        VGroup(mQ, mK, mV).arrange(RIGHT, buff=0.2).move_to([3.5, 1.95, 0])
        def bmat(M):
            return (r"\begin{bmatrix}" + rf"{M[0][0]:.2f} & {M[0][1]:.2f}\\ {M[1][0]:.2f} & {M[1][1]:.2f}"
                    + r"\end{bmatrix}")

        kv_identity = np.allclose(r["WK"], np.eye(2)) and np.allclose(r["WV"], np.eye(2))
        self.show_panel(
            "② 세 가지 역할: Query(찾는 것) · Key(꼬리표) · Value(건네줄 내용)",
            [r"Q=XW_Q,\ \ K=XW_K,\ \ V=XW_V,\qquad W_Q=" + bmat(r["WQ"]) +
             (r",\ \ W_K=W_V=I" if kv_identity else
              r",\ W_K=" + bmat(r["WK"]) + r",\ W_V=" + bmat(r["WV"]))],
            note=("이 예제에서는 키와 값을 임베딩 그대로 사용 (설명 단순화)" if kv_identity
                  else "키와 값도 각자의 가중치로 투영"))
        self.play(FadeIn(mQ, shift=DOWN * 0.1), FadeIn(mK, shift=DOWN * 0.1),
                  FadeIn(mV, shift=DOWN * 0.1), run_time=1.2)
        self.wait(0.8)

        # ============ ③ 유사도 점수 ============
        dots = Q[fq] @ K.T
        s = r["S"][fq]
        self.play(q_boxes[fq][0].animate.set_fill(Q_C, opacity=0.35))
        q_row_box = SurroundingRectangle(mQ[1].get_rows()[fq], color=Q_C, buff=0.06)
        k_all_box = SurroundingRectangle(mK[1].get_entries(), color=K_C, buff=0.08)
        self.show_panel(
            f"③ 유사도 점수: '{tok[fq]}'의 질문 벡터와 각 단어 키의 내적",
            [rf"s_j=\frac{{q_{fq + 1}\cdot k_j}}{{\sqrt{{d_k}}}}",
             rf"s=\tfrac{{1}}{{\sqrt{{{r['dk']}}}}}(" + ",\\ ".join(f"{d:.2f}" for d in dots)
             + ")=(" + ",\\ ".join(f"{v:.3f}" for v in s) + ")"], gap=0.75)
        self.play(Create(q_row_box), Create(k_all_box))
        self.wait(0.8)

        # ============ 히트맵 틀 + Z 틀 ============
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                cells.add(Rectangle(width=self.CELL_W, height=self.CELL_H,
                                    stroke_color=GREY_D, stroke_width=1.5,
                                    fill_color=GREY_E, fill_opacity=0.25)
                          .move_to([self.HM_X[j], self.HM_Y[i], 0]))
        def fit(m, w):
            return m.scale_to_fit_width(w) if m.width > w else m

        col_lbl = VGroup(*[fit(Text(tok[j], font=KO_FONT, font_size=15, color=K_C), self.CELL_W - 0.06)
                           .move_to([self.HM_X[j], self.HM_Y[0] + 0.48, 0]) for j in range(n)])
        row_lbl = VGroup(*[fit(Text(tok[i], font=KO_FONT, font_size=15, color=Q_C), 0.85)
                           .next_to(cells[i * n], LEFT, buff=0.12) for i in range(n)])
        hm_title = Text("어텐션 가중치 A", font=KO_FONT, font_size=17, color=GREY_A
                        ).move_to([self.HM_X[1], self.HM_Y[0] + 0.85, 0])
        mZ = mat(Z, "Z", ATT, d=3)
        mZ.move_to([0, self.HM_Y[1], 0]).align_to([6.85, 0, 0], RIGHT)
        z_rows = mZ[1].get_rows()
        for row in z_rows:
            row.set_opacity(0)
        self.play(FadeOut(q_row_box), FadeOut(k_all_box),
                  FadeIn(cells), FadeIn(col_lbl), FadeIn(row_lbl), FadeIn(hm_title),
                  FadeIn(mZ[0]), FadeIn(mZ[1].get_brackets()))
        self.add(mZ)
        self.cells = cells

        # ============ ④ softmax → 연결선 ============
        lines = self.att_lines(fq)
        best = int(A[fq].argmax())
        self.show_panel(
            "④ softmax: 점수를 합이 1인 가중치로 바꾸기",
            [rf"a_j=\frac{{e^{{s_j}}}}{{\sum_k e^{{s_k}}}}=(" +
             ",\\ ".join(f"{a:.2f}" for a in A[fq]) + ")"],
            note=f"'{tok[fq]}' → '{tok[best]}' 가중치가 가장 큼 ({A[fq][best]:.2f})   ·   선 굵기 = 가중치")
        self.play(LaggedStart(*[Create(l) for l in lines[0]], lag_ratio=0.25), run_time=1.5)
        self.play(*[FadeIn(lb) for lb in lines[1]])
        self.fill_row(fq)
        self.wait(1.0)

        # ============ ⑤ 출력 ============
        self.show_panel(
            "⑤ 출력: 가중치로 값(Value)을 섞어 새 표현 만들기",
            [rf"z_{fq + 1}=\sum_j a_j v_j=" +
             "+".join(rf"{A[fq][j]:.2f}\,v_{j + 1}" for j in range(n)) +
             rf"=({Z[fq][0]:.3f},\ {Z[fq][1]:.3f})"],
            note=f"'{tok[fq]}'의 새 표현 = 참고한 단어들의 값을 가중 평균한 것")
        self.play(z_rows[fq].animate.set_opacity(1))
        self.play(Indicate(z_rows[fq], color=ATT))
        self.wait(1.0)

        # ============ ⑥ 모든 단어 ============
        self.show_panel(
            "⑥ 모든 단어가 같은 계산을 한 번에 (행렬 연산)",
            [r"Z=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V"],
            note="각 행(질문 단어)마다 가중치의 합은 1")
        cur_box = fq
        for i in [k for k in range(n) if k != fq]:
            new_lines = self.att_lines(i)
            self.play(q_boxes[cur_box][0].animate.set_fill(opacity=0),
                      q_boxes[i][0].animate.set_fill(Q_C, opacity=0.35),
                      Transform(lines[0], new_lines[0]), Transform(lines[1], new_lines[1]),
                      run_time=1.0)
            cur_box = i
            self.fill_row(i)
            self.play(z_rows[i].animate.set_opacity(1), run_time=0.6)
            self.wait(0.6)

        # ============ ⑦ 검증 ============
        self.show_panel(
            f"⑦ 검증: 독립 구현 대조 + 불변식 + 무작위 입력 {r['n_random']}세트",
            [r"\max\left(|A-A_{\mathrm{ref}}|,\ |Z-Z_{\mathrm{ref}}|\right)=" + sci(r["max_ref"]),
             r"\max_i\Big|\sum_j a_{ij}-1\Big|=" + sci(r["row_sum_err"]) + r",\qquad a_{ij}\ge 0"],
            gap=0.75)
        refs = [k for k, v in r["diffs"].items() if v is not None]
        ref_name = {"loop": "순수 파이썬", "scipy": "scipy", "torch": "PyTorch"}
        badge = VGroup(
            Text("검증 통과", font=KO_FONT, font_size=20, color=OK_C, weight=BOLD),
            Text("· " + " · ".join(ref_name[k] for k in refs) + " 구현과 일치",
                 font=KO_FONT, font_size=15, color=GREY_B),
        ).arrange(RIGHT, buff=0.12)
        box = SurroundingRectangle(badge, color=OK_C, buff=0.1, corner_radius=0.1)
        VGroup(badge, box).move_to([3.9, -1.50, 0])
        self.play(FadeIn(badge, scale=0.9), Create(box))
        self.wait(2.5)

    # ------------------------------------------------------------------ helpers
    def att_lines(self, i):
        """질문 단어 i → 모든 키로 가는 연결선 (굵기·불투명도 = 가중치) + 가중치 라벨."""
        A = self.r["A"]
        start = np.array([self.TOK_X[i], self.QRY_Y + self.BOX_H / 2, 0])
        lines, labels = VGroup(), VGroup()
        for j in range(len(self.TOK_X)):
            end = np.array([self.TOK_X[j], self.KEY_Y - self.BOX_H / 2, 0])
            a = float(A[i][j])
            ln = Line(start, end, color=ATT, stroke_width=1.5 + 16 * a,
                      stroke_opacity=0.35 + 0.65 * a)
            lb = MathTex(f"{a:.2f}", font_size=24, color=ATT)
            lb.move_to(ln.point_from_proportion(0.55))
            lb.add_background_rectangle(opacity=0.9, buff=0.05)
            lines.add(ln)
            labels.add(lb)
        return VGroup(lines, labels)

    def fill_row(self, i):
        A = self.r["A"]
        n = len(self.TOK_X)
        anims, texts = [], []
        for j in range(n):
            a = float(A[i][j])
            cell = self.cells[i * n + j]
            anims.append(cell.animate.set_fill(interpolate_color(GREY_E, ATT, a), opacity=0.25 + 0.75 * a))
            t = MathTex(f"{a:.2f}", font_size=22, color=BLACK if a > 0.7 else WHITE).move_to(cell)
            texts.append(t)
        self.play(*anims, *[FadeIn(t) for t in texts], run_time=0.8)

    def show_panel(self, caption, tex_lines, note=None, gap=0.62):
        """하단 설명 패널: 한글 캡션(Text) + 수식(MathTex) + 한글 보충(Text)."""
        cap = Text(caption, font=KO_FONT, font_size=22, color=GREY_A).move_to([0, -1.95, 0])
        lines = VGroup(*[MathTex(t, font_size=30) for t in tex_lines])
        lines.arrange(DOWN, buff=gap - 0.45, aligned_edge=LEFT)
        lines.next_to(cap, DOWN, buff=0.25)
        if lines.width > 13.4:
            lines.scale_to_fit_width(13.4)
        parts = [cap, lines]
        if note:
            parts.append(Text(note, font=KO_FONT, font_size=19, color=YELLOW_B)
                         .next_to(lines, DOWN, buff=0.22))
        new = VGroup(*parts)
        if self.panel is None:
            self.play(FadeIn(new, shift=UP * 0.1), run_time=0.7)
        else:
            self.play(FadeOut(self.panel), FadeIn(new, shift=UP * 0.1), run_time=0.7)
        self.panel = new
        self.wait(1.2)
