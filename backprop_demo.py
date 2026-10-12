"""
================================================================================
backprop_demo.py · 역전파(Backpropagation) 시각화 — 특집호 Demo 7
================================================================================

구성
  backprop_core.py  : NumPy로 순전파·역전파 계산 + 수치미분(중앙 차분) 대조 검증
  backprop_demo.py  : (이 파일) 검증을 통과한 숫자만 받아서 그림  ← 숫자를 직접 쓰지 않음

장면 흐름 (약 80초)
  ① 순전파   : x → h₁, h₂ → ŷ → L  (파란색)
  ② 역전파   : L → ŷ → h₁, h₂ → x  (주황색), 오른쪽 표에 ∂L/∂w 채우기
  ③ 연쇄법칙 : ∂L/∂w₁ 한 경로를 5개 국소 미분의 곱으로 분해
  ④ 검증     : 해석적 기울기 vs 수치미분 최대 오차 표시 (실패하면 렌더링 자체를 중단)
  ⑤ 갱신     : w ← w − η·∂L/∂w, 손실 감소 확인

실행 (두 파일을 같은 폴더에 두고)
  conda activate 3Blue1Brown
  python backprop_core.py                          # 검증 리포트만 먼저 확인
  manim -pql backprop_demo.py BackpropChainRule    # 저화질 미리보기
  manim -pqh backprop_demo.py BackpropChainRule    # 원고·발표용

한글 규칙 : 한글은 항상 Text(font=KO_FONT), 수식은 MathTex — MathTex 안에 한글 금지
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manim import *
from backprop_core import run

# Windows 기본값. 다른 환경(리눅스·Docker)에서는 KO_FONT 환경변수로 교체
KO_FONT = os.environ.get("KO_FONT", "Malgun Gothic")

FWD = BLUE          # 순전파 값
BWD = ORANGE        # 기울기
OK_C = GREEN        # 검증 통과


def g(v):
    """사용자 입력값 표기: 1.0 → '1', 0.25 → '0.25' (자릿수를 임의로 자르지 않음)."""
    return f"{v:g}"


def gp(v):
    """g()의 음수 괄호 버전."""
    return f"({v:g})" if v < 0 else f"{v:g}"


def num(v, d=3):
    """LaTeX용 숫자 문자열. 음수는 괄호로 감싸 곱셈식에서 부호가 헷갈리지 않게 한다."""
    s = f"{v:.{d}f}"
    return f"({s})" if v < 0 else s


class BackpropChainRule(Scene):

    # ------------------------------------------------------------------ layout
    R = 0.40
    POS = {
        "x1": [-6.1, 1.6, 0], "x2": [-6.1, -0.6, 0],
        "h1": [-3.4, 1.6, 0], "h2": [-3.4, -0.6, 0],
        "o":  [-0.7, 0.5, 0], "L":  [1.2, 0.5, 0],
    }
    # 이름: (출발, 도착, 간선 위 위치 t, 법선 방향 ±1) → 라벨 좌표는 기하로 자동 계산
    #   (좌표를 손으로 박아 넣으면 노드 위치를 바꿀 때마다 라벨이 겹치는 문제가 반복됨)
    EDGES = {
        "w1": ("x1", "h1", 0.50, +1),
        "w2": ("x2", "h1", 0.28, -1),
        "w3": ("x1", "h2", 0.28, -1),
        "w4": ("x2", "h2", 0.50, -1),
        "w5": ("h1", "o",  0.50, +1),
        "w6": ("h2", "o",  0.50, -1),
    }
    LABEL_OFF = 0.27

    def edge_label_pos(self, a, b, t, side):
        A, B = np.array(self.POS[a], float), np.array(self.POS[b], float)
        d = (B - A) / np.linalg.norm(B - A)
        n = np.array([-d[1], d[0], 0.0])          # 진행 방향의 왼쪽 법선
        return A + (B - A) * t + side * self.LABEL_OFF * n
    WEIGHTS = ["w1", "w2", "w3", "w4", "w5", "w6"]

    def construct(self):
        # ============ 0. 계산 + 검증 (통과 못 하면 그리지 않는다) ============
        r = run()
        if not r["passed"]:
            raise RuntimeError(
                f"역전파 검증 실패 (max diff = {r['err_num']:.2e}) — "
                "검증되지 않은 수치로는 렌더링하지 않습니다."
            )
        self.r = r
        p, f, gr, it = r["params"], r["fwd"], r["grads"], r["inter"]
        self.panel = None

        # ============ 1. 제목 · 범례 ============
        title = Text("역전파: 연쇄법칙으로 기울기를 거꾸로 전달하기",
                     font=KO_FONT, font_size=32).to_edge(UP, buff=0.3)
        legend = VGroup(
            Text("■ 순전파 값", font=KO_FONT, font_size=18, color=FWD),
            Text("■ 기울기", font=KO_FONT, font_size=18, color=BWD),
        ).arrange(RIGHT, buff=0.35).move_to([4.57, 2.85, 0])
        self.play(Write(title), FadeIn(legend))

        # ============ 2. 네트워크 ============
        nodes = {}
        for k, tex in [("x1", "x_1"), ("x2", "x_2"), ("h1", "h_1"),
                       ("h2", "h_2"), ("o", r"\hat{y}")]:
            c = Circle(radius=self.R, color=GREY_B, stroke_width=3).move_to(self.POS[k])
            nodes[k] = VGroup(c, MathTex(tex, font_size=34).move_to(c))
        loss_box = RoundedRectangle(width=0.9, height=0.8, corner_radius=0.12,
                                    color=GREY_B, stroke_width=3).move_to(self.POS["L"])
        nodes["L"] = VGroup(loss_box, MathTex("L", font_size=34).move_to(loss_box))
        self.nodes = nodes

        edges, elabels = {}, {}
        for name, (a, b, t, side) in self.EDGES.items():
            lpos = self.edge_label_pos(a, b, t, side)
            edges[name] = Line(self.POS[a], self.POS[b], buff=self.R,
                               color=GREY_C, stroke_width=2.5)
            elabels[name] = MathTex(name[0] + "_" + name[1:], font_size=24,
                                    color=GREY_A).move_to(lpos)
        edges["oL"] = Arrow(nodes["o"][0].get_right(), loss_box.get_left(),
                            buff=0.05, color=GREY_C, stroke_width=2.5,
                            max_tip_length_to_length_ratio=0.15)
        self.edges = edges

        self.play(LaggedStart(*[Create(n) for n in nodes.values()], lag_ratio=0.12))
        self.play(*[Create(e) for e in edges.values()],
                  *[FadeIn(l) for l in elabels.values()], run_time=1.2)

        # ============ 3. 오른쪽 파라미터 표 ============
        cols_x = [2.95, 3.90, 5.05, 6.15]
        row_y = [2.05, 1.55, 1.10, 0.65, 0.20, -0.25, -0.70]
        header = VGroup(
            Text("가중치", font=KO_FONT, font_size=18).move_to([cols_x[0], row_y[0], 0]),
            Text("값", font=KO_FONT, font_size=18).move_to([cols_x[1], row_y[0], 0]),
            MathTex(r"\partial L/\partial w", font_size=24, color=BWD
                    ).move_to([cols_x[2], row_y[0], 0]),
            Text("갱신 후", font=KO_FONT, font_size=18).move_to([cols_x[3], row_y[0], 0]),
        )
        bg = RoundedRectangle(width=4.1, height=3.35, corner_radius=0.15,
                              stroke_color=GREY_D, stroke_width=1.5,
                              fill_color=GREY_E, fill_opacity=0.35
                              ).move_to([4.57, 0.67, 0])
        rule = Line([2.60, 1.80, 0], [6.55, 1.80, 0], color=GREY_D, stroke_width=1.5)
        table_rows = VGroup()
        for i, w in enumerate(self.WEIGHTS):
            y = row_y[i + 1]
            table_rows.add(VGroup(
                MathTex(w[0] + "_" + w[1:], font_size=26).move_to([cols_x[0], y, 0]),
                MathTex(f"{p[w]:.2f}", font_size=26).move_to([cols_x[1], y, 0]),
            ))
        self.cols_x, self.row_y = cols_x, row_y
        self.play(FadeIn(bg), FadeIn(header), Create(rule),
                  LaggedStart(*[FadeIn(row) for row in table_rows], lag_ratio=0.1))
        self.wait(0.5)

        # ======================= ① 순전파 =======================
        cap_fwd = "① 순전파(Forward): 입력에서 출력 방향으로 값을 계산"
        x_lbl1 = MathTex(f"x_1={g(f['x1'])}", font_size=26, color=FWD).next_to(nodes["x1"], UP, buff=0.12)
        x_lbl2 = MathTex(f"x_2={g(f['x2'])}", font_size=26, color=FWD).next_to(nodes["x2"], DOWN, buff=0.12)
        self.show_panel(cap_fwd, [r"\text{input: } x=(x_1, x_2)"
                                  + f"=({g(f['x1'])},\\ {g(f['x2'])})"])
        self.play(FadeIn(x_lbl1, shift=DOWN * 0.1), FadeIn(x_lbl2, shift=UP * 0.1))

        # h1
        self.flash(["w1", "w2"], FWD)
        self.show_panel(cap_fwd, [
            r"h_1=\sigma(w_1x_1+w_2x_2+b_1)",
            rf"=\sigma({num(p['w1'],2)}\cdot{gp(f['x1'])}+{num(p['w2'],2)}\cdot{gp(f['x2'])}"
            rf"+{num(p['b1'],2)})=\sigma({f['z1']:.3f})={f['h1']:.3f}",
        ])
        h1_lbl = MathTex(f"h_1={f['h1']:.3f}", font_size=26, color=FWD).next_to(nodes["h1"], UP, buff=0.12)
        self.play(FadeIn(h1_lbl, shift=DOWN * 0.1), nodes["h1"][0].animate.set_color(FWD))

        # h2
        self.flash(["w3", "w4"], FWD)
        self.show_panel(cap_fwd, [
            r"h_2=\sigma(w_3x_1+w_4x_2+b_2)",
            rf"=\sigma({num(p['w3'],2)}\cdot{gp(f['x1'])}+{num(p['w4'],2)}\cdot{gp(f['x2'])}"
            rf"+{num(p['b2'],2)})=\sigma({f['z2']:.3f})={f['h2']:.3f}",
        ])
        h2_lbl = MathTex(f"h_2={f['h2']:.3f}", font_size=26, color=FWD).next_to(nodes["h2"], DOWN, buff=0.12)
        self.play(FadeIn(h2_lbl, shift=UP * 0.1), nodes["h2"][0].animate.set_color(FWD))

        # y-hat
        self.flash(["w5", "w6"], FWD)
        self.show_panel(cap_fwd, [
            r"\hat{y}=\sigma(w_5h_1+w_6h_2+b_3)",
            rf"=\sigma({num(p['w5'],2)}\cdot{f['h1']:.3f}+{num(p['w6'],2)}\cdot{f['h2']:.3f}"
            rf"+{num(p['b3'],2)})=\sigma({f['zo']:.3f})={f['yhat']:.3f}",
        ])
        yh_lbl = MathTex(rf"\hat{{y}}={f['yhat']:.3f}", font_size=26, color=FWD
                         ).next_to(nodes["o"], UP, buff=0.12)
        self.play(FadeIn(yh_lbl, shift=DOWN * 0.1), nodes["o"][0].animate.set_color(FWD))

        # loss
        self.play(ShowPassingFlash(edges["oL"].copy().set_color(FWD).set_stroke(width=6),
                                   time_width=0.6), run_time=0.8)
        self.show_panel(cap_fwd, [
            rf"L=\tfrac{{1}}{{2}}(\hat{{y}}-y)^2=\tfrac{{1}}{{2}}({f['yhat']:.3f}-{gp(f['y'])})^2"
            rf"={f['loss']:.4f}\qquad (y={g(f['y'])})",
        ])
        L_lbl = MathTex(f"L={f['loss']:.4f}", font_size=26, color=FWD
                        ).next_to(nodes["L"], UP, buff=0.12)
        self.play(FadeIn(L_lbl, shift=DOWN * 0.1), nodes["L"][0].animate.set_color(FWD))
        self.wait(0.8)

        # ======================= ② 역전파 =======================
        cap_bwd = "② 역전파(Backward): 손실에서 입력 방향으로 기울기를 전달"
        self.show_panel(cap_bwd, [
            rf"\frac{{\partial L}}{{\partial \hat{{y}}}}=\hat{{y}}-y"
            rf"={f['yhat']:.3f}-{gp(f['y'])}={it['dL_dyhat']:.3f}",
        ])
        gL_lbl = MathTex(rf"\partial L/\partial\hat{{y}}={it['dL_dyhat']:.3f}",
                         font_size=24, color=BWD).next_to(nodes["L"], DOWN, buff=0.12)
        self.play(nodes["L"][0].animate.set_color(BWD), FadeIn(gL_lbl, shift=UP * 0.1))
        self.back_flash(["oL"])

        # delta_o
        self.show_panel(cap_bwd, [
            r"\delta_o=\frac{\partial L}{\partial z_o}"
            r"=\frac{\partial L}{\partial \hat{y}}\cdot\hat{y}(1-\hat{y})",
            rf"={num(it['dL_dyhat'])}\cdot{f['yhat']:.3f}(1-{f['yhat']:.3f})={it['delta_o']:.4f}",
        ])
        do_lbl = MathTex(rf"\delta_o={it['delta_o']:.4f}", font_size=24, color=BWD
                         ).next_to(nodes["o"], DOWN, buff=0.12)
        self.play(nodes["o"][0].animate.set_color(BWD), FadeIn(do_lbl, shift=UP * 0.1))

        # w5, w6
        self.back_flash(["w5", "w6"])
        self.show_panel(cap_bwd, [
            rf"\frac{{\partial L}}{{\partial w_5}}=\delta_o h_1={num(it['delta_o'],4)}\cdot{f['h1']:.3f}"
            rf"={gr['w5']:.4f}",
            rf"\frac{{\partial L}}{{\partial w_6}}=\delta_o h_2={num(it['delta_o'],4)}\cdot{f['h2']:.3f}"
            rf"={gr['w6']:.4f}",
        ])
        self.fill_grads(["w5", "w6"])

        # delta_1, delta_2
        self.show_panel(cap_bwd, [
            rf"\delta_1=\delta_o\,w_5\,h_1(1-h_1)={num(it['delta_o'],4)}\cdot{num(p['w5'],2)}"
            rf"\cdot{f['h1']:.3f}(1-{f['h1']:.3f})={it['delta_1']:.4f}",
            rf"\delta_2=\delta_o\,w_6\,h_2(1-h_2)={num(it['delta_o'],4)}\cdot{num(p['w6'],2)}"
            rf"\cdot{f['h2']:.3f}(1-{f['h2']:.3f})={it['delta_2']:.4f}",
        ])
        d1_lbl = MathTex(rf"\delta_1={it['delta_1']:.4f}", font_size=24, color=BWD
                         ).next_to(nodes["h1"], DOWN, buff=0.12).shift(RIGHT * 0.15)
        d2_lbl = MathTex(rf"\delta_2={it['delta_2']:.4f}", font_size=24, color=BWD
                         ).next_to(nodes["h2"], UP, buff=0.12).shift(RIGHT * 0.15)
        self.play(nodes["h1"][0].animate.set_color(BWD), nodes["h2"][0].animate.set_color(BWD),
                  FadeIn(d1_lbl, shift=UP * 0.1), FadeIn(d2_lbl, shift=DOWN * 0.1))
        # 설명 문장도 계산 결과에서 만든다: δ_h = δ_o · w · σ'(z), σ' > 0 이므로
        # sign(δ₁)·sign(δ₂) = sign(w₅)·sign(w₆). 부호가 다를 때만 보여준다.
        #   (고정 문구로 두면 사용자가 w 값을 바꿨을 때 '숫자는 맞고 설명은 틀린' 영상이 됨)
        if p["w5"] * p["w6"] < 0:
            neg_w = "w_6" if p["w6"] < 0 else "w_5"
            # [표현 수정] 수식과 한글 사이 간격을 직접 정한다: 조사는 붙이고(δ₁과, δ₂의), 단어 사이는 띄움
            Y = dict(color=YELLOW)
            parts = [(MathTex(neg_w + "<0", font_size=26, **Y), 0.0),
                     (Text("이므로", font=KO_FONT, font_size=20, **Y), 0.05),
                     (MathTex(r"\delta_1", font_size=26, **Y), 0.16),
                     (Text("과", font=KO_FONT, font_size=20, **Y), 0.04),
                     (MathTex(r"\delta_2", font_size=26, **Y), 0.16),
                     (Text("의 부호가 서로 반대", font=KO_FONT, font_size=20, **Y), 0.04)]
            note = VGroup(parts[0][0])
            for m, gap in parts[1:]:
                m.next_to(note, RIGHT, buff=gap)
                if isinstance(m, Text):                  # 한글은 수식 기준선에 맞춤
                    m.align_to(parts[0][0], DOWN).shift(DOWN * 0.02)
                note.add(m)
            # [위치 수정] 가운데 정렬하면 문장이 길 때 왼쪽 'h₂ = …' 라벨을 덮는다 → h₂ 라벨 오른쪽에서 시작
            note.move_to([0, -1.30, 0]).align_to([-2.65, 0, 0], LEFT)
            self.play(FadeIn(note), Indicate(d1_lbl, color=YELLOW), Indicate(d2_lbl, color=YELLOW))
            self.wait(1.2)
            self.play(FadeOut(note))

        # w1..w4
        self.back_flash(["w1", "w2", "w3", "w4"])
        self.show_panel(cap_bwd, [
            rf"\frac{{\partial L}}{{\partial w_1}}=\delta_1x_1={gr['w1']:.4f},\quad"
            rf"\frac{{\partial L}}{{\partial w_2}}=\delta_1x_2={gr['w2']:.4f}",
            rf"\frac{{\partial L}}{{\partial w_3}}=\delta_2x_1={gr['w3']:.4f},\quad"
            rf"\frac{{\partial L}}{{\partial w_4}}=\delta_2x_2={gr['w4']:.4f}",
        ])
        self.fill_grads(["w1", "w2", "w3", "w4"])
        self.wait(1.0)

        # ======================= ③ 연쇄법칙 한 경로 =======================
        cap_chain = "③ 연쇄법칙: 한 경로 위의 국소 미분을 모두 곱하면 기울기"
        path = VGroup(
            Line(self.POS["L"], self.POS["o"], buff=0.45),
            Line(self.POS["o"], self.POS["h1"], buff=self.R),
            Line(self.POS["h1"], self.POS["x1"], buff=self.R),
        ).set_color(BWD).set_stroke(width=8, opacity=0.55)
        self.show_panel(cap_chain, [
            r"\frac{\partial L}{\partial w_1}"
            r"=\frac{\partial L}{\partial \hat{y}}\cdot\frac{\partial \hat{y}}{\partial z_o}"
            r"\cdot\frac{\partial z_o}{\partial h_1}\cdot\frac{\partial h_1}{\partial z_1}"
            r"\cdot\frac{\partial z_1}{\partial w_1}",
            rf"={num(it['dL_dyhat'])}\cdot{it['dyhat_dzo']:.3f}\cdot{num(p['w5'],2)}"
            rf"\cdot{it['dh1_dz1']:.3f}\cdot{gp(f['x1'])}={gr['w1']:.4f}",
        ], gap=0.75)
        self.play(Create(path), run_time=1.5)
        self.play(Indicate(self.grad_cells["w1"], color=YELLOW, scale_factor=1.3))
        self.wait(1.5)
        self.play(FadeOut(path))

        # ======================= ④ 검증 =======================
        cap_ver = f"④ 검증: 수치 미분(중앙 차분)과 대조 · 데모 값 + 무작위 입력 {r['n_random']}세트"
        exp = f"{r['err_num']:.1e}".split("e")
        err_tex = rf"{exp[0]}\times10^{{{int(exp[1])}}}"
        self.show_panel(cap_ver, [
            r"\nabla_{\text{numeric}}=\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon},\quad"
            r"\epsilon=10^{-6}",
            r"\max\left|\nabla_{\text{backprop}}-\nabla_{\text{numeric}}\right|="
            + err_tex,
        ])
        badge = VGroup(
            Text("검증 통과", font=KO_FONT, font_size=22, color=OK_C, weight=BOLD),
            Text("· 편향 포함 9개 파라미터", font=KO_FONT, font_size=17, color=GREY_B),
        ).arrange(RIGHT, buff=0.15)
        box = SurroundingRectangle(badge, color=OK_C, buff=0.12, corner_radius=0.1)
        VGroup(badge, box).move_to([4.57, -1.36, 0])
        self.play(FadeIn(badge, scale=0.9), Create(box))
        self.wait(1.2)

        # ======================= ⑤ 갱신 =======================
        cap_upd = f"⑤ 경사하강 갱신: 기울기의 반대 방향으로 한 걸음 (η = {r['lr']:g})"
        fn = r["fwd_new"]
        self.show_panel(cap_upd, [
            r"w\leftarrow w-\eta\,\frac{\partial L}{\partial w}",
            rf"L:\ {f['loss']:.4f}\ \rightarrow\ {fn['loss']:.4f},\qquad"
            rf"\hat{{y}}:\ {f['yhat']:.3f}\ \rightarrow\ {fn['yhat']:.3f}",
        ])
        new_cells = []
        for i, w in enumerate(self.WEIGHTS):
            new_cells.append(MathTex(f"{r['params_new'][w]:.3f}", font_size=26, color=OK_C)
                             .move_to([self.cols_x[3], self.row_y[i + 1], 0]))
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in new_cells],
                              lag_ratio=0.12))
        L_new = MathTex(f"L={fn['loss']:.4f}", font_size=26, color=OK_C).move_to(L_lbl)
        yh_new = MathTex(rf"\hat{{y}}={fn['yhat']:.3f}", font_size=26, color=OK_C).move_to(yh_lbl)
        self.play(Transform(L_lbl, L_new), Transform(yh_lbl, yh_new))
        self.play(Indicate(L_lbl, color=OK_C))
        self.wait(2.5)

    # ------------------------------------------------------------------ helpers
    def show_panel(self, caption, tex_lines, gap=0.62):
        """하단 설명 패널 교체: 한글 캡션(Text) + 수식(MathTex) — 둘을 섞지 않는다."""
        cap = Text(caption, font=KO_FONT, font_size=22, color=GREY_A).move_to([0, -1.95, 0])
        lines = VGroup(*[MathTex(t, font_size=30) for t in tex_lines])
        lines.arrange(DOWN, buff=gap - 0.45, aligned_edge=LEFT)
        lines.next_to(cap, DOWN, buff=0.28)
        if lines.width > 13.4:
            lines.scale_to_fit_width(13.4)
        new = VGroup(cap, lines)
        if self.panel is None:
            self.play(FadeIn(new, shift=UP * 0.1), run_time=0.7)
        elif self.panel[0].text == cap.text:          # 캡션이 같으면 수식만 교체
            self.play(FadeOut(self.panel[1]), FadeIn(lines, shift=UP * 0.1), run_time=0.7)
            self.remove(self.panel[0])
            self.add(cap)
        else:
            self.play(FadeOut(self.panel), FadeIn(new, shift=UP * 0.1), run_time=0.7)
        self.panel = new
        self.wait(1.2)

    def flash(self, names, color):
        self.play(*[ShowPassingFlash(self.edges[n].copy().set_color(color).set_stroke(width=6),
                                     time_width=0.6) for n in names], run_time=0.9)

    def back_flash(self, names):
        anims = []
        for n in names:
            e = self.edges[n]
            rev = Line(e.get_end(), e.get_start(), color=BWD, stroke_width=6)
            anims.append(ShowPassingFlash(rev, time_width=0.6))
        self.play(*anims, run_time=0.9)

    def fill_grads(self, names):
        if not hasattr(self, "grad_cells"):
            self.grad_cells = {}
        cells = []
        for n in names:
            i = self.WEIGHTS.index(n)
            c = MathTex(f"{self.r['grads'][n]:.4f}", font_size=26, color=BWD
                        ).move_to([self.cols_x[2], self.row_y[i + 1], 0])
            self.grad_cells[n] = c
            cells.append(c)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in cells], lag_ratio=0.15))
