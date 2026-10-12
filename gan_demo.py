"""
================================================================================
gan_demo.py · GAN 모드 붕괴 시각화 — 특집호 Demo 10
================================================================================

구성
  gan_core.py  : NumPy로 기본 GAN · PacGAN 을 실제 학습 + 3중 검증 (기울기 · 지표 · 재현성)
  gan_demo.py  : (이 파일) 검증을 통과한 학습 스냅샷을 그대로 그림 — 손으로 그린 점은 하나도 없음

장면 흐름 (약 70초)
  ① 진짜 데이터        : 원 위의 8개 모드
  ② GAN 게임           : 생성자 vs 판별자
  ③ PacGAN            : 판별자가 m개씩 묶어서 판정
  ④ 실제 학습          : 같은 시드 · 같은 조건, 스냅샷 13장 + 덮은 모드 수 곡선
  ⑤ 측정 기반 판정     : '붕괴' 여부를 지표로만 말한다
  ⑥ 검증              : 통과하지 못하면 렌더링 자체를 중단

실행
  python gan_core.py --inject --seeds 5
  manim -pql gan_demo.py GANModeCollapse
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manim import *
from gan_core import run

KO_FONT = os.environ.get("KO_FONT", "Malgun Gothic")
BASE_C, PAC_C = BLUE_C, GREEN_C
OK_C = GREEN
N_SHOW, N_REAL = 300, 200


def sci(v):
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


class GANModeCollapse(Scene):

    PANEL = 3.2
    DOMAIN = 2.6
    LC = np.array([-5.0, 0.85, 0.0])
    RC = np.array([5.0, 0.85, 0.0])

    def to_screen(self, pts, center):
        s = self.PANEL / (2 * self.DOMAIN)
        p = np.clip(np.asarray(pts), -self.DOMAIN + 0.05, self.DOMAIN - 0.05)   # 그리기용 자르기 (지표와 무관)
        return [center + np.array([x * s, y * s, 0.0]) for x, y in p]

    def dots(self, pts, center, color, radius=0.028, opacity=0.9):
        return VGroup(*[Dot(q, radius=radius, color=color, fill_opacity=opacity)
                        for q in self.to_screen(pts, center)])

    def construct(self):
        # ============ 0. 학습 + 검증 ============
        r = run()
        if not r["passed"]:
            raise RuntimeError(f"GAN 검증 실패 ({r['verify']['stopped_at']}) — 렌더링하지 않습니다.")
        c = r["config"]
        p = c["pack"]
        K, s = c["K"], self.PANEL / (2 * self.DOMAIN)
        snaps = {1: r["runs"][1]["snaps"], p: r["runs"][p]["snaps"]}
        steps = r["runs"][1]["steps"]
        self.panel = None

        title = Text("GAN 모드 붕괴: 생성자가 몇 가지 답만 반복할 때",
                     font=KO_FONT, font_size=30).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ======================= ① 진짜 데이터 =======================
        frames = VGroup(*[Square(self.PANEL, color=GREY_D, stroke_width=1.5).move_to(cc)
                          for cc in (self.LC, self.RC)])
        rings = {}
        for key, cc in ((1, self.LC), (p, self.RC)):
            rings[key] = VGroup(*[Circle(radius=c["radius"] * s, color=GREY_C, stroke_width=2)
                                  .move_to(q) for q in self.to_screen(r["centers"], cc)])
        real = r["real"][:N_REAL]
        real_l = self.dots(real, self.LC, GREY_B, 0.022, 0.8)
        real_r = self.dots(real, self.RC, GREY_B, 0.022, 0.8)
        self.play(Create(frames), FadeIn(rings[1]), FadeIn(rings[p]))
        self.play(FadeIn(real_l, lag_ratio=0.01), FadeIn(real_r, lag_ratio=0.01), run_time=1.2)
        self.show_panel(
            f"① 진짜 데이터: 원 위에 놓인 {K}개의 봉우리(모드)",
            [rf"p_{{\mathrm{{data}}}}=\frac{{1}}{{{K}}}\sum_{{k=1}}^{{{K}}}\mathcal{{N}}(\mu_k,\ \sigma^2 I),"
             rf"\quad \sigma={c['std']:g}"],
            note=f"목표: 생성자가 {K}개 모드를 모두 골고루 만들어 내는 것")

        # ======================= ② GAN 게임 =======================
        lt = Text("기본 GAN", font=KO_FONT, font_size=22, color=BASE_C).next_to(frames[0], UP, buff=0.12)
        rt = Text(f"PacGAN ({p}개씩 묶어 판정)", font=KO_FONT, font_size=22, color=PAC_C
                  ).next_to(frames[1], UP, buff=0.12)
        self.play(FadeIn(lt), real_l.animate.set_opacity(0.18), real_r.animate.set_opacity(0.18))
        self.show_panel(
            "② GAN: 생성자 G와 판별자 D의 게임",
            [r"\min_G\max_D\ \mathbb{E}_{x}\big[\log D(x)\big]+\mathbb{E}_{z}\big[\log(1-D(G(z)))\big]"],
            note="생성자는 판별자를 가장 쉽게 속이는 몇 개의 답으로 몰리기 쉽다 → 모드 붕괴")

        # ======================= ③ PacGAN =======================
        args = r"x_1,\ x_2" if p == 2 else r"x_1,\dots,x_{" + str(p) + "}"
        self.play(FadeIn(rt))
        self.show_panel(
            f"③ 완화 기법 PacGAN: 판별자가 샘플을 {p}개씩 묶어서 판정",
            [rf"D({args})\ :\ \mathbb{{R}}^{{{2 * p}}}\rightarrow[0,1]"],
            note="묶음 안의 다양성이 부족하면 가짜로 들키기 쉽다 (Lin et al., NeurIPS 2018)")

        # ======================= ④ 실제 학습 =======================
        ax = Axes(x_range=[0, c["steps"], c["steps"] / 2], y_range=[0, K, K / 2],
                  x_length=4.4, y_length=2.4, tips=False,
                  axis_config={"color": GREY_B, "font_size": 20,
                               "decimal_number_config": {"num_decimal_places": 0,
                                                         "group_with_commas": False}},
                  x_axis_config={"include_numbers": True,
                                 "numbers_to_include": [c["steps"] // 2, c["steps"]]},
                  y_axis_config={"include_numbers": True, "numbers_to_include": [K // 2, K]},
                  ).move_to([0.15, 0.75, 0])
        ax_y = Text("덮은 모드 수", font=KO_FONT, font_size=16, color=GREY_A
                    ).next_to(ax.y_axis.get_end(), UP, buff=0.12)
        ax_x = Text("학습 단계", font=KO_FONT, font_size=16, color=GREY_A
                    ).next_to(ax.x_axis, DOWN, buff=0.1)
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.35, color=BASE_C, stroke_width=4),
                   Text("기본 GAN", font=KO_FONT, font_size=16, color=BASE_C)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.35, color=PAC_C, stroke_width=4),
                   Text("PacGAN", font=KO_FONT, font_size=16, color=PAC_C)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.4).next_to(ax_x, DOWN, buff=0.14)
        tlabel = MathTex(f"t = {steps[0]}", font_size=30).move_to([0, 2.55, 0])

        def counter(m, color, frame):
            g = VGroup(Text("덮은 모드", font=KO_FONT, font_size=20, color=color),
                       MathTex(f"{m}/{K}", font_size=32, color=color)).arrange(RIGHT, buff=0.15)
            return g.next_to(frame, DOWN, buff=0.18)

        cnt_l = counter(r["hist"][1][0], BASE_C, frames[0])
        cnt_r = counter(r["hist"][p][0], PAC_C, frames[1])
        gen_l = self.dots(snaps[1][0][:N_SHOW], self.LC, BASE_C)
        gen_r = self.dots(snaps[p][0][:N_SHOW], self.RC, PAC_C)
        self.show_panel(
            f"④ 같은 시드({c['seed']})·같은 조건으로 실제 학습: 0 → {c['steps']} 단계",
            [r"\text{coverage}=\#\{k:\ \hat p_k\ge " + f"{c['min_frac']:g}" + r"\}"],
            note=f"덮은 모드 = 반경 {c['radius']:g} 안에 샘플이 {c['min_frac']:.0%} 이상 모인 모드   ·   "
                 f"점 = 고정 잡음 z로 뽑은 생성 샘플 {N_SHOW}개")
        self.play(Create(ax), FadeIn(ax_y), FadeIn(ax_x), FadeIn(legend), FadeIn(tlabel),
                  FadeIn(gen_l), FadeIn(gen_r), FadeIn(cnt_l), FadeIn(cnt_r))

        def ring_anims(key, k):
            share = self.share(snaps[key][k], r["centers"], c)
            col = BASE_C if key == 1 else PAC_C
            return [rg.animate.set_stroke(col if share[j] >= c["min_frac"] else GREY_C,
                                          width=4 if share[j] >= c["min_frac"] else 2)
                    for j, rg in enumerate(rings[key])]

        for k in range(1, len(steps)):
            seg_l = Line(ax.c2p(steps[k - 1], r["hist"][1][k - 1]), ax.c2p(steps[k], r["hist"][1][k]),
                         color=BASE_C, stroke_width=4)
            seg_r = Line(ax.c2p(steps[k - 1], r["hist"][p][k - 1]), ax.c2p(steps[k], r["hist"][p][k]),
                         color=PAC_C, stroke_width=4)
            self.play(Transform(gen_l, self.dots(snaps[1][k][:N_SHOW], self.LC, BASE_C)),
                      Transform(gen_r, self.dots(snaps[p][k][:N_SHOW], self.RC, PAC_C)),
                      Create(seg_l), Create(seg_r),
                      Transform(tlabel, MathTex(f"t = {steps[k]}", font_size=30).move_to(tlabel)),
                      Transform(cnt_l, counter(r["hist"][1][k], BASE_C, frames[0])),
                      Transform(cnt_r, counter(r["hist"][p][k], PAC_C, frames[1])),
                      *ring_anims(1, k), *ring_anims(p, k),
                      run_time=0.9)
        self.wait(0.8)

        # ======================= ⑤ 측정 기반 판정 =======================
        f1, fp = r["final"][1], r["final"][p]
        if r["collapsed"][1] and not r["collapsed"][p]:
            verdict = f"→ 기본 GAN에서 모드 붕괴, PacGAN에서 완화 (붕괴 기준: {K // 2}개 이하)"
        elif r["collapsed"][1] and r["collapsed"][p]:
            verdict = f"→ 두 모델 모두 붕괴 (붕괴 기준: {K // 2}개 이하)"
        elif not r["collapsed"][1]:
            verdict = "→ 이 설정에서는 기본 GAN도 뚜렷한 붕괴가 없다"
        else:
            verdict = "→ 이 설정에서는 PacGAN이 오히려 더 적은 모드를 덮었다"
        self.show_panel(
            "⑤ 결과: '붕괴'는 그림이 아니라 측정값으로 판정",
            [rf"\max_k \hat p_k={f1['share'].max():.2f}\ (\text{{base}}),\qquad"
             rf"{fp['share'].max():.2f}\ (\text{{PacGAN}})"],
            note=f"덮은 모드: 기본 GAN {f1['covered']}/{K}, PacGAN {fp['covered']}/{K}   {verdict}")
        if r["collapsed"][1]:
            self.play(Indicate(cnt_l, color=YELLOW), Indicate(gen_l, color=YELLOW, scale_factor=1.05))
        self.wait(1.5)

        # ======================= ⑥ 검증 =======================
        v, m = r["verify"], r["verify"]["metric"]
        self.show_panel(
            "⑥ 검증: 기울기 · 지표 · 재현성 3중 관문",
            [r"\max\frac{|\nabla_{\mathrm{analytic}}-\nabla_{\mathrm{numeric}}|}"
             r"{\max(1,\ |\nabla_{\mathrm{numeric}}|)}=" + sci(v["grad_err"]),
             rf"\text{{real data}}:\ {m['real_cov']}/{K},\qquad"
             r"\max|\text{snapshot}-\text{retrain}|=" + sci(v["repro"])],
            note="진짜 데이터 보정 통과 · 지표 두 구현 일치 · 처음부터 다시 학습해도 비트 단위로 같은 결과",
            gap=0.72)
        badge = VGroup(Text("검증 통과", font=KO_FONT, font_size=20, color=OK_C, weight=BOLD),
                       Text("· 영상의 모든 점 = 실제 학습 결과", font=KO_FONT, font_size=15,
                            color=GREY_B)).arrange(RIGHT, buff=0.12)
        box = SurroundingRectangle(badge, color=OK_C, buff=0.1, corner_radius=0.1)
        VGroup(badge, box).move_to([0, -1.35, 0])
        self.play(FadeOut(legend), FadeIn(badge, scale=0.9), Create(box))
        self.wait(2.5)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def share(x, centers, c):
        d = np.linalg.norm(x[:, None, :] - centers[None], axis=2)
        j, near = d.argmin(1), d.min(1) < c["radius"]
        return np.bincount(j[near], minlength=c["K"]) / len(x)

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
