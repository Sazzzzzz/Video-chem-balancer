from manim import *
from utils import *

config.background_color = LOGO_WHITE

Mobject.set_default(color=BLACK)
Dot.set_default(color=BLACK)


class DecreasingCoefficients(Scene):
    """A Scene that shows the coefficients of a chemical equation decreasing to zero like a roller counter."""

    def construct(self):

        reaction = MathTex(
            "3",
            r"\ce{Pb^2+} +",
            "6",
            r"\ce{I^-} \ce{->} ",
            "3",
            r"\ce{PbI2}",
            tex_template=chem_tex_template,
        )
        rec = SurroundingRectangle(
            reaction, fill_color=WHITE, fill_opacity=1, stroke_width=0
        )
        screen = Rectangle(width=config.frame_width, height=config.frame_height)
        mask = Difference(screen, rec, color=config.background_color, fill_opacity=1)
        # Pb
        # get the center of the first coefficient (3)
        Pb_coeff = reaction[0]  # Access the '3' coefficient
        Pb_coeff.set_opacity(0)  # Hide the original coefficient
        Pb_coeff_center = Pb_coeff.get_center()

        Pb_digits = ["3", "2", "1", "0"]
        Pb_group = VGroup(*[MathTex(digit) for digit in Pb_digits])
        Pb_group.arrange(DOWN, buff=0.2)
        Pb_offset = Pb_coeff_center - Pb_group[0].get_center()
        Pb_group.shift(Pb_offset)

        # I-
        I_coeff = reaction[2]
        I_coeff.set_opacity(0)
        I_coeff_center = I_coeff.get_center()

        I_digits = ["6", "4", "2", "0"]
        I_group = VGroup(*[MathTex(digit) for digit in I_digits])
        I_group.arrange(DOWN, buff=0.2)
        Pb_offset = I_coeff_center - I_group[0].get_center()
        I_group.shift(Pb_offset)

        # PbI2
        PbI2_coeff = reaction[4]
        PbI2_coeff.set_opacity(0)
        PbI2_coeff_center = PbI2_coeff.get_center()

        PbI2_digits = ["3", "2", "1", "0"]
        PbI2_group = VGroup(*[MathTex(digit) for digit in PbI2_digits])
        PbI2_group.arrange(DOWN, buff=0.2)
        Pb_offset = PbI2_coeff_center - PbI2_group[0].get_center()
        PbI2_group.shift(Pb_offset)

        # set glow
        Pb_group[-1].set_color(ORANGE).set_stroke(color=ORANGE, width=2).set_glow(0.5)
        I_group[-1].set_color(ORANGE).set_stroke(color=ORANGE, width=2).set_glow(0.5)
        PbI2_group[-1].set_color(ORANGE).set_stroke(color=ORANGE, width=2).set_glow(0.5)

        self.add(Pb_group, I_group, PbI2_group, reaction, mask)

        self.wait(1)
        for i in range(1, 4):
            Pb_offset = Pb_coeff_center - Pb_group[i].get_center()
            I_offset = I_coeff_center - I_group[i].get_center()
            PbI2_offset = PbI2_coeff_center - PbI2_group[i].get_center()
            self.play(
                Pb_group.animate.shift(Pb_offset),
                I_group.animate.shift(I_offset),
                PbI2_group.animate.shift(PbI2_offset),
                # rate_func=toggle,
            )
            self.wait(0.1)

        self.wait(0.8)
