from manim import *

chem_tex_template = TexTemplate()
chem_tex_template.add_to_preamble(r"\usepackage{mhchem}")
chem_tex_template.add_to_preamble(
    r"\newcommand{\placeholder}[1]{\mathrel{\phantom{#1}}}"
)


def SharpArrow(start, end, **kwargs):
    return Arrow(
        start,
        end,
        buff=0.15,
        stroke_width=1.5,
        tip_length=0.15,
        **kwargs,
    )


__all__ = ["chem_tex_template", "SharpArrow"]
