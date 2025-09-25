"""A parser for chemical formulas."""

import re
from typing import NamedTuple


ELEMENTS_STR = """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
Rf Db Sg Bh Hs Mt Ds Rg Cn Fl Lv Ts Og
"""

ELEMENTS: set[str] = set(ELEMENTS_STR.split())


# class TokenKind(Enum):
#     ELEMENT = "ELEMENT"
#     NUMBER = "NUMBER"
#     LPAREN = "LPAREN"
#     RPAREN = "RPAREN"


class Token(NamedTuple):
    kind: str
    value: str | int


def tokenizer(code):
    """A simple tokenizer based on regular expressions."""
    token_specification = [
        ("ELEMENT", r"[A-Z][a-z]?"),
        ("NUMBER", r"\d+"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("NEGATIVE", r"-"),
        ("POSITIVE", r"\+"),
        ("SKIP", r"[ \t]+"),
        ("MISMATCH", r"."),
    ]

    tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)

    for mo in re.finditer(tok_regex, code):
        kind, value = mo.lastgroup, mo.group()
        match kind:
            case "NUMBER":
                value = int(value)
            case "SKIP":
                continue
            case "MISMATCH":
                raise RuntimeError(f"{value!r} unexpected")
            case None:
                raise RuntimeError("Unnamed group matched")

        yield Token(kind, value)


# --- Example Usage ---
if __name__ == "__main__":
    formula = "{[Co(NH3)4(OH)2]3Co}(SO4)3"
    for token in tokenizer(formula):
        print(token)
