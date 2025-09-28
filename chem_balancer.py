"""A parser for chemical formulas."""

import re
from typing import Iterable, NamedTuple
from enum import Enum

# Two letter element symbols should appear before one letter symbols to avoid partial matches

ELEMENTS_STR = """He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr 
Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu 
Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa Np Pu Am Cm Bk Cf Es Fm Md No Lr 
Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og H B C N O F P S K V Y I W U"""
ELEMENTS = ELEMENTS_STR.split()


class Token(NamedTuple):
    type: "TokenType"
    # tokenizers should only carry str for values, leaving parsing for convertion if needed
    value: str


class TokenType(Enum):
    ELEMENT = r"[a-zA-Z]+"
    # Charge must be processed before NUMBER
    CHARGE = r"\d*[+-]"
    NUMBER = r"\d+"
    DOT = r"·|\.|\*|·"
    LPAREN = r"\("
    RPAREN = r"\)"
    LBRACKET = r"\["
    RBRACKET = r"\]"
    LBRACE = r"\{"
    RBRACE = r"\}"
    WHITESPACE = r"[ \t]+"
    # MISMATCH should always be last; it matches any character and will mask valid tokens if placed earlier.
    MISMATCH = r"."


def tokenize(formula: str):
    token_regex = r"|".join(
        rf"(?P<{token_type.name}>{token_type.value})" for token_type in TokenType
    )

    for mo in re.finditer(token_regex, formula):
        name, value = mo.lastgroup, mo.group()
        assert name is not None
        kind = TokenType[name]

        match kind:
            case TokenType.ELEMENT:
                # Second match to separate elements and self-defined groups
                element_regex = r"|".join(ELEMENTS) + r"|[a-zA-Z]+"
                yield from (
                    Token(TokenType.ELEMENT, x.group())
                    for x in re.finditer(element_regex, value)
                )
                continue
            case TokenType.WHITESPACE:
                continue
            case TokenType.MISMATCH:
                raise RuntimeError(f"{value!r} unexpected")

        yield Token(kind, value)


# 1. `formula -> molecule (DOT [NUMBER] molecule)*`
# 2. `molecule -> term+ [CHARGE]`
# 3. `term -> element_unit | group_unit`
# 4. `element_unit -> ELEMENT [NUMBER]`
# 5. `group_unit -> LPAREN formula RPAREN [NUMBER]`
class Parser:
    BRACKET_PAIRS = {
        TokenType.LPAREN: TokenType.RPAREN,
        TokenType.LBRACKET: TokenType.RBRACKET,
        TokenType.LBRACE: TokenType.RBRACE,
    }

    def __init__(self, tokens: Iterable[Token]):
        self.tokens = iter(tokens)
        self.advance()

    def advance(self) -> None:
        """Moves to the next token."""
        try:
            self.current_token = next(self.tokens)
        except StopIteration:
            self.current_token = None

    def parse_formula(self):
        """formula -> molecule (DOT molecule)*"""
        molecules = [{"molecule": self.parse_molecule(), "count": 1}]

        while (token := self.current_token) and token.type == TokenType.DOT:
            self.advance()  # Consume DOT
            count = 1
            if (token := self.current_token) and token.type == TokenType.NUMBER:
                count = int(token.value)
                self.advance()  # Consume NUMBER
            molecules.append({"molecule": self.parse_molecule(), "count": count})

        return {"type": "formula", "molecules": molecules}

    def parse_molecule(self):
        """molecule -> term+ [CHARGE]"""
        terms = []
        while (
            token := self.current_token
        ) and token.type in Parser.BRACKET_PAIRS.keys() | {TokenType.ELEMENT}:
            terms.append(self.parse_term())

        charge = 0
        if (token := self.current_token) and token.type == TokenType.CHARGE:
            match token.value:
                case "+" | "-":
                    charge = 1 if token.value == "+" else -1
                case _ if token.value.endswith("+"):
                    charge = int(token.value[:-1]) if token.value[:-1] else 1
                case _ if token.value.endswith("-"):
                    charge = -int(token.value[:-1]) if token.value[:-1] else -1
                case _:
                    raise RuntimeError(f"Invalid charge format: {token.value}")
            self.advance()  # Consume CHARGE

        return {"type": "molecule", "terms": terms, "charge": charge}

    def parse_term(self):
        """term -> element_unit | group_unit"""
        token = self.current_token
        if token is None:
            raise RuntimeError("Unexpected end of input while parsing term")

        match token.type:
            case TokenType.ELEMENT:
                return self.parse_element_unit()
            case bracket if bracket in Parser.BRACKET_PAIRS:
                return self.parse_group_unit()
            case _:
                raise RuntimeError(f"Unexpected token {token} while parsing term")

    def parse_element_unit(self):
        """element_unit -> ELEMENT [NUMBER]"""
        element_token = self.current_token
        assert element_token is not None and element_token.type == TokenType.ELEMENT
        element = element_token.value
        self.advance()  # Consume ELEMENT

        count = 1  # Default count
        if (token := self.current_token) and token.type == TokenType.NUMBER:
            count = int(token.value)
            self.advance()  # Consume NUMBER

        return {"element": element, "count": count}

    def parse_group_unit(self):
        """group_unit -> LPAREN formula RPAREN [NUMBER]"""
        left_bracket = self.current_token
        assert left_bracket is not None and left_bracket.type in Parser.BRACKET_PAIRS
        right_bracket_type = Parser.BRACKET_PAIRS[left_bracket.type]
        self.advance()  # Consume left bracket

        formula = self.parse_formula()
        right_bracket = self.current_token
        if right_bracket is None or right_bracket.type != right_bracket_type:
            raise RuntimeError(
                f"Expected closing bracket {right_bracket_type} but found {right_bracket}"
            )

        self.advance()  # Consume right bracket

        count = 1  # Default group count
        if (token := self.current_token) and token.type == TokenType.NUMBER:
            count = int(token.value)
            self.advance()  # Consume NUMBER

        return {
            "group": formula,
            "brackets": (left_bracket.type, right_bracket.type),
            "count": count,
        }


if __name__ == "__main__":
    from pprint import pprint

    formula = "Cu(en)2 2+ . 5H2O"
    tokens = list(tokenize(formula))
    pprint(tokens)
    parser = Parser(tokens)
    ast = parser.parse_formula()
    pprint(ast)
