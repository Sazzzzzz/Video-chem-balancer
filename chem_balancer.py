"""A parser for chemical formulas."""

import re
from typing import Any, Counter, NamedTuple, Optional
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

    def is_positive(self) -> bool:
        """Check if the token is TokenType.CHARGE and positive."""
        return self.type == TokenType.CHARGE and self.value == "+"

    def is_negative(self) -> bool:
        """Check if the token is TokenType.CHARGE and negative."""
        return self.type == TokenType.CHARGE and self.value == "-"

    def is_conjunction(self) -> bool:
        """Check if the token is `+` or `-`. Mainly for conjunction in chemical equations."""
        return self.is_positive() or self.is_negative()


class TokenType(Enum):
    ELEMENT = r"[a-zA-Z]+"
    # EQUALS must be processed before CHARGE
    EQUALS = r"==|=|->|⇌|⇄"
    # Charge must be processed before NUMBER
    # Then sorted by frequency of use
    CHARGE = r"\d*[+-]"
    NUMBER = r"\d+"
    DOT = r"·|\.|\*"
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
                # First split based on element regex
                # push every unknown part as new Element token
                pos = 0
                element_regex = r"|".join(ELEMENTS)
                for m in re.finditer(element_regex, value):
                    if m.start() != pos:
                        yield Token(TokenType.ELEMENT, value[pos : m.start()])
                        yield Token(TokenType.ELEMENT, m.group())
                        pos = m.end()
                if pos != len(value):
                    yield Token(TokenType.ELEMENT, value[pos:])
                continue
            case TokenType.WHITESPACE:
                continue
            case TokenType.MISMATCH:
                raise RuntimeError(f"{value!r} unexpected")

        yield Token(kind, value)


class FormulaParser:
    """A LL(2) parser for chemical formulas. Implements the following grammar:
    + `formula -> molecule (DOT [NUMBER] molecule)*`
    + `molecule -> term+ [CHARGE]`
    + `term -> element_unit | group_unit`
    + `element_unit -> ELEMENT [NUMBER]`
    + `group_unit -> LPAREN molecule RPAREN [NUMBER]`"""

    BRACKET_PAIRS = {
        TokenType.LPAREN: TokenType.RPAREN,
        TokenType.LBRACKET: TokenType.RBRACKET,
        TokenType.LBRACE: TokenType.RBRACE,
    }

    def __init__(
        self,
        tokens: list[Token],
        is_parsing_equation: bool = False,
    ):
        self.pos = 0
        self.tokens = tokens
        self.is_parsing_equation = is_parsing_equation
        self.current_token: Optional[Token] = self.tokens[0] if tokens else None

    def advance(self) -> None:
        """Moves to the next token."""
        self.pos += 1
        try:
            self.current_token = self.tokens[self.pos]
        except IndexError:
            self.current_token = None

    def peek(self, offset: int = 1) -> Token | None:
        """Peeks ahead in the token stream without advancing."""
        peek_position = self.pos + offset
        # Sadly Python doesn't have indexing with default value
        if peek_position < len(self.tokens):
            return self.tokens[peek_position]
        return None

    def parse_formula(self):
        """formula -> molecule (DOT molecule)*"""
        molecules = [{"molecule": self.parse_molecule(), "count": 1}]

        while (token := self.current_token) and token.type == TokenType.DOT:
            # Stop if we're parsing an equation and hit equation delimiters
            if self.is_parsing_equation and (
                token.type == TokenType.EQUALS or token.is_conjunction()
            ):
                break

            self.advance()  # Consume DOT
            count = 1
            if (token := self.current_token) and token.type == TokenType.NUMBER:
                count = int(token.value)
                self.advance()  # Consume NUMBER
            molecules.append(
                {
                    "molecule": self.parse_molecule(),
                    "count": count,
                }
            )

        return {
            "type": "formula",
            "molecules": molecules,
            "formula": "".join(token.value for token in self.tokens[: self.pos]),
        }

    def parse_molecule(self):
        """molecule -> term+ [CHARGE]"""
        terms = []
        while (
            token := self.current_token
        ) and token.type in FormulaParser.BRACKET_PAIRS.keys() | {TokenType.ELEMENT}:
            # Stop parsing if we encounter equation delimiters
            if self.is_parsing_equation and token.type in {TokenType.EQUALS}:
                break
            if self.is_parsing_equation and token.is_conjunction():
                break
            terms.append(self.parse_term())

        charge = 0
        if (token := self.current_token) and token.type == TokenType.CHARGE:
            # In equation parsing mode, check if '+' or '-' is a conjunction (separator)
            # rather than a charge by looking at what follows
            if self.is_parsing_equation and token.value in ("+", "-"):
                next_token = self.peek()
                # If followed by a number (stoichiometric coefficient), element, or bracket,
                # treat it as a conjunction, not a charge
                if next_token and next_token.type in {
                    TokenType.NUMBER,
                    TokenType.ELEMENT,
                    TokenType.LPAREN,
                    TokenType.LBRACKET,
                    TokenType.LBRACE,
                }:
                    return {"type": "molecule", "terms": terms, "charge": charge}

            match token.value:
                case "+":
                    charge = 1
                case "-":
                    charge = -1
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
            case bracket if bracket in FormulaParser.BRACKET_PAIRS:
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

        return {"symbol": element, "count": count}

    def parse_group_unit(self):
        """group_unit -> LPAREN formula RPAREN [NUMBER]"""
        left_bracket = self.current_token
        assert (
            left_bracket is not None
            and left_bracket.type in FormulaParser.BRACKET_PAIRS
        )
        right_bracket_type = FormulaParser.BRACKET_PAIRS[left_bracket.type]
        self.advance()  # Consume left bracket

        molecule = self.parse_molecule()
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
            "group": molecule,
            "brackets": (left_bracket.type, right_bracket.type),
            "count": count,
        }


def get_chemical_ast(formula: str) -> dict[str, Any]:
    tokens = tokenize(formula)
    parser = FormulaParser(list(tokens))
    return parser.parse_formula()


class ChemicalCounter:
    """Calculates the count of each element in a chemical formula AST."""

    def __init__(self, ast: dict[str, Any]) -> None:
        self.ast = ast

    def calculate(self) -> Counter[str]:
        return self._evaluate_formula(self.ast)

    def _evaluate_element(self, element_info: dict[str, Any]) -> Counter[str]:
        return Counter({element_info["symbol"]: element_info["count"]})

    def _evaluate_term(self, term_info: dict[str, Any]) -> Counter[str]:
        term_counter = Counter()
        match term_info:
            case {"symbol": _, "count": _}:
                term_counter += self._evaluate_element(term_info)
            case {"group": _, "count": _}:
                term_counter += self._evaluate_group(term_info)
        return term_counter

    def _evaluate_group(self, group_info: dict[str, Any]) -> Counter[str]:
        group_counter = self._evaluate_molecule(group_info["group"])
        multiplied_counter = Counter()
        for element, count in group_counter.items():
            multiplied_counter[element] = count * group_info["count"]
        return multiplied_counter

    def _evaluate_molecule(self, molecule_info: dict[str, Any]) -> Counter[str]:
        molecule_counter = sum(
            (self._evaluate_term(term) for term in molecule_info["terms"]), Counter()
        )
        molecule_counter["charge"] = molecule_info["charge"]

        return molecule_counter

    def _evaluate_formula(self, formula: dict[str, Any]) -> Counter[str]:
        total_counter = Counter()
        for molecule_info in formula["molecules"]:
            molecule = molecule_info["molecule"]
            count = molecule_info["count"]
            molecule_counter = self._evaluate_molecule(molecule)
            for element, qty in molecule_counter.items():
                total_counter[element] += qty * count
        return total_counter


def get_chemical_composition(formula: str) -> Counter[str]:
    ast = get_chemical_ast(formula)
    calculator = ChemicalCounter(ast)
    return calculator.calculate()


class EquationParser:
    """A LL(1) parser for chemical equations. Implements the following grammar:
    + equation -> compound_list EQUALS compound_list
    + compound_list -> stoichiometric_compound (PLUS stoichiometric_compound)*
    + stoichiometric_compound -> [NUMBER] formula"""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[self.position] if self.tokens else None

    def advance(self) -> None:
        """Moves to the next token."""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None

    def parse_equation(self) -> dict[str, Any]:
        """equation -> compound_list EQUALS compound_list"""
        left_compounds = self.parse_compound_list()

        if (token := self.current_token) is None or token.type != TokenType.EQUALS:
            raise RuntimeError(f"Expected EQUALS but found {token}")
        self.advance()  # Consume EQUALS

        right_compounds = self.parse_compound_list()

        return {
            "type": "equation",
            "reactants": left_compounds,
            "products": right_compounds,
        }

    def parse_compound_list(self) -> list[dict[str, Any]]:
        """compound_list -> stoichiometric_compound (PLUS stoichiometric_compound)*"""
        compounds = [self.parse_stoichiometric_compound()]

        # * The reason we use token.value == "+" instead of defining a PLUS token type is that
        # * Fe3+ and NH4+ are impossible to distinguish
        # * current solution is to force users to use space manually before charges
        # * to use atomic symbols as PLUS, MINUS etc., we need LL(2) parsing for chemical formulas
        # * Furthermore, we need to recognize whitespaces to identify compact charges like `3+`
        # * Although in theory by recognizing whitespaces and use methods like `next_non_whitespace_token`
        # * we can still refactor current code to a more rigorous one.

        while (token := self.current_token) and token.is_positive():
            self.advance()  # Consume PLUS
            compounds.append(self.parse_stoichiometric_compound())

        return compounds

    def parse_stoichiometric_compound(self) -> dict[str, Any]:
        """stoichiometric_compound -> [NUMBER] formula"""
        count = 1  # Default count
        if (token := self.current_token) and token.type == TokenType.NUMBER:
            count = int(token.value)
            self.advance()  # Consume NUMBER

        formula_parser = FormulaParser(
            self.tokens[self.position :], is_parsing_equation=True
        )
        formula = formula_parser.parse_formula()
        # Advance the main parser's position by the number of tokens consumed by the formula parser
        self.position += formula_parser.pos
        self.current_token = (
            self.tokens[self.position] if self.position < len(self.tokens) else None
        )

        return {"molecule": formula, "count": count}

    def parse(self) -> dict[str, Any]:
        return self.parse_equation()

def get_equation_ast(equation: str) -> dict[str, Any]:
    tokens = list(tokenize(equation))
    parser = EquationParser(tokens)
    return parser.parse()


if __name__ == "__main__":
    from pprint import pprint

    equation = "2Ag+ + CO3 2- == Ag2CO3"
    pprint(get_equation_ast(equation))