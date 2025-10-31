"""A parser for chemical formulas."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, NamedTuple, Optional, TypeAlias

from utils import Counter, scale

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def is_conjunction(self) -> bool:
        """Check if the token is `+` or `-`. Mainly for conjunction in chemical equations."""
        return self.type in {TokenType.PLUS, TokenType.MINUS}


class TokenType(Enum):
    _PREPROCESSING = r"\(aq\)|\(s\)|\(l\)|\(g\)"
    # ELECTRON is the first token type
    ELECTRON = r"e-"
    ELEMENT = r"[a-zA-Z]+"
    # EQUALS must be processed before CHARGE
    # Traditional '=' would be recognized as double bond and ignored
    EQUALS = r"==|->|⇌|⇄"
    MINUS = r"-"
    PLUS = r"\+"
    # Charge must be processed before NUMBER
    # Then sorted by frequency of use
    NUMBER = r"\d+"
    DOT = r"·|\.|\*"
    LPAREN = r"\("
    RPAREN = r"\)"
    LBRACKET = r"\["
    RBRACKET = r"\]"
    LBRACE = r"\{"
    RBRACE = r"\}"
    WHITESPACE = r"[ \t]+"
    # MISMATCH should always be last; it matches any character
    MISMATCH = r"."


BRACKET_PAIRS = {
    TokenType.LPAREN: TokenType.RPAREN,
    TokenType.LBRACKET: TokenType.RBRACKET,
    TokenType.LBRACE: TokenType.RBRACE,
}


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
            case TokenType.MISMATCH | TokenType._PREPROCESSING:
                logger.warning(f"Unexpected character {value!r} in formula, skipping.")
                continue

        yield Token(kind, value)


class FormulaParser:
    """A LL(2) parser for chemical formulas. Implements the following grammar:
    + `formula -> charged_molecule (DOT [NUMBER] charged_molecule)*`
    + `charged_molecule = molecule [charge]`
    + `molecule -> term+`
    + `charge -> [NUMBER] (PLUS | MINUS)`
    + `term -> element_unit | group_unit`
    + `element_unit -> ELEMENT [NUMBER]`
    + `group_unit -> LPAREN molecule RPAREN [NUMBER]`"""

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
        """Moves to the next token. Ignoring all whitespaces"""
        self.pos += 1
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type != TokenType.WHITESPACE:
                self.current_token = token
                return
            self.pos += 1
        self.current_token = None

    def peek(self, offset: int = 1) -> Token | None:
        """Peeks ahead in the token stream without advancing. Does NOT ignore whitespaces."""
        peek_position = self.pos + offset
        # Sadly Python doesn't have indexing with default value
        if peek_position < len(self.tokens):
            return self.tokens[peek_position]
        return None

    def peek_non_whitespace(self, offset: int = 1) -> Token | None:
        """Peeks ahead in the token stream without advancing. Ignores whitespaces."""
        peek_position = self.pos + offset
        while peek_position < len(self.tokens):
            token = self.tokens[peek_position]
            if token.type != TokenType.WHITESPACE:
                return token
            peek_position += 1
        return None

    def parse_formula(self):
        """formula -> molecule (DOT molecule)*

        This is the entry point for parsing a chemical formula."""
        molecules = [{"molecule": self.parse_charged_molecule(), "count": 1}]

        while (token := self.current_token) and token.type == TokenType.DOT:
            self.advance()  # Consume DOT
            count = 1
            if (token := self.current_token) and token.type == TokenType.NUMBER:
                count = int(token.value)
                self.advance()  # Consume NUMBER
            molecules.append(
                {
                    "molecule": self.parse_charged_molecule(),
                    "count": count,
                }
            )

        return {
            "type": "formula",
            "molecules": molecules,
            "formula": "".join(token.value for token in self.tokens[: self.pos]),
        }

    # To resolve the optional NUMBER token appearing before charges and after term and group
    # We use the following strategy:
    # When parsing `group_unit` or `element_unit` and encountering a number, look at the next character (with whitespace).
    # If it is tightly before PLUS or MINUS, stop parsing and return to the caller parse_molecule. In this case charge processing
    # function should be called immediately otherwise it is considered not legitimate grammar.
    # Otherwise, consider it as a multiple of `group_unit` or `element_unit`, consume the token.

    # In the charge processing function, if the first token is PLUS/MINUS and in chemical equation mode,
    # look at the next non-empty character. If it is not PLUS/MINUS, consider this PLUS/MINUS as a connector and stop parsing.
    # Otherwise, consider it as a single charge. If the first token is a number, parse it directly.
    def parse_charged_molecule(self) -> dict[str, Any]:
        """charged_molecule -> ELECTRON | (molecule [charge])"""
        if (token := self.current_token) and token.type == TokenType.ELECTRON:
            self.advance()  # Consume ELECTRON
            return {"type": "molecule", "terms": [], "charge": -1}

        molecule = self.parse_molecule()
        charge = 0
        if (token := self.current_token) and (
            # Optional NUMBER before charge
            token.type == TokenType.NUMBER or token.is_conjunction()
        ):
            charge = self.parse_charge()

        return {"type": "molecule", "terms": molecule, "charge": charge}

    def parse_molecule(self) -> list[dict[str, Any]]:
        """molecule -> term+ [charge]"""
        terms: list[dict[str, Any]] = []
        while (token := self.current_token) and token.type in BRACKET_PAIRS.keys() | {
            TokenType.ELEMENT
        }:
            # Stop parsing if we encounter equation delimiters
            if self.is_parsing_equation and token.type in {TokenType.EQUALS}:
                break
            if self.is_parsing_equation and token.is_conjunction():
                break
            terms.append(self.parse_term())

        return terms

    def parse_charge(self) -> int:
        """charge -> [NUMBER] (PLUS | MINUS)"""
        charge = 1
        if (token := self.current_token) and token.type == TokenType.NUMBER:
            charge = int(token.value)
            self.advance()  # Consume NUMBER

        if (token := self.current_token) and token.is_conjunction():
            if (
                self.is_parsing_equation
                and (next_token := self.peek_non_whitespace())
                and not next_token.is_conjunction()
                and next_token.type != TokenType.EQUALS
            ):
                return 0
            sign = 1 if token.type == TokenType.PLUS else -1
            charge *= sign
            self.advance()  # Consume PLUS or MINUS
        else:
            raise RuntimeError(f"Expected PLUS or MINUS but found {token}")
        return charge

    def parse_term(self):
        """term -> element_unit | group_unit"""
        token = self.current_token
        if token is None:
            raise RuntimeError("Unexpected end of input while parsing term")

        match token.type:
            case TokenType.ELEMENT:
                return self.parse_element_unit()
            case bracket if bracket in BRACKET_PAIRS:
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
            if (t := self.peek()) and t.is_conjunction():
                return {"symbol": element, "count": count}
            count = int(token.value)
            self.advance()  # Consume NUMBER

        return {"symbol": element, "count": count}

    def parse_group_unit(self):
        """group_unit -> LPAREN molecule RPAREN [NUMBER]"""
        left_bracket = self.current_token
        assert left_bracket is not None and left_bracket.type in BRACKET_PAIRS
        right_bracket_type = BRACKET_PAIRS[left_bracket.type]
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
            if (t := self.peek()) and t.is_conjunction():
                return {
                    "group": molecule,
                    "brackets": (left_bracket.type, right_bracket.type),
                    "count": count,
                }
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


Element: TypeAlias = str
"""For clarity, an Element is represented as a string symbol.
The substitution from `str` to `Element` happens in building procedure"""


class FormulaBuilder:
    """Calculates the count of each element in a chemical formula AST."""

    def __init__(self, ast: dict[str, Any]) -> None:
        self.ast = ast

    def calculate(self) -> Counter[Element]:
        return self._evaluate_formula(self.ast)

    def _evaluate_element(self, element_info: dict[str, Any]) -> Counter[Element]:
        return Counter({element_info["symbol"]: element_info["count"]})

    def _evaluate_term(self, term_info: dict[str, Any]) -> Counter[Element]:
        term_counter = Counter()
        match term_info:
            case {"symbol": _, "count": _}:
                term_counter += self._evaluate_element(term_info)
            case {"group": _, "count": _, "brackets": _}:
                term_counter += self._evaluate_group(term_info)
        return term_counter

    def _evaluate_group(self, group_info: dict[str, Any]) -> Counter[Element]:
        group_counter = sum(
            (self._evaluate_term(term) for term in group_info["group"]), Counter()
        )
        return scale(group_counter, group_info["count"])

    def _evaluate_molecule(self, molecule_info: dict[str, Any]) -> Counter[Element]:
        molecule_counter = sum(
            (self._evaluate_term(term) for term in molecule_info["terms"]),
            Counter(),
        )
        if charge := molecule_info.get("charge", 0):
            molecule_counter["charge"] = charge

        return molecule_counter

    def _evaluate_formula(self, formula: dict[str, Any]) -> Counter[Element]:
        total_counter: Counter[Element] = Counter()
        for molecule_info in formula["molecules"]:
            molecule = molecule_info["molecule"]
            count = molecule_info["count"]
            molecule_counter = self._evaluate_molecule(molecule)
            scaled_counter = scale(molecule_counter, count)
            total_counter.update(scaled_counter)
        return total_counter


def _get_chemical_composition_from_ast(ast: dict[str, Any]) -> Counter[Element]:
    calculator = FormulaBuilder(ast)
    return calculator.calculate()


def get_chemical_composition(formula: str) -> Counter[Element]:
    ast = get_chemical_ast(formula)
    return _get_chemical_composition_from_ast(ast)


class EquationParser:
    """A LL(1) parser for chemical equations. Implements the following grammar:
    + equation -> compound_list EQUALS compound_list
    + compound_list -> stoichiometric_compound (PLUS stoichiometric_compound)*
    + stoichiometric_compound -> [NUMBER] formula"""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else None

    def advance(self) -> None:
        """Moves to the next token. Ignoring all whitespaces"""
        self.pos += 1
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type != TokenType.WHITESPACE:
                self.current_token = token
                return
            self.pos += 1
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

        while (token := self.current_token) and token.is_conjunction():
            is_neg_coeff: bool = True if token.type != TokenType.PLUS else False

            self.advance()  # Consume PLUS
            compounds.append(self.parse_stoichiometric_compound(is_neg_coeff))

        return compounds

    def parse_stoichiometric_compound(
        self, invert_coeff: bool = False
    ) -> dict[str, Any]:
        """stoichiometric_compound -> [NUMBER] formula"""
        count = 1  # Default count
        if (token := self.current_token) and token.type == TokenType.NUMBER:
            count = int(token.value)
            self.advance()  # Consume NUMBER

        formula_parser = FormulaParser(
            self.tokens[self.pos :], is_parsing_equation=True
        )
        formula = formula_parser.parse_formula()
        # Advance the main parser's position by the number of tokens consumed by the formula parser
        self.pos += formula_parser.pos
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else None
        )
        count = -count if invert_coeff else count
        return {"molecule": formula, "count": count}

    def parse(self) -> dict[str, Any]:
        return self.parse_equation()


def get_equation_ast(equation: str) -> dict[str, Any]:
    tokens = list(tokenize(equation))
    parser = EquationParser(tokens)
    return parser.parse()


# I won't bother using pydantic here ...
@dataclass(frozen=True)
class Formula:
    """A chemical formula with its element composition.

    The formula is immutable once created (frozen dataclass).
    The composition `Counter` is encapsulated to prevent modification.
    """

    formula: str
    _composition: Counter[Element]

    def __post_init__(self) -> None:
        if not isinstance(self._composition, Counter):
            object.__setattr__(self, "_composition", Counter(self._composition))

    @property
    def composition(self) -> MappingProxyType[Element, int]:
        return MappingProxyType(self._composition)

    def __hash__(self) -> int:
        return hash((self.formula, frozenset(self._composition.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Formula):
            return NotImplemented
        return self.formula == other.formula and self._composition == other._composition

    def __lt__(self, other: "Formula") -> bool:
        return self.formula < other.formula


@dataclass
class BaseEquation:
    """A chemical equation with reactants and products."""

    reactants: Counter[Formula]
    products: Counter[Formula]

    def __str__(self) -> str:
        is_single_reactant, is_single_product = (
            len(self.reactants) == 1,
            len(self.products) == 1,
        )
        reactant_strs = [
            f"{c if c != 1 else ''}{f.formula}"
            for f, c in self.reactants.items()
            if (c != 0 or is_single_reactant)
        ]
        product_strs = [
            f"{c if c != 1 else ''}{f.formula}"
            for f, c in self.products.items()
            if (c != 0 or is_single_product)
        ]
        s = " + ".join(reactant_strs) + " == " + " + ".join(product_strs)
        return s.replace(" + -", " - ")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BaseEquation):
            return NotImplemented
        return self.reactants == value.reactants and self.products == value.products


class EquationBuilder:
    """Calculates the element counts for each side of a chemical equation based on its AST.
    Returns a Equation dataclass instance."""

    def __init__(self, ast: dict[str, Any]):
        self.ast = ast

    def build(self) -> BaseEquation:
        reactants = self._build_side(self.ast["reactants"])
        products = self._build_side(self.ast["products"])
        return BaseEquation(reactants=reactants, products=products)

    def _build_side(self, side: list[dict[str, Any]]) -> Counter[Formula]:
        counted_formulas: Counter[Formula] = Counter()
        for compound_info in side:
            count = compound_info["count"]

            formula_ast = compound_info["molecule"]
            formula_str = formula_ast["formula"].strip()
            if not formula_str:
                raise RuntimeError("Empty formula in equation")
            composition = _get_chemical_composition_from_ast(formula_ast)

            formula = Formula(formula_str, composition)
            counted_formulas[formula] += count
        return counted_formulas


if __name__ == "__main__":
    from pprint import pprint

    equation_str = "2H2 + O2 -> 2H2O"
    ast = get_equation_ast(equation_str)
    pprint(ast)
