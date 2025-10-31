from copy import deepcopy
from itertools import chain

import sympy as sp

from chem_parser import (
    BaseEquation,
    Element,
    EquationBuilder,
    Formula,
    get_equation_ast,
)
from utils import Counter, scale


class Equation(BaseEquation):
    def __init__(self, equation_str):
        ast = get_equation_ast(equation_str)
        base = EquationBuilder(ast).build()
        super().__init__(base.reactants, base.products)

    @classmethod
    def from_counted_formula(cls, counted_formulas: Counter[Formula]) -> "Equation":
        reactants: Counter[Formula] = Counter()
        products: Counter[Formula] = Counter()
        for f, c in counted_formulas.items():
            if c >= 0:
                reactants[f] = c
            else:
                products[f] = -c
        inst = cls.__new__(cls)
        super(Equation, inst).__init__(reactants, products)
        return inst

    @property
    def elements(self) -> set[Element]:
        return set(
            element
            for compound in self.reactants + self.products
            for element in compound.composition.keys()
        )

    @property
    def substances(self) -> chain[Formula]:
        return chain(self.reactants.keys(), self.products.keys())

    def is_balanced(self) -> bool:
        """Check if the equation is balanced."""
        reactant_counts = sum(
            (scale(f.composition, c) for f, c in self.reactants.items()),
            Counter(),
        )
        product_counts = sum(
            (scale(f.composition, c) for f, c in self.products.items()),
            Counter(),
        )
        return reactant_counts == product_counts

    def balance(self) -> list["Equation"]:
        """Balances the chemical equation using sympy with rref method."""
        coeff_matrix = sp.Matrix(
            [
                [substance.composition.get(element, 0) for substance in self.substances]
                for element in self.elements
            ]
        )
        solutions = coeff_matrix.nullspace()
        equations: list[Equation] = []
        for solution in solutions:
            equation = deepcopy(self)
            lcm = sp.lcm([term.q for term in solution])
            solution = [term * lcm for term in solution]
            counted_formulas = Counter(
                {f: c for f, c in zip(equation.substances, solution)}
            )
            balanced_equation = Equation.from_counted_formula(counted_formulas)
            equations.append(balanced_equation)
        return equations


if __name__ == "__main__":
    equation_str = "Cl- + ClO3 - + H+ + Cl2O + O2 -> Cl2 + H2O + ClO2"
    equation = Equation(equation_str)
    eqs = equation.balance()
    for eq in eqs:
        print(eq)
