from collections import Counter
from itertools import chain
from copy import deepcopy

import sympy as sp

from chem_parser import (
    BaseEquation,
    CountedFormula,
    EquationBuilder,
    get_equation_ast,
    scale_counter,
)


class Equation(BaseEquation):
    def __init__(self, equation_str):
        ast = get_equation_ast(equation_str)
        base = EquationBuilder(ast).build()
        super().__init__(base.reactants, base.products)

    @property
    def elements(self):
        return set(
            element
            for compound in self.reactants + self.products
            for element in compound.composition.keys()
        )

    @property
    def substances(self) -> list[Counter[str]]:
        return list(
            chain(
                (cf.composition for cf in self.reactants),
                (cf.composition for cf in self.products),
            )
        )

    @property
    def counted_substances(self) -> list[CountedFormula]:
        return list(chain(self.reactants, self.products))

    def is_balanced(self) -> bool:
        """Check if the equation is balanced."""
        reactant_counts = sum(
            (scale_counter(cf.composition, cf.count) for cf in self.reactants),
            Counter(),
        )
        product_counts = sum(
            (scale_counter(cf.composition, cf.count) for cf in self.products), Counter()
        )
        return reactant_counts == product_counts

    def balance(self) -> list["Equation"]:
        """Balances the chemical equation using sympy with rref method."""
        coeff_matrix = sp.Matrix(
            [
                [substance.get(element, 0) for substance in self.substances]
                for element in self.elements
            ]
        )
        solutions = coeff_matrix.nullspace()
        equations: list[Equation] = []
        for solution in solutions:
            equation = deepcopy(self)
            lcm = sp.lcm([term.q for term in solution])
            solution = [term * lcm for term in solution]
            for i, substance in enumerate(equation.counted_substances):
                substance.count = abs(solution[i])
            equations.append(equation)
        return equations


if __name__ == "__main__":
    equation_str = "Cl- + H+ + ClO3 - + Cl2O + O2 == ClO2 + Cl2 + H2O"
    equation = Equation(equation_str)
    eqs = equation.balance()
    for eq in eqs:
        print(eq)
