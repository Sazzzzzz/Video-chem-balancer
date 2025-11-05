from copy import deepcopy
from itertools import chain
from typing import overload

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
    # So they say combination is better than inheritance...
    @overload
    def __init__(
        self, *, reactants: Counter[Formula], products: Counter[Formula]
    ) -> None: ...

    @overload
    def __init__(self, equation_str: str) -> None: ...

    def __init__(
        self,
        equation_str: str | None = None,
        *,
        reactants: Counter[Formula] | None = None,
        products: Counter[Formula] | None = None,
    ):
        if equation_str is not None:
            ast = get_equation_ast(equation_str)
            base = EquationBuilder(ast).build()
            super().__init__(base.reactants, base.products)
        elif reactants is not None and products is not None:
            super().__init__(reactants, products)
        else:
            raise ValueError(
                "Either equation_str or reactants and products must be provided."
            )

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
    def elements(self) -> list[Element]:
        """Elements involved in the equation. Ordered by occurrence in reactants and then products."""
        return list(
            dict.fromkeys(
                chain.from_iterable(f.composition.keys() for f in self.substances)
            )
        )

    @property
    def substances(self) -> list[Formula]:
        """Substances involved in the equation. Ordered by occurrence in reactants and then products."""
        return list(chain(self.reactants.keys(), self.products.keys()))

    @property
    def coeff_matrix(self) -> sp.Matrix:
        """Coefficient matrix representing the equation."""
        return sp.Matrix(
            [
                [substance.composition.get(element, 0) for substance in self.substances]
                for element in self.elements
            ]
        )

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
        solutions = self.coeff_matrix.nullspace()
        equations: list[Equation] = []
        for solution in solutions:
            equation = deepcopy(self)
            lcm = sp.lcm([term.q for term in solution])
            sign = 1 if solution[0] > 0 else -1
            solution = [term * sign * lcm for term in solution]
            counted_formulas = Counter(
                {f: c for f, c in zip(equation.substances, solution)}
            )
            balanced_equation = Equation.from_counted_formula(counted_formulas)
            equations.append(balanced_equation)
        return equations

    def exchange_sides(self) -> "Equation":
        """Helper function which returns a new Equation with reactants and products exchanged."""
        return self.__class__(reactants=self.products, products=self.reactants)

    # To determine whether an equation is solvable by observation, we start from rows with only two non-zero entries in the coefficient matrix.
    # Another row is confirmed solvable if it contains at most one non-zero entry besides current confirmed entries. A equation is fully solvable if all rows are confirmed.

    # This problem may be related to:
    # 1. Topological sorting (also related to univerity course scheduling problem)
    # 2. Disjoint set union (equivalence class)

    # The following code demostrate a solution of greedy method. In this procedure, confirmed rows are monotonically increased, and new confirmed rows will not affect the previous confirmed rows. Thus by testing for all initial confirmed rows, we can determine whether the equation is fully solvable.
    # todo: Topological sorting and disjoint set union
    def find_observable_solution_order(self) -> list[Formula]:
        """Determine whether the equation is solvable by observation.
        If solvable, return one possible order of substances to observe."""
        row_indexes = {
            row_index: set(col_index for col_index, elem in enumerate(row) if elem != 0)
            for row_index, row in enumerate(self.coeff_matrix.tolist())
        }
        possible_starts = {
            row_index: col_indexes
            for row_index, col_indexes in row_indexes.items()
            if len(col_indexes) == 2
        }
        if possible_starts == {}:
            return []

        def confirm(
            confirmed_cols: set[int], rows_remained: set[int], chain: list[int]
        ) -> list[int]:
            while rows_remained:
                for row in rows_remained:
                    determined = row_indexes[row] - confirmed_cols
                    if len(determined) <= 1:
                        confirmed_cols |= row_indexes[row]
                        rows_remained.remove(row)
                        chain.append(determined.pop()) if determined else None
                        break
                else:
                    break
            if len(chain) == len(self.substances):
                return chain
            else:
                return []

        for start_row in possible_starts.keys():
            confirmed_cols: set[int] = row_indexes[start_row].copy()
            rows_remained: set[int] = set(row_indexes.keys()) - {start_row}
            chain: list[int] = list(row_indexes[start_row])
            if result := confirm(confirmed_cols, rows_remained, chain):
                return [self.substances[i] for i in result]
        return []


if __name__ == "__main__":
    equation_str = "ClO3 - + Cl- + H+ == Cl2 + H2O"
    equation = Equation(equation_str)
    print(f"{equation.balance()[0]}")
