use std::collections::BTreeMap;
use std::fmt;

pub type Element = String;
pub type Counter<T> = BTreeMap<T, i64>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Formula {
    pub formula: String,
    pub composition: Counter<Element>,
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formula)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Equation {
    pub reactants: Counter<Formula>,
    pub products: Counter<Formula>,
}

impl Equation {
    pub fn new(reactants: Counter<Formula>, products: Counter<Formula>) -> Self {
        Self {
            reactants,
            products,
        }
    }

    /// Returns a list of all distinct elements in the equation
    pub fn elements(&self) -> Vec<Element> {
        let mut elems = Vec::new();
        let handle_side = |side: &Counter<Formula>, elems: &mut Vec<Element>| {
            for formula in side.keys() {
                for el in formula.composition.keys() {
                    if !elems.contains(el) {
                        elems.push(el.clone());
                    }
                }
            }
        };
        handle_side(&self.reactants, &mut elems);
        handle_side(&self.products, &mut elems);
        elems
    }

    /// Returns all substances involved in the equation (reactants then products)
    pub fn substances(&self) -> Vec<Formula> {
        let mut subs = Vec::new();
        for f in self.reactants.keys() {
            subs.push(f.clone());
        }
        for f in self.products.keys() {
            subs.push(f.clone());
        }
        subs
    }
}

impl fmt::Display for Equation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_side = |side: &Counter<Formula>| -> String {
            let is_single = side.len() == 1;
            let mut parts = Vec::new();
            for (form, &count) in side {
                if count != 0 || is_single {
                    let prefix = if count == 1 {
                        String::new()
                    } else {
                        count.to_string()
                    };
                    parts.push(format!("{}{}", prefix, form));
                }
            }
            parts.join(" + ")
        };

        let lhs = format_side(&self.reactants);
        let rhs = format_side(&self.products);
        write!(f, "{} == {}", lhs, rhs)
    }
}
