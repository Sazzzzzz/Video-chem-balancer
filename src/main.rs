use chem_balancer::parser::get_chemical_composition;
fn main() {
    let formulas = vec!["Cu2+", "NH4+"];

    for formula in formulas {
        println!("\nFormula: {}", formula);

        match get_chemical_composition(formula) {
            Ok(composition) => {
                println!("Composition:");
                for (element, count) in composition {
                    println!("  {}: {}", element, count);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        println!("{}", "-".repeat(40));
    }
}
