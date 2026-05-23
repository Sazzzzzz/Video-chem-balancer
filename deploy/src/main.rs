use chem_balancer::parser::parse_equation;
use chem_balancer::solver::balance;
use std::env;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        let eq_str = args[1..].join(" ");
        process_equation(&eq_str);
    } else {
        println!("Chemical Equation Balancer REPL");
        println!("Enter a chemical equation to balance, or type 'q' to quit.");

        loop {
            print!("> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let eq_str = input.trim();
            if eq_str.eq_ignore_ascii_case("q") || eq_str.eq_ignore_ascii_case("quit") {
                break;
            }
            if eq_str.is_empty() {
                continue;
            }

            process_equation(eq_str);
        }
    }
}

fn process_equation(eq_str: &str) {
    println!("Parsing equation: {}", eq_str);
    match parse_equation(eq_str) {
        Ok(eq) => {
            let balanced_eqs = balance(&eq);
            if balanced_eqs.is_empty() {
                println!("No valid integer balanced solution found.");
            } else {
                for b_eq in balanced_eqs {
                    println!("{}", b_eq);
                }
            }
        }
        Err(e) => {
            println!("Failed to parse equation: {}", e);
        }
    }
}
