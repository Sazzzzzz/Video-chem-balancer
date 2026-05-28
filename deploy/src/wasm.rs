use crate::domain::Equation;
use crate::parser::{Token, parse_equation, tokenize};
use crate::solver::{hilbert_balance, matrix_balance};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct BalanceResult {
    balanced: Vec<Vec<Token>>,
    error: Option<String>,
}

fn equation_to_tokens(eq: &Equation) -> Vec<Token> {
    tokenize(&eq.to_string())
}

#[wasm_bindgen]
pub fn tokenize_equation(input: &str) -> String {
    let tokens = tokenize(input);
    serde_json::to_string(&tokens).unwrap_or_else(|_e| "[]".to_string())
}

#[wasm_bindgen]
pub fn balance(input: &str, method: &str) -> String {
    match parse_equation(input) {
        Ok(eq) => {
            let results = match method {
                "matrix" => matrix_balance(&eq),
                "hilbert" => hilbert_balance(&eq),
                _ => {
                    return serde_json::to_string(&BalanceResult {
                        balanced: vec![],
                        error: Some("Invalid balancing method specified.".into()),
                    })
                    .unwrap();
                }
            };

            if results.is_empty() {
                serde_json::to_string(&BalanceResult {
                    balanced: vec![],
                    error: Some("No valid integer balanced solution found.".into()),
                })
                .unwrap()
            } else {
                let balanced: Vec<Vec<Token>> = results.iter().map(equation_to_tokens).collect();
                serde_json::to_string(&BalanceResult {
                    balanced,
                    error: None,
                })
                .unwrap()
            }
        }
        Err(e) => serde_json::to_string(&BalanceResult {
            balanced: vec![],
            error: Some(format!("Failed to parse equation: {}", e)),
        })
        .unwrap(),
    }
}
