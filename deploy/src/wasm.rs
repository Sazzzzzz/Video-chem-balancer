use crate::parser::parse_equation;
use crate::solver::balance;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct BalanceResult {
    balanced: Vec<String>,
    error: Option<String>,
}

#[wasm_bindgen]
pub fn balance_equation(input: &str) -> String {
    match parse_equation(input) {
        Ok(eq) => {
            let results = balance(&eq);
            if results.is_empty() {
                serde_json::to_string(&BalanceResult {
                    balanced: vec![],
                    error: Some("No valid integer balanced solution found.".into()),
                })
                .unwrap()
            } else {
                let balanced: Vec<String> = results.iter().map(|r| r.to_string()).collect();
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
