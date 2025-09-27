/// A parser for chemical formulas.
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};
#[derive(EnumIter, Debug, Display)]
enum Token {
    Element { symbol: String },
    Number { value: u32 },
}

impl Token {
    const fn name(&self) -> &'static str {
        match self {
            Token::Element { .. } => "Element",
            Token::Number { .. } => "Number",
        }
    }
    const fn regex(&self) -> &'static str {
        match self {
            Token::Element { .. } => r"[A-Z][a-z]?",
            Token::Number { .. } => r"\d+",
        }
    }
}

fn tokenizer(formula: &str) -> Vec<Token> {
    let regex_str = Token::iter()
        .map(|t| format!("(?P<{}>{})", t.name(), t.regex()))
        .collect::<Vec<_>>()
        .join("|");
    let token_regex = regex::Regex::new(&regex_str).unwrap();
    token_regex
        .captures_iter(formula)
        .map(|caps| {
            // To identify which token kind was matched, we iterate over all token types
            // A more efficient approach could be used if performance is critical
            for token in Token::iter() {
                if let Some(mat) = caps.name(token.name()) {
                    let mat_str = mat.as_str();
                    return match token {
                        Token::Element { .. } => Token::Element {
                            symbol: mat_str.to_string(),
                        },
                        Token::Number { .. } => Token::Number {
                            value: mat_str.parse().unwrap(),
                        },
                    };
                }
            }
            unreachable!()
        })
        .collect()
}

fn main() {
    let formula = "C6H12O6";
    let tokens = tokenizer(formula);
    for token in tokens {
        println!("{:?}", token);
    }
}
