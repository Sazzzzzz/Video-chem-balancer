use std::{str::FromStr, sync::LazyLock};
/// A parser for chemical formulas.
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

const ELEMENTS_STR: &str =
    "He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa Np Pu Am Cm Bk Cf Es Fm Md No Lr
Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og H B C N O F P S K V Y I W U";
// constants in rust
static ELEMENTS: LazyLock<Vec<&str>> = LazyLock::new(|| {
    ELEMENTS_STR
        .split(char::is_whitespace)
        .collect()
});

#[derive(EnumIter, Debug, Display, EnumString)]
enum TokenType {
    Element,
    Charge,
    Number,
    Dot,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Whitespace,
    Mismatch,
}

#[derive(Debug)]
struct Token {
    kind: TokenType,
    value: String,
}
impl TokenType {
    const fn regex(&self) -> &'static str {
        match self {
            TokenType::Element => r"[a-zA-Z]+",
            TokenType::Number => r"\d+",
            TokenType::Charge => r"\d*[+-]",
            TokenType::Dot => r"·|\.|\*",
            TokenType::LParen => r"\(",
            TokenType::RParen => r"\)",
            TokenType::LBracket => r"\[",
            TokenType::RBracket => r"\]",
            TokenType::LBrace => r"\{",
            TokenType::RBrace => r"\}",
            TokenType::Whitespace => r"\s+",
            TokenType::Mismatch => r".",
        }
    }
}

fn tokenize(formula: &str) -> Vec<Token> {
    let regex_str = TokenType::iter()
        .map(|t| format!("(?P<{}>{})", t.to_string(), t.regex()))
        .collect::<Vec<_>>()
        .join("|");
    let token_regex = regex::Regex::new(&regex_str).unwrap();

    token_regex
        .captures_iter(formula)
        .flat_map(|caps| {
            let (kind, value) = TokenType::iter()
                .find_map(|token_type| {
                    let name = token_type.to_string();
                    caps.name(&name)
                        .map(|m| (TokenType::from_str(&name).unwrap(), m.as_str()))
                })
                .expect("No matching group found during tokenization");

            match kind {
                TokenType::Element => {
                    let element_regex_str = format!("{}|[a-zA-Z]+", ELEMENTS.join("|"));
                    println!("Regex pattern: '{}'", element_regex_str);
                    let element_regex = regex::Regex::new(&element_regex_str).unwrap();

                    element_regex
                        .find_iter(value)
                        .map(|m| Token {
                            kind: TokenType::Element,
                            value: m.as_str().to_string(),
                        })
                        .collect()
                }
                TokenType::Whitespace => Vec::new(), // Skip whitespace (empty vec)
                TokenType::Mismatch => panic!("Unexpected character: {}", value),
                _ => vec![Token {
                    kind,
                    value: value.to_string(),
                }], // Single token wrapped in Vec
            }
        })
        .collect()
}

fn main() {
    let formula = "Cu (NO3)2";
    let tokens = tokenize(formula);
    for token in tokens {
        println!("{:?}", token);
    }
}
