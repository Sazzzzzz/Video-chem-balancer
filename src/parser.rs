use counter::Counter;
use std::{str::FromStr, sync::LazyLock};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};
#[derive(EnumIter, Debug, Display, EnumString, PartialEq, Clone)]
pub enum TokenType {
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

#[derive(Debug, Clone)]
pub struct Token {
    kind: TokenType,
    value: String,
}

impl Token {
    fn new(kind: TokenType, value: String) -> Self {
        Token { kind, value }
    }
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

    const fn get_closing_bracket(opening: &TokenType) -> Option<TokenType> {
        match opening {
            TokenType::LParen => Some(TokenType::RParen),
            TokenType::LBracket => Some(TokenType::RBracket),
            TokenType::LBrace => Some(TokenType::RBrace),
            _ => None,
        }
    }
}

// ----- Tokenization -----
const ELEMENTS_STR: &str =
    "He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa Np Pu Am Cm Bk Cf Es Fm Md No Lr
Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og H B C N O F P S K V Y I W U";

static ELEMENTS: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| ELEMENTS_STR.split_whitespace().collect());

// Lazy-initialized regex for all tokens
static TOKEN_REGEX: LazyLock<regex::Regex> = LazyLock::new(|| {
    let regex_str = TokenType::iter()
        .map(|t| format!("(?P<{}>{})", t.to_string(), t.regex()))
        .collect::<Vec<_>>()
        .join("|");
    regex::Regex::new(&regex_str).unwrap()
});
// todo: better strategy from string to TokenType
// Lazy-initialized regex specifically for splitting element groups
static ELEMENT_REGEX: LazyLock<regex::Regex> = LazyLock::new(|| {
    let element_regex_str = format!("{}", ELEMENTS.join("|"));
    regex::Regex::new(&element_regex_str).unwrap()
});

pub fn tokenize(formula: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();

    for caps in TOKEN_REGEX.captures_iter(formula) {
        // find matching token type and value
        let (kind, value) = TokenType::iter()
            .find_map(|token_type| {
                let name = token_type.to_string();
                caps.name(&name)
                    .map(|mat| (TokenType::from_str(&name).unwrap(), mat.as_str()))
            })
            .ok_or("Failed to match token")?;

        // return tokens based on type
        match kind {
            // First split based on element regex
            // push every unknown part as new Element token
            TokenType::Element => {
                let mut pos = 0;
                ELEMENT_REGEX.find_iter(value).for_each(|m| {
                    if m.start() != pos {
                        tokens.push(Token::new(
                            TokenType::Element,
                            value[pos..m.start()].to_string(),
                        ))
                    }
                    tokens.push(Token::new(TokenType::Element, m.as_str().to_string()));
                    pos = m.end();
                });
                if pos != value.len() {
                    tokens.push(Token::new(TokenType::Element, value[pos..].to_string()))
                }
            }

            TokenType::Whitespace => {} // Skip whitespace
            TokenType::Mismatch => {
                return Err(format!("Invalid character in formula: '{}'", value));
            }
            _ => tokens.push(Token::new(kind, value.to_string())),
        }
    }
    Ok(tokens)
}

// ----- Parsing -----

// Strongly typed AST structures
#[derive(Debug, Clone)]
pub struct Formula {
    molecules: Vec<MoleculeInfo>,
}

#[derive(Debug, Clone)]
pub struct MoleculeInfo {
    molecule: Molecule,
    count: u32,
}

#[derive(Debug, Clone)]
pub struct Molecule {
    terms: Vec<Term>,
    charge: i32,
}

#[derive(Debug, Clone)]
pub enum Term {
    Element { symbol: String, count: u32 },
    Group { formula: Formula, count: u32 },
}

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            position: 0,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    /// formula -> molecule (DOT \[NUMBER\] molecule)*
    fn parse_formula(&mut self) -> Result<Formula, String> {
        let mut molecules = vec![MoleculeInfo {
            molecule: self.parse_molecule()?,
            count: 1,
        }];

        while let Some(Token {
            kind: TokenType::Dot,
            ..
        }) = self.current_token()
        {
            self.advance(); // Consume DOT
            let mut count = 1;
            if let Some(
                token @ Token {
                    kind: TokenType::Number,
                    ..
                },
            ) = self.current_token()
            {
                count = token.value.parse().map_err(|_| "Invalid number")?;
                if count == 0 {
                    return Err("Hydrate count must be greater than zero".to_string());
                }
                self.advance(); // Consume NUMBER
            }
            molecules.push(MoleculeInfo {
                molecule: self.parse_molecule()?,
                count,
            });
        }

        Ok(Formula { molecules })
    }

    /// molecule -> term+ \[CHARGE\]
    fn parse_molecule(&mut self) -> Result<Molecule, String> {
        let mut terms = Vec::new();

        while let Some(token) = self.current_token() {
            match token.kind {
                TokenType::Element => {
                    terms.push(self.parse_element_unit()?);
                }
                TokenType::LParen | TokenType::LBracket | TokenType::LBrace => {
                    terms.push(self.parse_group_unit()?);
                }
                _ => break,
            }
        }

        let mut charge = 0;
        if let Some(
            token @ Token {
                kind: TokenType::Charge | TokenType::Number,
                ..
            },
        ) = self.current_token()
        {
            charge = self.parse_charge(&token.value)?;
        }

        Ok(Molecule { terms, charge })
    }

    /// CHARGE: \d*[+-]
    fn parse_charge(&self, value: &str) -> Result<i32, String> {
        match value {
            "+" => Ok(1),
            "-" => Ok(-1),
            _ if value.ends_with('+') || value.ends_with('-') => {
                let (num_str, sign) = value.split_at(value.len() - 1);
                let magnitude = if num_str.is_empty() {
                    1
                } else {
                    num_str.parse::<i32>().map_err(|_| "Invalid charge number")?
                };
                Ok(if sign == "+" { magnitude } else { -magnitude })
            }
            _ => Err("Invalid charge format".to_string()),
        }
    }

    /// element_unit -> ELEMENT [NUMBER\]
    fn parse_element_unit(&mut self) -> Result<Term, String> {
        let element_token = self
            .current_token()
            .ok_or("Unexpected end of input while parsing element")?;

        if element_token.kind != TokenType::Element {
            return Err("Expected element".to_string());
        }

        let element = element_token.value.clone();
        self.advance(); // Consume ELEMENT

        let mut count = 1;
        if let Some(
            token @ Token {
                kind: TokenType::Number,
                ..
            },
        ) = self.current_token()
        {
            count = token.value.parse().map_err(|_| "Invalid number")?;
            self.advance(); // Consume NUMBER
        }

        Ok(Term::Element {
            symbol: element,
            count,
        })
    }

    /// group_unit -> LPAREN formula RPAREN \[NUMBER\]
    fn parse_group_unit(&mut self) -> Result<Term, String> {
        let left_bracket = self
            .current_token()
            .ok_or("Unexpected end of input while parsing group")?;
        let right_bracket_type =
            TokenType::get_closing_bracket(&left_bracket.kind).ok_or("Invalid opening bracket")?;

        self.advance(); // Consume left bracket

        let formula = self.parse_formula()?;

        let right_bracket = self.current_token().ok_or("Expected closing bracket")?;
        if right_bracket.kind != right_bracket_type {
            return Err(format!(
                "Expected closing bracket {:?} but found {:?}",
                right_bracket_type, right_bracket.kind
            ));
        }

        self.advance(); // Consume right bracket

        let mut count = 1;
        if let Some(
            token @ Token {
                kind: TokenType::Number,
                ..
            },
        ) = self.current_token()
        {
            count = token.value.parse().map_err(|_| "Invalid number")?;
            self.advance(); // Consume NUMBER
        }

        Ok(Term::Group { formula, count })
    }
}

pub fn get_ast(formula: &str) -> Result<Formula, String> {
    let tokens = tokenize(formula)?;
    let mut parser = Parser::new(tokens);
    parser.parse_formula()
}

// ----- Calculator -----

struct Calculator {
    ast: Formula,
}

impl Calculator {
    fn new(ast: Formula) -> Self {
        Calculator { ast }
    }

    fn calculate(&self) -> Counter<String, i32> {
        self.evaluate_formula(&self.ast)
    }

    fn evaluate_formula(&self, formula: &Formula) -> Counter<String, i32> {
        formula
            .molecules
            .iter()
            .flat_map(|molecule_info| {
                self.evaluate_molecule(&molecule_info.molecule)
                    .into_iter()
                    .map(move |(element, qty)| (element, qty * molecule_info.count as i32))
            })
            .collect()
    }

    fn evaluate_molecule(&self, molecule: &Molecule) -> Counter<String, i32> {
        let mut counter: Counter<String, i32> = molecule
            .terms
            .iter()
            .flat_map(|term| self.evaluate_term(term))
            .collect();

        if molecule.charge != 0 {
            counter.insert("charge".to_string(), molecule.charge);
        }

        counter
    }

    fn evaluate_term(&self, term: &Term) -> Counter<String, i32> {
        match term {
            Term::Element { symbol, count } => {
                Counter::from_iter([(symbol.clone(), *count as i32)])
            }
            Term::Group { formula, count } => self
                .evaluate_formula(formula)
                .into_iter()
                .map(|(element, qty)| (element, qty * (*count as i32)))
                .collect(),
        }
    }
}

/// Get the chemical composition of a formula as a Counter.
/// The Counter maps element symbols to their counts, and includes "charge" if applicable.
/// # Examples
/// ```
/// let composition = chem_balancer::parser::get_chemical_composition("Fe3+").unwrap();
/// assert_eq!(*composition.get("Fe").unwrap(), 1);
/// assert_eq!(*composition.get("charge").unwrap(), 3);
/// ```
pub fn get_chemical_composition(formula: &str) -> Result<Counter<String, i32>, String> {
    let ast = get_ast(formula)?;
    let calculator = Calculator::new(ast);
    Ok(calculator.calculate())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tokenize() {
        let formula = "H2O";
        let tokens = tokenize(formula).unwrap();
        let expected = vec![
            Token::new(TokenType::Element, "H".to_string()),
            Token::new(TokenType::Number, "2".to_string()),
            Token::new(TokenType::Element, "O".to_string()),
        ];
        assert_eq!(tokens.len(), expected.len());
        for (t1, t2) in tokens.iter().zip(expected.iter()) {
            assert_eq!(t1.kind, t2.kind);
            assert_eq!(t1.value, t2.value);
        }
    }
    #[test]
    fn test_get_chemical_composition_nh4() {
        //* NH4+ must be input as "NH4 +"
        //* if NH4+ is parsed as NH4 +, then Fe3+ would never be parsed as Fe 3+
        let composition = get_chemical_composition("NH4 +").unwrap();
        assert_eq!(*composition.get("N").unwrap(), 1);
        assert_eq!(*composition.get("H").unwrap(), 4);
        assert_eq!(*composition.get("charge").unwrap(), 1);
    }
    #[test]
    fn test_get_chemical_composition_ethanol() {
        let composition = get_chemical_composition("CH3CH2OH").unwrap();
        assert_eq!(*composition.get("C").unwrap(), 2);
        assert_eq!(*composition.get("H").unwrap(), 6);
        assert_eq!(*composition.get("O").unwrap(), 1);
    }
    #[test]

    fn test_get_chemical_composition_alum() {
        let composition = get_chemical_composition("KAl(SO4)2.12H2O").unwrap();
        assert_eq!(*composition.get("K").unwrap(), 1);
        assert_eq!(*composition.get("Al").unwrap(), 1);
        assert_eq!(*composition.get("S").unwrap(), 2);
        assert_eq!(*composition.get("O").unwrap(), 20);
        assert_eq!(*composition.get("H").unwrap(), 24);
    }
    #[test]
    fn test_get_chemical_composition_cu_en() {
        let composition = get_chemical_composition("Cu(en)2 2+").unwrap();
        assert_eq!(*composition.get("Cu").unwrap(), 1);
        assert_eq!(*composition.get("en").unwrap(), 2);
        assert_eq!(*composition.get("charge").unwrap(), 2);
    }
    #[test]
    fn test_get_chemical_composition_etoet() {
        let composition = get_chemical_composition("EtOEt").unwrap();
        assert_eq!(*composition.get("Et").unwrap(), 2);
        assert_eq!(*composition.get("O").unwrap(), 1);
    }
    #[test]
    fn test_get_chemical_composition_fe3() {
        let composition = get_chemical_composition("Fe 3+").unwrap();
        assert_eq!(*composition.get("Fe").unwrap(), 1);
        assert_eq!(*composition.get("charge").unwrap(), 3);
    }
}
