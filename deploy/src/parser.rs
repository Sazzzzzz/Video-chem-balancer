use crate::domain::{Counter, Element, Equation, Formula};
use anyhow::{Result, anyhow};
use regex::Regex;
use serde::Serialize;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize)]
pub enum TokenType {
    Preprocessing,
    Electron,
    Element,
    Equals,
    Minus,
    Plus,
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

#[derive(Debug, Clone, Serialize)]
pub struct Token {
    pub ttype: TokenType,
    pub value: String,
}

impl Token {
    pub fn is_conjunction(&self) -> bool {
        matches!(self.ttype, TokenType::Plus | TokenType::Minus)
    }
}

pub fn tokenize(formula: &str) -> Vec<Token> {
    static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
    let regex = TOKEN_REGEX.get_or_init(|| {
        let pattern = [
            r"(?P<Preprocessing>\(aq\)|\(s\)|\(l\)|\(g\))",
            r"(?P<Electron>e-)",
            r"(?P<Element>[a-zA-Z]+)",
            r"(?P<Equals>==|->|⇌|⇄)",
            r"(?P<Minus>-)",
            r"(?P<Plus>\+)",
            r"(?P<Number>\d+)",
            r"(?P<Dot>·|\.|\*)",
            r"(?P<LParen>\()",
            r"(?P<RParen>\))",
            r"(?P<LBracket>\[)",
            r"(?P<RBracket>\])",
            r"(?P<LBrace>\{)",
            r"(?P<RBrace>\})",
            r"(?P<Whitespace>[ \t]+)",
            r"(?P<Mismatch>.)",
        ]
        .join("|");
        Regex::new(&pattern).unwrap()
    });

    let mut tokens = Vec::new();

    static REGEX_EL: OnceLock<Regex> = OnceLock::new();
    let el_re = REGEX_EL.get_or_init(|| {
        let element_regex = r"He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Th|Pa|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og|H|B|C|N|O|F|P|S|K|V|Y|I|W|U";
        Regex::new(element_regex).unwrap()
    });

    for caps in regex.captures_iter(formula) {
        let names = [
            ("Preprocessing", TokenType::Preprocessing),
            ("Electron", TokenType::Electron),
            ("Element", TokenType::Element),
            ("Equals", TokenType::Equals),
            ("Minus", TokenType::Minus),
            ("Plus", TokenType::Plus),
            ("Number", TokenType::Number),
            ("Dot", TokenType::Dot),
            ("LParen", TokenType::LParen),
            ("RParen", TokenType::RParen),
            ("LBracket", TokenType::LBracket),
            ("RBracket", TokenType::RBracket),
            ("LBrace", TokenType::LBrace),
            ("RBrace", TokenType::RBrace),
            ("Whitespace", TokenType::Whitespace),
            ("Mismatch", TokenType::Mismatch),
        ];

        let mut matched_name_val = None;
        for (name, ttype) in &names {
            if let Some(m) = caps.name(name) {
                matched_name_val = Some((*ttype, m.as_str().to_string()));
                break;
            }
        }

        if let Some((ttype, value)) = matched_name_val {
            match ttype {
                TokenType::Element => {
                    let mut pos = 0;
                    for m in el_re.find_iter(&value) {
                        if m.start() > pos {
                            tokens.push(Token {
                                ttype: TokenType::Element,
                                value: value[pos..m.start()].to_string(),
                            });
                        }
                        tokens.push(Token {
                            ttype: TokenType::Element,
                            value: m.as_str().to_string(),
                        });
                        pos = m.end();
                    }
                    if pos != value.len() {
                        tokens.push(Token {
                            ttype: TokenType::Element,
                            value: value[pos..].to_string(),
                        });
                    }
                }
                TokenType::Mismatch | TokenType::Preprocessing => {}
                _ => tokens.push(Token { ttype, value }),
            }
        }
    }

    tokens
}

#[derive(Debug, Clone)]
pub struct AstTerm {
    pub symbol: Option<String>,
    pub group: Option<Vec<AstTerm>>,
    pub count: i64,
}

#[derive(Debug, Clone)]
pub struct AstMolecule {
    pub terms: Vec<AstTerm>,
    pub charge: i64,
}

#[derive(Debug, Clone)]
pub struct AstFormula {
    pub formula: String,
    pub molecules: Vec<(AstMolecule, i64)>, // (molecule, count)
}

fn bracket_pair(t: TokenType) -> Option<TokenType> {
    match t {
        TokenType::LParen => Some(TokenType::RParen),
        TokenType::LBracket => Some(TokenType::RBracket),
        TokenType::LBrace => Some(TokenType::RBrace),
        _ => None,
    }
}

fn is_bracket(t: TokenType) -> bool {
    bracket_pair(t).is_some()
}

pub struct FormulaParser {
    tokens: Vec<Token>,
    pos: usize,
    pub is_parsing_equation: bool,
}

impl FormulaParser {
    pub fn new(tokens: Vec<Token>, is_parsing_equation: bool) -> Self {
        Self {
            tokens,
            pos: 0,
            is_parsing_equation,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn advance(&mut self) {
        self.pos += 1;
        while self.pos < self.tokens.len() {
            if self.tokens[self.pos].ttype != TokenType::Whitespace {
                return;
            }
            self.pos += 1;
        }
    }

    pub fn current_token(&self) -> Option<&Token> {
        let mut p = self.pos;
        while p < self.tokens.len() {
            if self.tokens[p].ttype != TokenType::Whitespace {
                return Some(&self.tokens[p]);
            }
            p += 1;
        }
        None
    }

    pub fn peek(&self, offset: usize) -> Option<&Token> {
        let peek_pos = self.pos + offset;
        if peek_pos < self.tokens.len() {
            Some(&self.tokens[peek_pos])
        } else {
            None
        }
    }

    pub fn peek_non_whitespace(&self, offset: usize) -> Option<&Token> {
        let mut p = self.pos + offset;
        while p < self.tokens.len() {
            if self.tokens[p].ttype != TokenType::Whitespace {
                return Some(&self.tokens[p]);
            }
            p += 1;
        }
        None
    }

    pub fn parse_formula(&mut self) -> Result<AstFormula> {
        let mut molecules = Vec::new();
        molecules.push((self.parse_charged_molecule()?, 1));

        while let Some(tok) = self.current_token() {
            if tok.ttype == TokenType::Dot {
                self.advance();
                let mut count = 1;
                if let Some(num_tok) = self.current_token()
                    && num_tok.ttype == TokenType::Number
                {
                    count = num_tok.value.parse::<i64>().unwrap_or(1);
                    self.advance();
                }
                molecules.push((self.parse_charged_molecule()?, count));
            } else {
                break;
            }
        }

        let formula_str = self.tokens[0..self.pos.min(self.tokens.len())]
            .iter()
            .map(|t| t.value.as_str())
            .collect::<String>();

        Ok(AstFormula {
            formula: formula_str,
            molecules,
        })
    }

    fn parse_charged_molecule(&mut self) -> Result<AstMolecule> {
        if let Some(tok) = self.current_token()
            && tok.ttype == TokenType::Electron
        {
            self.advance();
            return Ok(AstMolecule {
                terms: vec![],
                charge: -1,
            });
        }

        let molecule = self.parse_molecule()?;
        let mut charge = 0;

        if let Some(tok) = self.current_token()
            && (tok.ttype == TokenType::Number || tok.is_conjunction())
        {
            charge = self.parse_charge()?;
        }

        Ok(AstMolecule {
            terms: molecule,
            charge,
        })
    }

    fn parse_molecule(&mut self) -> Result<Vec<AstTerm>> {
        let mut terms = Vec::new();
        while let Some(tok) = self.current_token() {
            if is_bracket(tok.ttype) || tok.ttype == TokenType::Element {
                if self.is_parsing_equation && tok.ttype == TokenType::Equals {
                    break;
                }
                if self.is_parsing_equation && tok.is_conjunction() {
                    break;
                }
                terms.push(self.parse_term()?);
            } else {
                break;
            }
        }
        Ok(terms)
    }

    fn parse_charge(&mut self) -> Result<i64> {
        let mut charge = 1;
        if let Some(tok) = self.current_token()
            && tok.ttype == TokenType::Number
        {
            charge = tok.value.parse::<i64>().unwrap_or(1);
            self.advance();
        }

        if let Some(tok) = self.current_token() {
            if tok.is_conjunction() {
                if self.is_parsing_equation {
                    if let Some(nxt) = self.peek_non_whitespace(1) {
                        if !nxt.is_conjunction() && nxt.ttype != TokenType::Equals {
                            return Ok(0);
                        }
                    } else {
                        return Ok(0);
                    }
                }
                let sign = if tok.ttype == TokenType::Plus { 1 } else { -1 };
                charge *= sign;
                self.advance();
            } else {
                return Err(anyhow!("Expected PLUS or MINUS but found {:?}", tok));
            }
        } else {
            return Err(anyhow!("Expected PLUS or MINUS"));
        }
        Ok(charge)
    }

    fn parse_term(&mut self) -> Result<AstTerm> {
        let tok = self.current_token().cloned();
        if let Some(t) = tok {
            match t.ttype {
                TokenType::Element => self.parse_element_unit(),
                bt if is_bracket(bt) => self.parse_group_unit(),
                _ => Err(anyhow!("Unexpected token {:?} while parsing term", t)),
            }
        } else {
            Err(anyhow!("Unexpected end of input while parsing term"))
        }
    }

    fn parse_element_unit(&mut self) -> Result<AstTerm> {
        let element = self.current_token().unwrap().value.clone();
        self.advance();

        let mut count = 1;
        if let Some(tok) = self.current_token()
            && tok.ttype == TokenType::Number
        {
            if let Some(pk) = self.peek(1)
                && pk.is_conjunction()
            {
                // If the conjunction immediately after the number is itself
                // followed by whitespace, it's likely a separator between
                // compounds (e.g. "O3+ H2O2" where "+" separates compounds),
                // so treat the number as the element count. If there's no
                // whitespace after the conjunction (e.g. "Fe2+"), treat it
                // as a charge indicator and leave the number for charge parsing.
                let mut treat_as_charge = true;
                if let Some(after) = self.peek(2)
                    && after.ttype == TokenType::Whitespace
                {
                    treat_as_charge = false;
                }
                if treat_as_charge {
                    return Ok(AstTerm {
                        symbol: Some(element),
                        count,
                        group: None,
                    });
                }
            }
            count = tok.value.parse::<i64>().unwrap_or(1);
            self.advance();
        }

        Ok(AstTerm {
            symbol: Some(element),
            count,
            group: None,
        })
    }

    fn parse_group_unit(&mut self) -> Result<AstTerm> {
        let left_bracket = self.current_token().unwrap().ttype;
        let right_bracket_type = bracket_pair(left_bracket).unwrap();
        self.advance();

        let molecule = self.parse_molecule()?;
        if let Some(rb) = self.current_token() {
            if rb.ttype != right_bracket_type {
                return Err(anyhow!(
                    "Expected closing bracket {:?} but found {:?}",
                    right_bracket_type,
                    rb
                ));
            }
        } else {
            return Err(anyhow!("Expected closing bracket {:?}", right_bracket_type));
        }
        self.advance();

        let mut count = 1;
        if let Some(tok) = self.current_token()
            && tok.ttype == TokenType::Number
        {
            if let Some(pk) = self.peek(1)
                && pk.is_conjunction()
            {
                return Ok(AstTerm {
                    symbol: None,
                    count,
                    group: Some(molecule),
                });
            }
            count = tok.value.parse::<i64>().unwrap_or(1);
            self.advance();
        }
        Ok(AstTerm {
            symbol: None,
            count,
            group: Some(molecule),
        })
    }
}

pub struct FormulaBuilder;

impl FormulaBuilder {
    pub fn calculate(ast: &AstFormula) -> Counter<Element> {
        Self::evaluate_formula(ast)
    }

    fn evaluate_formula(formula: &AstFormula) -> Counter<Element> {
        let mut total = Counter::new();
        for (molecule, count) in &formula.molecules {
            let mut mol_counter = Self::evaluate_molecule(molecule);
            for v in mol_counter.values_mut() {
                *v *= *count;
            }
            for (k, v) in mol_counter {
                *total.entry(k).or_insert(0) += v;
            }
        }
        total
    }

    fn evaluate_molecule(molecule: &AstMolecule) -> Counter<Element> {
        let mut total = Counter::new();
        for term in &molecule.terms {
            let term_counter = Self::evaluate_term(term);
            for (k, v) in term_counter {
                *total.entry(k).or_insert(0) += v;
            }
        }
        if molecule.charge != 0 {
            *total.entry("charge".to_string()).or_insert(0) += molecule.charge;
        }
        total
    }

    fn evaluate_term(term: &AstTerm) -> Counter<Element> {
        let mut term_counter = Counter::new();
        if let Some(ref sym) = term.symbol {
            *term_counter.entry(sym.clone()).or_insert(0) += term.count;
        } else if let Some(ref grp) = term.group {
            let mut grp_counter = Counter::new();
            for g_term in grp {
                let curr = Self::evaluate_term(g_term);
                for (k, v) in curr {
                    *grp_counter.entry(k).or_insert(0) += v;
                }
            }
            for (k, v) in grp_counter {
                *term_counter.entry(k).or_insert(0) += v * term.count;
            }
        }
        term_counter
    }
}

pub fn get_chemical_composition(formula: &str) -> Result<Counter<Element>> {
    let tokens = tokenize(formula);
    let mut parser = FormulaParser::new(tokens, false);
    let ast = parser.parse_formula()?;
    Ok(FormulaBuilder::calculate(&ast))
}

type EquationResult = (Vec<(AstFormula, i64)>, Vec<(AstFormula, i64)>);

pub struct EquationParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl EquationParser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn current_token(&self) -> Option<&Token> {
        let mut p = self.pos;
        while p < self.tokens.len() {
            if self.tokens[p].ttype != TokenType::Whitespace {
                return Some(&self.tokens[p]);
            }
            p += 1;
        }
        None
    }

    fn advance(&mut self) {
        self.pos += 1;
        while self.pos < self.tokens.len() {
            if self.tokens[self.pos].ttype != TokenType::Whitespace {
                return;
            }
            self.pos += 1;
        }
    }

    pub fn parse_equation(&mut self) -> Result<EquationResult> {
        let reactants = self.parse_compound_list()?;

        let mut has_equals = false;
        if let Some(tok) = self.current_token()
            && tok.ttype == TokenType::Equals
        {
            has_equals = true;
            self.advance();
        }

        if !has_equals {
            return Err(anyhow!("Expected EQUALS"));
        }

        let products = self.parse_compound_list()?;

        Ok((reactants, products))
    }

    fn parse_compound_list(&mut self) -> Result<Vec<(AstFormula, i64)>> {
        let mut compounds = vec![self.parse_stoichiometric_compound(false)?];

        while let Some(tok) = self.current_token() {
            if tok.is_conjunction() {
                let is_neg = tok.ttype == TokenType::Minus;
                self.advance();
                compounds.push(self.parse_stoichiometric_compound(is_neg)?);
            } else {
                break;
            }
        }

        Ok(compounds)
    }

    fn parse_stoichiometric_compound(&mut self, invert_coeff: bool) -> Result<(AstFormula, i64)> {
        let mut count = 1;
        if let Some(tok) = self.current_token()
            && tok.ttype == TokenType::Number
        {
            count = tok.value.parse::<i64>().unwrap_or(1);
            self.advance();
        }

        let mut formula_parser = FormulaParser::new(self.tokens[self.pos..].to_vec(), true);
        let formula = formula_parser.parse_formula()?;
        self.pos += formula_parser.pos();

        let final_count = if invert_coeff { -count } else { count };
        Ok((formula, final_count))
    }
}

pub struct EquationBuilder;

impl EquationBuilder {
    pub fn build(eq_str: &str) -> Result<Equation> {
        let tokens = tokenize(eq_str);
        let mut parser = EquationParser::new(tokens);
        let (left, right) = parser.parse_equation()?;

        let reactants = Self::build_side(&left)?;
        let products = Self::build_side(&right)?;

        Ok(Equation::new(reactants, products))
    }

    fn build_side(side: &[(AstFormula, i64)]) -> Result<Counter<Formula>> {
        let mut counted = Counter::new();
        for (formula_ast, count) in side {
            let comp = FormulaBuilder::calculate(formula_ast);
            let formula = Formula {
                formula: formula_ast.formula.trim().to_string(),
                composition: comp,
            };
            if formula.formula.is_empty() {
                return Err(anyhow!("Empty formula in equation"));
            }
            *counted.entry(formula).or_insert(0) += count;
        }
        Ok(counted)
    }
}

pub fn parse_equation(equation_str: &str) -> Result<Equation> {
    EquationBuilder::build(equation_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let formula = "H2O";
        let tokens = tokenize(formula);
        let expected = [
            Token {
                ttype: TokenType::Element,
                value: "H".to_string(),
            },
            Token {
                ttype: TokenType::Number,
                value: "2".to_string(),
            },
            Token {
                ttype: TokenType::Element,
                value: "O".to_string(),
            },
        ];
        assert_eq!(tokens.len(), expected.len());
        for (t1, t2) in tokens.iter().zip(expected.iter()) {
            assert_eq!(t1.ttype, t2.ttype);
            assert_eq!(t1.value, t2.value);
        }
    }

    #[test]
    fn test_get_chemical_composition_nh4() {
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
        assert_eq!(*composition.get("K").unwrap_or(&0), 1);
        assert_eq!(*composition.get("Al").unwrap_or(&0), 1);
        assert_eq!(*composition.get("S").unwrap_or(&0), 2);
        assert_eq!(*composition.get("O").unwrap_or(&0), 20);
        assert_eq!(*composition.get("H").unwrap_or(&0), 24);
    }

    #[test]
    fn test_get_chemical_composition_cu_en() {
        let composition = get_chemical_composition("Cu(en)2 2+").unwrap();
        assert_eq!(*composition.get("Cu").unwrap_or(&0), 1);
        assert_eq!(*composition.get("en").unwrap_or(&0), 2);
        assert_eq!(*composition.get("charge").unwrap_or(&0), 2);
    }

    #[test]
    fn test_get_chemical_composition_etoet() {
        let composition = get_chemical_composition("EtOEt").unwrap();
        assert_eq!(*composition.get("Et").unwrap_or(&0), 2);
        assert_eq!(*composition.get("O").unwrap_or(&0), 1);
    }

    #[test]
    fn test_get_chemical_composition_fe3() {
        let composition = get_chemical_composition("Fe 3+").unwrap();
        assert_eq!(*composition.get("Fe").unwrap_or(&0), 1);
        assert_eq!(*composition.get("charge").unwrap_or(&0), 3);
    }
}
