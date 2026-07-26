/*
 * Copyright 2018-2026 CREATOR Team.
 *
 * This file is part of CREATOR.
 *
 * CREATOR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CREATOR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CREATOR.  If not, see <http://www.gnu.org/licenses/>.
 */

//! Module containing the interpreter engine used to expand pseudoinstructions
//!
//! The entry point for expanding pseudoinstructions is the [`expand()`] function

use num_bigint::BigUint;

use crate::architecture::Pseudoinstruction;
use crate::number::Number;
use crate::parser::{Instruction, ParseError, Token};
use crate::span::Range;

use super::{Context, ErrorData, ErrorKind, InstructionDefinition};
use super::{Expr, ParsedArgs};
use super::{Span, Spanned};

/// Pseudoinstruction evaluation error kind
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    EvaluationError(String),
    ParseError(ParseError),
}

/// Pseudoinstruction evaluation error
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Definition of the string at the point of the error
    pub definition: String,
    /// Location in the definition that caused the error
    pub span: Range,
    /// Type of the error
    pub kind: Kind,
}

impl Error {
    /// Adds a span and a pseudoinstruction name to the error, promoting it to a [`ErrorData`]
    ///
    /// # Parameters
    ///
    /// * `def`: definition of the pseudoinstruction
    /// * `span`: location in the assembly code that caused the error
    #[must_use]
    fn compile_error(self, def: &Pseudoinstruction, span: Span) -> ErrorData {
        ErrorKind::PseudoinstructionError {
            name: def.name.to_owned(),
            error: Box::new(self),
        }
        .add_span(span)
    }
}

#[cfg(feature = "pseudoinstructions")]
mod js {
    use js_sys::{wasm_bindgen::JsValue, Array, BigInt};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// Converts a `JS` value to a `JS` string
        #[wasm_bindgen(js_name = String)]
        fn string(x: JsValue) -> js_sys::JsString;
    }

    // Function
    // NOTE: Modification of [`js_sys::Function`] to add `catch` to the constructor
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(extends = js_sys::Object, is_type_of = JsValue::is_function, typescript_type = "Function")]
        #[derive(Clone, Debug, PartialEq, Eq)]
        type Function;

        /// The `Function` constructor creates a new `Function` object. Calling the
        /// constructor directly can create functions dynamically, but suffers from
        /// security and similar (but far less significant) performance issues
        /// similar to `eval`. However, unlike `eval`, the `Function` constructor
        /// allows executing code in the global scope, prompting better programming
        /// habits and allowing for more efficient code minification.
        ///
        /// [MDN documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function)
        #[wasm_bindgen(constructor, catch)]
        fn new_with_args(args: &str, body: &str) -> Result<Function, JsValue>;

        /// The `call()` method calls a function with a given this value and
        /// arguments provided individually.
        ///
        /// [MDN documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function/call)
        #[wasm_bindgen(method, catch, js_name = call)]
        fn call2(
            this: &Function,
            context: &JsValue,
            arg1: &BigInt,
            arg2: &Array,
        ) -> Result<JsValue, JsValue>;
    }

    /// Evaluates code corresponding to the body of a `JS` function with the address and arguments array
    ///
    /// # Parameters
    ///
    /// * `src`: function body to evaluate
    /// * `address`: address in which the pseudoinstruction is being compiled into
    /// * `args`: `JS` array with the arguments of the pseudoinstruction
    ///
    /// # Errors
    ///
    /// Errors if there is any exception during parsing/execution of the code
    pub fn eval_fn(src: &str, address: &super::BigUint, args: &Array) -> Result<String, String> {
        let to_string = |x| String::from(string(x));
        Function::new_with_args("pc, args", src)
            .map_err(to_string)?
            .call2(&JsValue::TRUE, &crate::js::to_js_bigint(address), args)
            .map_err(to_string)
            .map(to_string)
    }
}

/// Result of expanding a pseudoinstruction
type ExpandedInstructions<'a> = Vec<Spanned<(InstructionDefinition<'a>, ParsedArgs)>>;

/// Expands a pseudoinstruction to a sequence of instructions, which might be real or another
/// pseudoinstruction
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `address`: address in which the instruction is being compiled into
/// * `instruction`: pseudoinstruction definition to use
/// * `args`: arguments of the instruction being expanded
///
/// # Errors
///
/// Errors if there is any problem expanding the pseudoinstruction
pub fn expand<'arch>(
    ctx: &mut Context<'arch>,
    address: &BigUint,
    instruction: (&'arch Pseudoinstruction, Span),
    args: &ParsedArgs,
) -> Result<ExpandedInstructions<'arch>, ErrorData> {
    const JS_PREFIX: &str = "js:\n";
    // Function to evaluate a label within an expression
    let ident_eval = |label: &str| super::label_eval(&ctx.label_table, address, label);

    let (instruction, span) = instruction;
    let arch = ctx.arch;

    // Expansion
    let def = &instruction.definition;
    let def = if let Some(body) = def.strip_prefix(JS_PREFIX) {
        if cfg!(feature = "pseudoinstructions") {
            use js_sys::{wasm_bindgen::JsValue, Array};

            let js_args: Array = args
                .iter()
                .map(|arg| match arg.value.0.eval(ident_eval, &arch.modifiers) {
                    Ok(Number::Int(x)) => JsValue::from(crate::js::to_js_bigint(&x)),
                    Ok(Number::Float { value, .. }) => JsValue::from(value),
                    Err(_) => {
                        if let Expr::Identifier((ident, _)) = &arg.value.0 {
                            JsValue::from(ident)
                        } else {
                            JsValue::null()
                        }
                    }
                })
                .collect();
            js::eval_fn(body, address, &js_args).map_err(|error| {
                Error {
                    definition: def.clone(),
                    span: JS_PREFIX.len()..def.len(),
                    kind: Kind::EvaluationError(error),
                }
                .compile_error(instruction, span)
            })?
        } else {
            unimplemented!("Evaluating js code during pseudoinstruction expansion requires the `pseudoinstruction` feature flag");
        }
    } else {
        def.clone()
    };

    // Lex the instructions
    let (def, file_id) = ctx.file_cache.add(def, span);
    let parse_err = |error| {
        Error {
            definition: def.to_string(),
            span: 0..def.len(),
            kind: Kind::ParseError(error),
        }
        .compile_error(instruction, span)
    };
    let tokens = crate::parser::Instruction::lex(def, file_id).map_err(parse_err)?;
    // Process the resulting instruction sequence
    tokens
        .split(|(t, _)| [Token::Literal(';'), Token::Ctrl('\n')].contains(t))
        // Split leaves empty elements between consecutive delimiters and after the final one
        .filter(|t| !t.is_empty())
        .map(|tokens| {
            fn replace(expr: &mut Expr, f: impl Copy + Fn(&str) -> Option<Expr>) {
                match expr {
                    // TODO: collapse if let guard when raising MSRV
                    Expr::Identifier((ident, _)) => {
                        if let Some(rep) = f(ident) {
                            *expr = rep;
                        }
                    }
                    Expr::UnaryOp { operand, .. } => replace(&mut operand.0, f),
                    Expr::BinaryOp { lhs, rhs, .. } => {
                        replace(&mut lhs.0, f);
                        replace(&mut rhs.0, f);
                    }
                    _ => {}
                }
            }
            let (name, inst_args) =
                Instruction::parse_name((def.len(), file_id), tokens).map_err(parse_err)?;
            let span = chumsky::span::Span::new(file_id, name.1.start..inst_args.1.end);
            // Parse the instruction
            let (inst_def, mut inst_args) = super::parse_instruction(arch, name, inst_args)?;
            // Replace the identifiers that match the name of an argument with their value
            for arg in &mut inst_args {
                replace(&mut arg.value.0, |name| {
                    args.iter()
                        .find(|arg| instruction.syntax.fields[arg.field_idx].name == name)
                        .map(|x| x.value.0.clone())
                });
            }
            Ok(((inst_def, inst_args), span))
        })
        .collect()
}
