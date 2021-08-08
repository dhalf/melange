use proc_macro2::Ident;
use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    LitInt, Result, Token, TypePath,
};

#[derive(Clone)]
pub enum Axis {
    Static(LitInt),
    Dynamic(Token![?]),
    Inferred(Token![_]),
    Raw(TypePath),
}

impl Parse for Axis {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(LitInt) {
            input.parse().map(Axis::Static)
        } else if lookahead.peek(Token![?]) {
            input.parse().map(Axis::Dynamic)
        } else if lookahead.peek(Token![_]) {
            input.parse().map(Axis::Inferred)
        } else {
            input.parse().map(Axis::Raw)
        }
    }
}

impl ToTokens for Axis {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Axis::Static(lit) => lit.to_tokens(tokens),
            Axis::Dynamic(q) => q.to_tokens(tokens),
            Axis::Inferred(u) => u.to_tokens(tokens),
            Axis::Raw(path) => path.to_tokens(tokens),
        }
    }
}

#[derive(Clone)]
pub struct BoundAxis {
    pub axis: Axis,
    pub binding: Option<(Token![@], Ident)>,
}

impl Parse for BoundAxis {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(BoundAxis {
            axis: input.parse()?,
            binding: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![@]) {
                    Some((input.parse()?, input.parse()?))
                } else {
                    None
                }
            },
        })
    }
}

impl ToTokens for BoundAxis {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.axis.to_tokens(tokens);
        if let Some((at, name)) = &self.binding {
            at.to_tokens(tokens);
            name.to_tokens(tokens);
        }
    }
}

pub struct AST {
    pub axes: Punctuated<BoundAxis, Token![,]>,
}

impl Parse for AST {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(AST {
            axes: Punctuated::parse_terminated(input)?,
        })
    }
}

impl ToTokens for AST {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.axes.to_tokens(tokens);
    }
}
