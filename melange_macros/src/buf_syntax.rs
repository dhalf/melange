use syn::{Expr, Type, Token, Result};
use syn::punctuated::Punctuated;
use syn::parse::{Parse, ParseStream};
use quote::ToTokens;

pub struct BufAST {
    pub elems: Punctuated<Expr, Token![,]>,
    pub colon: Token![;],
    pub elem_ty: Type,
    pub size_ty: Option<(Token![;], Type)>,
}

impl Parse for BufAST {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(BufAST {
            elems: Punctuated::parse_separated_nonempty(input)?,
            colon: input.parse()?,
            elem_ty: input.parse()?,
            size_ty: match input.parse::<Token![;]>() {
                Ok(sm2) => Some((sm2, input.parse()?)),
                Err(_) => None,
            },
        })
    }
}

impl ToTokens for BufAST {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.elems.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.elem_ty.to_tokens(tokens);
        if let Some((semi, size_ty)) = &self.size_ty {
            semi.to_tokens(tokens);
            size_ty.to_tokens(tokens);
        }
    }
}
