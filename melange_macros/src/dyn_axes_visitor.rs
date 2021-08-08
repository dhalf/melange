use proc_macro2::{Ident, Span};
use quote::quote;
use syn::visit_mut::VisitMut;
use syn::{Macro, Token, TypePath};
use uuid::Uuid;
use crate::axes_syntax::*;
use crate::typenum_uint_generation::*;

pub struct FillAnonAxes {
    pub structs: Vec<Ident>,
    pub aliases: Vec<Ident>,
    pub paths: Vec<TypePath>,
}

impl VisitMut for FillAnonAxes {
    fn visit_macro_mut(&mut self, node: &mut Macro) {
        if node.path.is_ident("ax") {
            let mut ast: AST = match node.parse_body() {
                Ok(ast) => ast,
                Err(err) => {
                    println!("{}", err);
                    return;
                }
            };
            for ax in ast.axes.iter_mut() {
                match ax {
                    BoundAxis {
                        axis: Axis::Dynamic(_),
                        binding: b,
                    } => {
                        let ident = Ident::new(
                            &format!("__dyn_axes_{}", Uuid::new_v4().to_simple()),
                            Span::call_site(),
                        );
                        self.structs.push(ident.clone());
                        let path = syn::parse2::<TypePath>(quote! { DynAx<#ident> })
                                .expect("Unable to parse generated path.");
                        if let Some((_, name)) = b {
                            self.aliases.push(name.clone());
                            self.paths.push(path);
                            ax.axis = Axis::Inferred(Token![_](Span::call_site()));
                            ax.binding = Some((Token![@](Span::call_site()), name.clone()));
                        } else {
                            ax.axis = Axis::Raw(path);
                            ax.binding = None;
                        }
                    }
                    BoundAxis {
                        axis: Axis::Static(lit),
                        binding: Some((_, name)),
                    } => {
                        let uint = lit_to_typenum_uint_tokens(lit);
                        let path = syn::parse2::<TypePath>(quote! { StatAx<#uint> })
                            .expect("Unable to parse generated typenum representation.");
                        self.aliases.push(name.clone());
                        self.paths.push(path);
                        ax.binding = None;
                    }
                    BoundAxis { axis: Axis::Raw(_), binding: Some(_) } => panic!("Bindings are only supported with integer literals, `?` and `_`."),
                    _ => {}
                }
            }
            node.tokens = quote! { #ast }
        }
    }
}
