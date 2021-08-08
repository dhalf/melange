use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, Item, Stmt, LitInt};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

mod axes_syntax;
use axes_syntax::*;

mod dyn_axes_visitor;
use dyn_axes_visitor::*;

mod typenum_uint_generation;
use typenum_uint_generation::*;

mod buf_syntax;
use buf_syntax::*;

#[proc_macro]
pub fn buf(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BufAST);

    let BufAST { elems, elem_ty, .. } = input;
    let output = match input.size_ty {
        Some((_, size_ty)) => {
            let value = elems[0].clone();
            quote! {
                {
                    let buf: <#size_ty as StackBuffer<[#elem_ty; 1]>>::Buffer = <#size_ty as StackBuffer<[#elem_ty; 1]>>::Buffer::fill(#value);
                    buf
                }
            }
        }
        None => {
            let len = lit_to_typenum_uint_tokens(&LitInt::new(&format!("{}", elems.len()), Span::call_site()));
            let idx = 0..elems.len();
            let elems = elems.iter();
            quote! {
                {
                    let mut buf: <#len as StackBuffer<[#elem_ty; 1]>>::Buffer = <#len as StackBuffer<[#elem_ty; 1]>>::Buffer::default();
                    let sl = buf.as_mut();
                    #(sl[#idx] = #elems;)*
                    buf
                }
            }
        }
    };
    //println!("{}", output);
    output.into()
}

#[proc_macro]
pub fn ax(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as AST);

    let mut output = quote! { Ax0 };
    for axis in input.axes.iter().rev() {
        let axis = match axis {
            BoundAxis { axis: Axis::Static(lit), binding: None } => {
                let uint = lit_to_typenum_uint_tokens(lit);
                quote! { StatAx<#uint> }
            },
            BoundAxis { axis: Axis::Static(_), binding: Some(_) } => panic!(
                "Unhandled binding definition, consider using the `dyn_axes` attribute on the enclosing item."
            ),
            BoundAxis { axis: Axis::Dynamic(_), .. } => panic!(
                "Unhandled dynamic axis definition, consider using the `dyn_axes` attribute on the enclosing item."
            ),
            BoundAxis { axis: Axis::Inferred(_), binding: Some((_, name)) } => quote! { #name },
            BoundAxis { axis: Axis::Inferred(_), binding: None } => quote! { _ },
            BoundAxis { axis: Axis::Raw(path), .. } => quote! { #path },
        };
        output = quote! { Ax<#axis, #output> };
    }
    //println!("{}", output);
    output.into()
}

#[proc_macro_attribute]
pub fn dyn_axes(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(item as Item);

    if let Item::Fn(_) | Item::Mod(_) = item {} else {
        panic!("`dyn_axes` attribute only supports function and module items.")
    }

    let mut visitor = FillAnonAxes {
        structs: Vec::new(),
        aliases: Vec::new(),
        paths: Vec::new(),
    };
    visitor.visit_item_mut(&mut item);

    let FillAnonAxes {
        structs,
        aliases,
        paths,
    } = visitor;
    let namespace = quote! {
        pub mod __dyn_axes_namespace {
            use super::*;
            #(pub struct #structs;)*
            #(pub type #aliases = #paths;)*
        }
    };
    let namespace = syn::parse2(namespace).expect("Unable to parse genrated `__dyn_axes_namespace` module.");
    let use_stmt = syn::parse2(quote! { use __dyn_axes_namespace::*; }).unwrap();

    match &mut item {
        Item::Fn(f) => {
            f.block.stmts.insert(0, Stmt::Item(namespace));
            f.block.stmts.insert(1, Stmt::Item(use_stmt));
        },
        Item::Mod(m) => {
            if let Some((_, contained)) = &mut m.content {
                contained.insert(0, namespace);
                contained.insert(0, use_stmt);
            }
        }
        _ => {},
    }
    let output = quote! { #item };
    //println!("{}", output);
    output.into()
}
