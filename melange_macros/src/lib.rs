use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, DeriveInput, Item, LitInt, Stmt};

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
            let len = lit_to_typenum_uint_tokens(&LitInt::new(
                &format!("{}", elems.len()),
                Span::call_site(),
            ));
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

    if let Item::Fn(_) | Item::Mod(_) = item {
    } else {
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
    let namespace =
        syn::parse2(namespace).expect("Unable to parse genrated `__dyn_axes_namespace` module.");
    let use_stmt = syn::parse2(quote! { use __dyn_axes_namespace::*; }).unwrap();

    match &mut item {
        Item::Fn(f) => {
            f.block.stmts.insert(0, Stmt::Item(namespace));
            f.block.stmts.insert(1, Stmt::Item(use_stmt));
        }
        Item::Mod(m) => {
            if let Some((_, contained)) = &mut m.content {
                contained.insert(0, namespace);
                contained.insert(0, use_stmt);
            }
        }
        _ => {}
    }
    let output = quote! { #item };
    //println!("{}", output);
    output.into()
}

#[proc_macro_attribute]
pub fn rvar(_: TokenStream, item: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(item as DeriveInput);
    impl_rvar_macro(&ast)
}

fn impl_rvar_macro(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let rgrad_name = format_ident!("{}RGrad", name);
    let alloc_data_name = format_ident!("{}AllocData", name);
    let destruct_name = format_ident!("{}Destructured", name);
    let vis = &ast.vis;
    let generics: &Vec<_> = &ast.generics.params.iter().collect();
    let generic_types: &Vec<_> = &ast
        .generics
        .params
        .iter()
        .filter(|&p| {
            if let syn::GenericParam::Type(_) = p {
                true
            } else {
                false
            }
        })
        .collect();
    let gen = match &ast.data {
        syn::Data::Struct(s) => {
            let empty = Punctuated::new();
            let fields = match &s.fields {
                syn::Fields::Named(f) => &f.named,
                syn::Fields::Unnamed(f) => &f.unnamed,
                syn::Fields::Unit => &empty,
            };
            let destruct_fields: Vec<_> = fields
                .iter()
                .map(|f| {
                    let mut f = f.clone();
                    let ty = &f.ty;
                    f.ty = syn::parse2(quote! { RVar<#ty> }).unwrap();
                    f
                })
                .collect();
            let rgrad_fields: Vec<_> = fields
                .iter()
                .map(|f| {
                    let mut f = f.clone();
                    let ty = &f.ty;
                    f.ty = syn::parse2(quote! { Grad<#ty> }).unwrap();
                    f
                })
                .collect();
            let alloc_data_fields: Vec<_> = fields
                .iter()
                .map(|f| {
                    let mut f = f.clone();
                    let ty = &f.ty;
                    f.ty = syn::parse2(quote! { <#ty as Differentiable>::AllocData }).unwrap();
                    f
                })
                .collect();
            let rgrad_zero_impl_fields: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    match &f.ident {
                        Some(name) => quote! { #name: Grad::NonAllocated(data.#name) },
                        None => quote! { Grad::NonAllocated(data.#i) },
                    }
                })
                .collect();
            let rgrad_to_grad_impl_fields: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    match &f.ident {
                        Some(name) => quote! { #name: rgrad.#name.take() },
                        None => quote! { rgrad.#i.take() },
                    }
                })
                .collect();
            let grad_to_rgrad_impl_fields: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    match &f.ident {
                        Some(name) => quote! { #name: Grad::new(self.#name) },
                        None => quote! { Grad::new(self.#i) },
                    }
                })
                .collect();
            let zero_impl_fields: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let ty = &f.ty;
                    match &f.ident {
                        Some(name) => quote! { #name: <#ty as Differentiable>::zero(data.#name) },
                        None => quote! { <#ty as Differentiable>::zero(data.#i) },
                    }
                })
                .collect();
            let alloc_data_impl_fields: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let ty = &f.ty;
                    match &f.ident {
                        Some(name) => quote! { #name: <#ty as Differentiable>::alloc_data(&self.#name) },
                        None => quote! { <#ty as Differentiable>::alloc_data(&self.#i) },
                    }
                })
                .collect();
            let destruct_impl_fields: Vec<_> = fields.iter().enumerate().map(|(i, f)| {
                match &f.ident {
                    Some(name) => quote! { #name: {
                            let data = data.clone();
                            RVar::new_destructured_field_var(self.#name, RVar::clone(&parent), move |x| {
                                let mut grad = Self::zero_rgrad(data.clone());
                                grad.#name = Grad::new(x);
                                grad
                            })
                        }
                    },
                    None => quote! { {
                            let data = data.clone();
                            RVar::new_destructured_field_var(self.#i, RVar::clone(&parent), move |x| {
                                let mut grad = Self::zero_rgrad(data.clone());
                                grad.#i = Grad::new(x);
                                grad
                            })
                        }
                    },
                }
            }).collect();
            let add_impl_fields: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let inner = &quote! {
                        (Grad::Allocated(x), Grad::Allocated(y)) => **x += *y,
                        (g @ Grad::NonAllocated(_), h @ Grad::Allocated(_)) => *g = h,
                        _ => (),
                    };
                    match &f.ident {
                        Some(name) => quote! { match (&mut self.#name, other.#name) {
                            #inner
                        }},
                        None => quote! { match (&mut self.#i, other.#i) {
                            #inner
                        }},
                    }
                })
                .collect();
            quote! {
                #[derive(Clone, Debug)]
                #ast

                #[derive(Clone, Debug)]
                #vis struct #destruct_name<#(#generics),*>
                where
                    #(#generic_types: Differentiable),*
                {
                    #(#destruct_fields),*
                }

                #[derive(Clone, Debug)]
                #vis struct #rgrad_name<#(#generics),*>
                where
                    #(#generic_types: Differentiable),*
                {
                    #(#rgrad_fields),*
                }

                #[derive(Clone, Debug)]
                #vis struct #alloc_data_name<#(#generics),*>
                where
                    #(#generic_types: Differentiable),*
                {
                    #(#alloc_data_fields),*
                }

                impl<#(#generics),*> Differentiable for #name
                where
                    #(#generic_types: Differentiable),*
                {
                    type RGrad = #rgrad_name;
                    type AllocData = #alloc_data_name;
                    fn rgrad_to_grad(rgrad: Self::RGrad) -> Self {
                        #name {
                            #(#rgrad_to_grad_impl_fields),*
                        }
                    }
                    fn grad_to_rgrad(self) -> Self::RGrad {
                        #rgrad_name {
                            #(#grad_to_rgrad_impl_fields),*
                        }
                    }
                    fn alloc_data(&self) -> Self::AllocData {
                        #alloc_data_name {
                            #(#alloc_data_impl_fields),*
                        }
                    }
                    fn zero(data: Self::AllocData) -> Self {
                        #name {
                            #(#zero_impl_fields),*
                        }
                    }
                    fn zero_rgrad(data: Self::AllocData) -> Self::RGrad {
                        #rgrad_name {
                            #(#rgrad_zero_impl_fields),*
                        }
                    }
                }

                impl<#(#generics),*> Destructure for #name
                where
                    #(#generic_types: Differentiable),*
                {
                    type Destructured = #destruct_name;
                    fn internal_destructure(self, parent: RVar<Self>) -> Self::Destructured {
                        let data = self.alloc_data();
                        #destruct_name {
                            #(#destruct_impl_fields),*
                        }
                    }
                }

                impl<#(#generics),*> std::ops::AddAssign<#rgrad_name<#(#generics),*>> for #rgrad_name<#(#generics),*>
                where
                    #(#generic_types: Differentiable),*
                {
                    fn add_assign(&mut self, other: #rgrad_name<#(#generics),*>) {
                        #(#add_impl_fields);*
                    }
                }
            }
        }
        _ => {
            quote! {}
        }
    };
    //println!("{}", gen);
    gen.into()
}
