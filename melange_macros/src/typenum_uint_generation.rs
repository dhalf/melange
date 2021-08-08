use quote::quote;
use syn::LitInt;

pub fn u32_to_bits(mut num: u32) -> [bool; 32] {
    let mut bits = [false; 32];
    let mut i = 31;
    while num > 0 {
        bits[i] = num % 2 != 0;
        num /= 2;
        if i > 0 {
            i -= 1;
        } else {
            break;
        }
    }
    bits
}

pub fn lit_to_typenum_uint_tokens(lit: &LitInt) -> proc_macro2::TokenStream {
    let bits = u32_to_bits(lit.base10_parse().unwrap());
    bits.iter()
        .skip_while(|&&b| !b)
        .map(|&b| {
            if b {
                quote! { B1 }
            } else {
                quote! { B0 }
            }
        })
        .fold(quote! { UTerm }, |acc, bit| quote! { UInt<#acc, #bit> })
}
