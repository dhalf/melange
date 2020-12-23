use super::*;
use crate::tensor::index::Index;
use crate::tensor::layout::{DynamicLayout, StaticLayout};
use crate::tensor::shape::{intrinsic_strides_in_place, Reduction, StaticShape};
use std::convert::TryFrom;
use std::io::{Error, ErrorKind};
use std::marker::PhantomData;
use typenum::{Unsigned, U0};

impl<Y, Z, T, S> TryFrom<Vec<T>>
    for Tensor<Static, Y, Z, T, S, DefaultAllocator, Vec<T>, StaticLayout<S>>
where
    S: StaticShape,
{
    type Error = Error;
    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        if value.len() == S::NumElements::USIZE {
            Ok(Tensor {
                data: value,
                layout: StaticLayout::new(),
                _phantoms: PhantomData,
            })
        } else {
            Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected a vector of length {}, got {}.",
                    S::NumElements::USIZE,
                    value.len()
                ),
            ))
        }
    }
}

impl<Y, Z, T, S> TryFrom<Vec<T>>
    for Tensor<Dynamic, Y, Z, T, S, DefaultAllocator, Vec<T>, DynamicLayout<S::Len>>
where
    S: Reduction<U0> + Shape,
    <S as Reduction<U0>>::Output: StaticShape,
{
    type Error = Error;
    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        let len = value.len();
        if len % <<S as Reduction<U0>>::Output as StaticShape>::NumElements::USIZE == 0 {
            let mut shape = <S as Reduction<U0>>::Output::to_vec();
            if shape.len() > 0 {
                shape[0] = len / <<S as Reduction<U0>>::Output as StaticShape>::NumElements::USIZE;
            } else {
                shape.push(len / <<S as Reduction<U0>>::Output as StaticShape>::NumElements::USIZE);
            }
            Ok(Tensor {
                data: value,
                layout: DynamicLayout {
                    shape: Index::try_from(shape.clone()).unwrap(),
                    strides: Index::try_from(intrinsic_strides_in_place(shape)).unwrap(),
                    num_elements: len,
                    opt_chunk_size: len,
                },
                _phantoms: PhantomData,
            })
        } else {
            Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected a vector having a length that is a multiple of {}, got {}.",
                    <<S as Reduction<U0>>::Output as StaticShape>::NumElements::USIZE,
                    len
                ),
            ))
        }
    }
}
