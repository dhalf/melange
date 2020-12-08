use super::*;
use crate::tensor::layout::{StaticLayout, DynamicLayout};
use crate::tensor::shape::{StaticShape, intrinsic_strides_in_place, Reduction};
use std::convert::TryFrom;
use std::io::{Error, ErrorKind};
use std::marker::PhantomData;
use typenum::{Unsigned, U0};
use crate::tensor::index::Index;

impl<Y, Z, T, S> TryFrom<Vec<T>> for Tensor<Static, Y, Z, T, S, Vec<T>, StaticLayout<S>>
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

impl<Y, Z, T, S> TryFrom<Vec<T>> for Tensor<Dynamic, Y, Z, T, S, Vec<T>, DynamicLayout<S::Len>>
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

/// Contiguous copy of a tensor.
/// 
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// 
/// let a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![1, 2]).unwrap();
/// let a = Broadcast::<Shape2D<U2, U2>>::broadcast(&a); // a is now strided.
/// let b = a.to_contiguous(); // this is a contiguous copy of a.
/// assert_eq!(a, b);
/// ```
pub trait ToContiguous {
    /// Output type.
    type Output;

    /// Outputs a contiguous copy of the tensor.
    /// 
    /// See trait-level documentation.
    fn to_contiguous(self) -> Self::Output;
}

impl<'a, Y, Z, T, S, D, L> ToContiguous for &'a Tensor<Static, Y, Z, T, S, D, L>
where
    Self: StridedIterator<Item=&'a [T]>,
    S: StaticShape,
    L: Layout<S::Len>,
    T: Copy,
{
    type Output = Tensor<Static, Contiguous, Normal, T, S, Vec<T>, StaticLayout<S>>;
    fn to_contiguous(self) -> Self::Output {
        let mut buffer: Vec<T> = Vec::with_capacity(S::NumElements::USIZE);
        for chunk in self.strided_iter(self.opt_chunk_size()) {
            buffer.extend(chunk.iter());
        }

        Tensor {
            data: buffer,
            layout: StaticLayout::new(),
            _phantoms: PhantomData,
        }
    }
}

impl<'a, Y, Z, T, S, D, L> ToContiguous for &'a Tensor<Dynamic, Y, Z, T, S, D, L>
where
    Self: StridedIterator<Item=&'a [T]>,
    S: StaticShape,
    L: Layout<S::Len>,
    T: Copy,
{
    type Output = Tensor<Dynamic, Contiguous, Normal, T, S, Vec<T>, DynamicLayout<S::Len>>;
    fn to_contiguous(self) -> Self::Output {
        let mut buffer: Vec<T> = Vec::with_capacity(self.num_elements());
        for chunk in self.strided_iter(self.opt_chunk_size()) {
            buffer.extend(chunk.iter());
        }

        Tensor {
            data: buffer,
            layout: DynamicLayout {
                shape: self.shape(),
                strides: Index::try_from(intrinsic_strides_in_place(self.shape().into())).unwrap(),
                num_elements: self.num_elements(),
                opt_chunk_size: self.num_elements(),
            },
            _phantoms: PhantomData,
        }
    }
}
