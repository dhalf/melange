use super::{Tensor, Static, Dynamic, Contiguous, Normal};
use crate::tensor::layout::{StaticLayout, DynamicLayout, Layout};
use crate::tensor::shape::{Shape, StaticShape, intrinsic_strides_in_place};
use crate::tensor::index::Index;
use typenum::Unsigned;
use std::marker::PhantomData;
use std::convert::TryFrom;
use crate::tensor::strided_iterator::StridedIterator;

pub trait StaticAlloc<T, S>: Sized {
    type Alloc;
    fn fill(value: T) -> Self::Alloc;

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
    /// assert_eq!(a, <DefaultAllocator as StaticAlloc<i32, Shape2D<U2, U2>>>::to_contiguous(&a));
    /// ```
    fn to_contiguous<'a, Y, Z, D, L>(tensor: &'a Tensor<Static, Y, Z, T, S, Self, D, L>) -> Self::Alloc
    where
        &'a Tensor<Static, Y, Z, T, S, Self, D, L>: StridedIterator<Item=&'a [T]>,
        S: StaticShape,
        L: Layout<S::Len>,
        T: Copy;
}

pub trait DynamicAlloc<T, S>
where
    Self: Sized,
    S: Shape,
{
    type Alloc;
    fn fill(shape: Index<S::Len>, value: T) -> Self::Alloc;
    fn to_contiguous<'a, Y, Z, D, L>(tensor: &'a Tensor<Dynamic, Y, Z, T, S, Self, D, L>) -> Self::Alloc
    where
        &'a Tensor<Dynamic, Y, Z, T, S, Self, D, L>: StridedIterator<Item=&'a [T]>,
        L: Layout<S::Len>,
        T: Copy;
}

#[derive(Debug)]
pub struct DefaultAllocator;

impl<T, S> StaticAlloc<T, S> for DefaultAllocator
where
    T: Copy,
    S: StaticShape,
{
    type Alloc = Tensor<Static, Contiguous, Normal, T, S, DefaultAllocator, Vec<T>, StaticLayout<S>>;
    fn fill(value: T) -> Self::Alloc {
        Tensor {
            data: vec![value; S::NumElements::USIZE],
            layout: StaticLayout::new(),
            _phantoms: PhantomData,
        }
    }
    fn to_contiguous<'a, Y, Z, D, L>(tensor: &'a Tensor<Static, Y, Z, T, S, Self, D, L>) -> Self::Alloc
    where
        &'a Tensor<Static, Y, Z, T, S, Self, D, L>: StridedIterator<Item=&'a [T]>,
        S: StaticShape,
        L: Layout<S::Len>,
        T: Copy,
    {
        let mut buffer: Vec<T> = Vec::with_capacity(S::NumElements::USIZE);
        for chunk in tensor.strided_iter(tensor.opt_chunk_size()) {
            buffer.extend(chunk.iter());
        }

        Tensor {
            data: buffer,
            layout: StaticLayout::new(),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S> DynamicAlloc<T, S> for DefaultAllocator
where
    T: Copy,
    S: Shape,
{
    type Alloc = Tensor<Dynamic, Contiguous, Normal, T, S, DefaultAllocator, Vec<T>, DynamicLayout<S::Len>>;
    fn fill(shape: Index<S::Len>, value: T) -> Self::Alloc {
        let strides = intrinsic_strides_in_place(shape.clone().into());
        let num_elements = shape.iter().product();
        
        Tensor {
            data: vec![value; num_elements],
            layout: DynamicLayout {
                shape,
                strides: Index::try_from(strides).unwrap(),
                num_elements,
                opt_chunk_size: num_elements,
            },
            _phantoms: PhantomData,
        }
    }
    fn to_contiguous<'a, Y, Z, D, L>(tensor: &'a Tensor<Dynamic, Y, Z, T, S, Self, D, L>) -> Self::Alloc
    where
        &'a Tensor<Dynamic, Y, Z, T, S, Self, D, L>: StridedIterator<Item=&'a [T]>,
        L: Layout<S::Len>,
        T: Copy,
    {
        let mut buffer: Vec<T> = Vec::with_capacity(tensor.num_elements());
        for chunk in tensor.strided_iter(tensor.opt_chunk_size()) {
            buffer.extend(chunk.iter());
        }

        Tensor {
            data: buffer,
            layout: DynamicLayout {
                shape: tensor.shape(),
                strides: Index::try_from(intrinsic_strides_in_place(tensor.shape().into())).unwrap(),
                num_elements: tensor.num_elements(),
                opt_chunk_size: tensor.num_elements(),
            },
            _phantoms: PhantomData,
        }
    }
}

pub trait AllocLike {
    type Alloc;
    type Scalar;
    fn fill_like(&self, value: Self::Scalar) -> Self::Alloc;
    fn to_contiguous(&self) -> Self::Alloc;
}

impl<Y, Z, T, S, A, D, L> AllocLike for Tensor<Static, Y, Z, T, S, A, D, L>
where
    T: Copy,
    S: StaticShape,
    A: StaticAlloc<T, S>,
    L: Layout<S::Len>,
    for<'a> &'a Self: StridedIterator<Item=&'a [T]>,
{
    type Alloc = <A as StaticAlloc<T, S>>::Alloc;
    type Scalar = T;
    fn fill_like(&self, value: Self::Scalar) -> Self::Alloc {
        <A as StaticAlloc<T, S>>::fill(value)
    }
    fn to_contiguous(&self) -> Self::Alloc {
        <A as StaticAlloc<T, S>>::to_contiguous(self)
    }
}

impl<Y, Z, T, S, A, D, L> AllocLike for Tensor<Dynamic, Y, Z, T, S, A, D, L>
where
    T: Copy,
    S: Shape,
    L: Layout<S::Len>,
    A: DynamicAlloc<T, S>,
    for<'a> &'a Self: StridedIterator<Item=&'a [T]>,
{
    type Alloc = <A as DynamicAlloc<T, S>>::Alloc;
    type Scalar = T;
    fn fill_like(&self, value: Self::Scalar) -> Self::Alloc {
        <A as DynamicAlloc<T, S>>::fill(self.shape(), value)
    }
    fn to_contiguous(&self) -> Self::Alloc {
        <A as DynamicAlloc<T, S>>::to_contiguous(self)
    }
}
