//! Collection of tools to interact with multidimensional arrays that are at the
//! core of ML pipelines.
//!
//! It defines various ways to store data and optimized mathematical operations.
//!
//! Contrary to other ndarray modules such as numpy in Python or ndarray in Rust,
//! this module allows full size checking at compile time thanks to the use of
//! the `const-generics` feature.
//!
//! # Examples
//! ```
//! use melange::prelude::*;
//! use typenum::{U1, U2};
//!
//! let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![
//!     1, 0,
//!     0, 1
//! ]).unwrap();
//! let b: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![
//!     1, 1
//! ]).unwrap();
//! let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![
//!     2, 1,
//!     1, 2
//! ]).unwrap();
//! assert_eq!(a.add(&b.broadcast()), c);
//! ```
//!
//! [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html

use crate::axes::block::Block;
use crate::axes::broadcast::Broadcast;
use crate::axes::static_axes::StaticAxes;
use crate::axes::stride::Stride;
use crate::axes::transpose::Transpose;
use crate::axes::*;
use crate::hkt::{
    KindTypeTypeType, StackBufferConstructor, VecConstructor, ViewConstructorPA2,
    ViewMutConstructorPA2,
};
use crate::iter::strided_iter::StridedIter;
use crate::iter::strided_iter::StridedIterMut;
use crate::iter::StreamingIterator;
use crate::stack_buffer::{Buffer, Glue, StackBuffer};
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::marker::PhantomData;
use typenum::Unsigned;

pub mod alloc;
use alloc::*;

pub mod linalg;
use linalg::*;

pub mod elementwise_ops;
pub mod reductions;

/// Core type of this module that represents a tensor.
#[derive(Debug)]
pub struct Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem>,
    S: Axes,
{
    buffer: B::Applied,
    size: <S::Len as StackBuffer<[usize; 1]>>::Buffer,
    offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer,
    stride: <S::Len as StackBuffer<[usize; 1]>>::Buffer,
    opt_chunk_size: usize,
    _phantoms: PhantomData<*const C>,
}

macro_rules! view_fns {
    ($broadcast:ident $broadcast_dyn:ident $stride:ident $stride_dyn:ident
        $block:ident $block_dyn:ident $transpose:ident $buffer:ident { $($mut:tt)? } { $borrow:ident }
        { $($bound_ty:ty: $bound_path:path)? }) => {
        /// Returns a broadcasted view (a tensor whose buffer is a reference to the original
        /// tensor's buffer) into the tensor.
        ///
        /// The output size is a constant of type [`Axes`] that can also be infered
        /// from the context.
        ///
        /// # Examples
        /// ```
        /// # #![allow(incomplete_features)]
        /// // #![feature(const_generics)]
        /// // #![feature(const_evaluatable_checked)]
        /// use melange::prelude::*;
        /// use std::convert::TryFrom;
        ///
        /// let a: StaticVecTensor1D<i32, 2> = Tensor::try_from(vec![1, 2]).unwrap();
        /// // let b: StaticVecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
        /// // println!("{:?}", a.broadcast::<{ ax!(2, 2) }>());
        /// // assert_eq!(a.broadcast(), b);
        /// ```
        #[inline]
        pub fn $broadcast<'a, Z>(&'a $($mut)? self) -> Tensor<$buffer<'a, B>, T, Z, Strided>
        where
            S: Broadcast<Z>,
            Z: StaticAxes,
            $($bound_ty: $bound_path,)?
        {
            self.$broadcast_dyn(Z::runtime())
        }
        /// Dynamic version of [`broadcast`](Tensor::broadcast) that takes in an
        /// additional size parameter.
        pub fn $broadcast_dyn<'a, Z>(
            &'a $($mut)? self,
            size: <Z::Len as StackBuffer<[usize; 1]>>::Buffer,
        ) -> Tensor<$buffer<'a, B>, T, Z, Strided>
        where
            S: Broadcast<Z>,
            Z: Axes,
            $($bound_ty: $bound_path,)?
        {
            assert!(
                self.size
                    .as_ref()
                    .iter()
                    .zip(size.as_ref().iter())
                    .all(|(&s, &z)| s == 1 || s == z),
                "Cannot broadcast size {:?} into {:?}.",
                self.size,
                size
            );

            let stride =
                collect::<_, _, Z::Len>(
                    self.size
                        .as_ref()
                        .iter()
                        .zip(self.stride.as_ref().iter())
                        .map(|(&s, &z)| if s == 1 { 0 } else { z }),
                );

            Tensor {
                buffer: self.buffer.$borrow(),
                offset: pad::<_, S::Len, Z::Len>(&self.offset, 0),
                opt_chunk_size: opt_chunk_size::<Z::Len>(&size, &stride),
                size,
                stride,
                _phantoms: PhantomData,
            }
        }
        /// Returns a strided view (a tensor whose buffer is a reference to the original
        /// tensor's buffer) into the tensor.
        ///
        /// The stride is a constant parameter of type [`Axes`].
        #[inline]
        pub fn $stride<'a, Z>(&'a $($mut)? self) -> Tensor<$buffer<'a, B>, T, <S as Stride<Z>>::Output, Strided>
        where
            S: Stride<Z>,
            <S as Stride<Z>>::Output: Axes<Len = S::Len>,
            Z: StaticAxes<Len = S::Len>,
            $($bound_ty: $bound_path,)?
        {
            self.$stride_dyn::<Z>(Z::runtime())
        }
        /// Dynamic version of [`stride`](Tensor::stride) that takes in an
        /// additional stride parameter.
        pub fn $stride_dyn<'a, Z>(
            &'a $($mut)? self,
            stride: <Z::Len as StackBuffer<[usize; 1]>>::Buffer,
        ) -> Tensor<$buffer<'a, B>, T, <S as Stride<Z>>::Output, Strided>
        where
            S: Stride<Z>,
            <S as Stride<Z>>::Output: Axes<Len = S::Len>,
            Z: Axes<Len = S::Len>,
            $($bound_ty: $bound_path,)?
        {
            let size = collect::<_, _, <<S as Stride<Z>>::Output as Axes>::Len>(
                self.size
                    .as_ref()
                    .iter()
                    .zip(stride.as_ref().iter())
                    .map(|(&s, &z)| s / z + if s % z > 0 { 1 } else { 0 }),
            );
            let total_stride = total_stride::<<<S as Stride<Z>>::Output as Axes>::Len>(&self.size, &stride);

            Tensor {
                buffer: self.buffer.$borrow(),
                offset: self.offset,
                opt_chunk_size: opt_chunk_size::<<<S as Stride<Z>>::Output as Axes>::Len>(&size, &total_stride),
                size,
                stride: total_stride,
                _phantoms: PhantomData,
            }
        }
        /// Returns a view (a tensor whose buffer is a reference to the original
        /// tensor's buffer) into a sub-block of the tensor.
        ///
        /// The output size and offset are constants of type [`Axes`].
        ///
        /// # Safety
        /// The view actually borrows the entire tensor making it impossible to
        /// create two mutable non overlapping blocks at the same time for now.
        #[inline]
        pub fn $block<'a, Z, O>(&'a $($mut)? self) -> Tensor<$buffer<'a, B>, T, Z, Strided>
        where
            S: Block<O, Z>,
            Z: StaticAxes<Len = S::Len>,
            O: StaticAxes<Len = S::Len>,
            $($bound_ty: $bound_path,)?
        {
            self.$block_dyn::<Z, O>(Z::runtime(), O::runtime())
        }
        /// Dynamic version of [`block`](Tensor::block) that takes in an
        /// additional size and offset parameters.
        pub fn $block_dyn<'a, Z, O>(
            &'a $($mut)? self,
            size: <Z::Len as StackBuffer<[usize; 1]>>::Buffer,
            offset: <O::Len as StackBuffer<[usize; 1]>>::Buffer,
        ) -> Tensor<$buffer<'a, B>, T, Z, Strided>
        where
            S: Block<O, Z>,
            Z: Axes<Len = S::Len>,
            O: Axes<Len = S::Len>,
            $($bound_ty: $bound_path,)?
        {
            let total_offset = collect::<_, _, Z::Len>(
                self.offset
                    .as_ref()
                    .iter()
                    .zip(offset.as_ref().iter())
                    .map(|(&a, &b)| a + b)
            );

            Tensor {
                buffer: self.buffer.$borrow(),
                offset: total_offset,
                opt_chunk_size: opt_chunk_size::<Z::Len>(&size, &self.stride),
                size,
                stride: self.stride,
                _phantoms: PhantomData,
            }
        }
        /// Returns a transposed view (a tensor whose buffer is a reference to the original
        /// tensor's buffer) into the tensor.
        pub fn $transpose<'a>(&'a $($mut)? self) -> Tensor<$buffer<'a, B>, T, <S as Transpose>::Output, Transposed>
        where
            S: Transpose,
            <S as Transpose>::Output: Axes<Len = S::Len>,
            $($bound_ty: $bound_path,)?
        {
            Tensor {
                buffer: self.buffer.$borrow(),
                size: collect::<_, _, <<S as Transpose>::Output as Axes>::Len>(self.size.as_ref().iter().rev().cloned()),
                offset: collect::<_, _, <<S as Transpose>::Output as Axes>::Len>(self.offset.as_ref().iter().rev().cloned()),
                stride: collect::<_, _, <<S as Transpose>::Output as Axes>::Len>(self.stride.as_ref().iter().rev().cloned()),
                opt_chunk_size: 1,
                _phantoms: PhantomData,
            }
        }
    };
}

impl<B, T, S> Tensor<B, T, S, Contiguous>
where
    B: KindTypeTypeType<T, S::Elem>,
    S: Axes,
{
    /// Returns a (staticaly-sized) tensor filled with the given value.
    pub fn fill(value: T) -> Self
    where
        B: Alloc<T, S::Elem>,
        S: StaticAxes,
    {
        let size = S::runtime();
        Tensor {
            buffer: B::fill(value, S::Elem::USIZE),
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size),
            size,
            opt_chunk_size: S::Elem::USIZE,
            _phantoms: PhantomData,
        }
    }
    /// Returns a tensor filled with the given value of the given size.
    pub fn fill_dyn(value: T, size: <S::Len as StackBuffer<[usize; 1]>>::Buffer) -> Self
    where
        B: Alloc<T, S::Elem> + DynamicBuffer,
    {
        let len = size.as_ref().iter().product();
        Tensor {
            buffer: B::fill(value, len),
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size),
            size,
            opt_chunk_size: len,
            _phantoms: PhantomData,
        }
    }
}

impl<B, T, S, C> Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem>,
    S: Axes,
{
    #[inline]
    fn chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> StridedIter<'a, T, <S::Len as StackBuffer<[usize; 1]>>::Buffer>
    where
        B::Applied: AsRef<[T]>,
    {
        StridedIter::new(
            self.buffer.as_ref(),
            self.size,
            self.stride,
            self.offset,
            chunk_size,
        )
    }
    #[inline]
    fn chunks_mut<'a>(
        &'a mut self,
        chunk_size: usize,
    ) -> StridedIterMut<'a, T, <S::Len as StackBuffer<[usize; 1]>>::Buffer>
    where
        B::Applied: AsMut<[T]>,
    {
        StridedIterMut::new(
            self.buffer.as_mut(),
            self.size,
            self.stride,
            self.offset,
            chunk_size,
        )
    }
    /// Applies given function `f` in-place to all the elements of the tensor.
    pub fn for_each<F: FnMut(&mut T)>(&mut self, mut f: F)
    where
        T: 'static,
        B::Applied: AsMut<[T]>,
    {
        let mut it = self.chunks_mut(self.opt_chunk_size);
        while let Some(chunk) = it.next() {
            for x in chunk.iter_mut() {
                f(x);
            }
        }
    }
    /// Applies given function `f` in-place to all the elements of the tensor,
    /// the second argument being the corresponding element of the second tensor.
    pub fn zip_with_mut<U, Bu, Cu, F>(&mut self, other: &Tensor<Bu, U, S, Cu>, mut f: F)
    where
        F: FnMut(&mut T, &U),
        B::Applied: AsMut<[T]>,
        Bu: KindTypeTypeType<U, S::Elem>,
        Bu::Applied: AsRef<[U]>,
        T: 'static,
        U: 'static,
    {
        assert_eq!(
            self.size.as_ref(),
            other.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other.size
        );

        let chunk_size = self.opt_chunk_size.min(other.opt_chunk_size);
        let it_a = self.chunks_mut(chunk_size);
        let it_b = other.chunks(chunk_size);
        let mut it = it_a.streaming_zip(it_b);

        while let Some((chunk_a, chunk_b)) = it.next() {
            for (x, y) in chunk_a.iter_mut().zip(chunk_b.iter()) {
                f(x, y);
            }
        }
    }
    /// Applies given function `f` in-place to all the elements of the tensor,
    /// the second and third arguments being the corresponding elements of the
    /// second and third tensors.
    pub fn zip2_with_mut<U, V, Bu, Bv, Cu, Cv, F>(
        &mut self,
        other0: &Tensor<Bu, U, S, Cu>,
        other1: &Tensor<Bv, V, S, Cv>,
        mut f: F,
    ) where
        F: FnMut(&mut T, &U, &V),
        B::Applied: AsMut<[T]>,
        Bu: KindTypeTypeType<U, S::Elem>,
        Bu::Applied: AsRef<[U]>,
        Bv: KindTypeTypeType<V, S::Elem>,
        Bv::Applied: AsRef<[V]>,
        T: 'static,
        U: 'static,
        V: 'static,
    {
        assert_eq!(
            self.size.as_ref(),
            other0.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other0.size
        );
        assert_eq!(
            self.size.as_ref(),
            other1.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other1.size
        );

        let chunk_size = self
            .opt_chunk_size
            .min(other0.opt_chunk_size)
            .min(other1.opt_chunk_size);
        let it_a = self.chunks_mut(chunk_size);
        let it_b = other0.chunks(chunk_size);
        let it_c = other1.chunks(chunk_size);
        let mut it = it_a.streaming_zip(it_b).streaming_zip(it_c);

        while let Some(((chunk_a, chunk_b), chunk_c)) = it.next() {
            for ((x, y), z) in chunk_a.iter_mut().zip(chunk_b.iter()).zip(chunk_c.iter()) {
                f(x, y, z);
            }
        }
    }
    /// Applies given function `f` in-place to all the elements of the tensor,
    /// the second and third arguments being the corresponding elements of the
    /// second and third tensors.
    pub fn zip3_with_mut<U, V, W, Bu, Bv, Bw, Cu, Cv, Cw, F>(
        &mut self,
        other0: &Tensor<Bu, U, S, Cu>,
        other1: &Tensor<Bv, V, S, Cv>,
        other2: &Tensor<Bw, W, S, Cw>,
        mut f: F,
    ) where
        F: FnMut(&mut T, &U, &V, &W),
        B::Applied: AsMut<[T]>,
        Bu: KindTypeTypeType<U, S::Elem>,
        Bu::Applied: AsRef<[U]>,
        Bv: KindTypeTypeType<V, S::Elem>,
        Bv::Applied: AsRef<[V]>,
        Bw: KindTypeTypeType<W, S::Elem>,
        Bw::Applied: AsRef<[W]>,
        T: 'static,
        U: 'static,
        V: 'static,
        W: 'static,
    {
        assert_eq!(
            self.size.as_ref(),
            other0.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other0.size
        );
        assert_eq!(
            self.size.as_ref(),
            other1.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other1.size
        );
        assert_eq!(
            self.size.as_ref(),
            other2.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other2.size
        );

        let chunk_size = self
            .opt_chunk_size
            .min(other0.opt_chunk_size)
            .min(other1.opt_chunk_size)
            .min(other2.opt_chunk_size);
        let it_a = self.chunks_mut(chunk_size);
        let it_b = other0.chunks(chunk_size);
        let it_c = other1.chunks(chunk_size);
        let it_d = other2.chunks(chunk_size);
        let mut it = it_a.streaming_zip(it_b).streaming_zip(it_c).streaming_zip(it_d);

        while let Some((((chunk_a, chunk_b), chunk_c), chunk_d)) = it.next() {
            for (((x, y), z), w) in chunk_a.iter_mut().zip(chunk_b.iter()).zip(chunk_c.iter()).zip(chunk_d.iter()) {
                f(x, y, z, w);
            }
        }
    }
    view_fns! {
        broadcast broadcast_dyn stride stride_dyn block block_dyn
        transpose ViewConstructorPA2 {/* no mut */} { as_ref } { B::Applied: AsRef<[T]> }
    }
    view_fns! {
        broadcast_mut broadcast_dyn_mut stride_mut stride_dyn_mut block_mut
        block_dyn_mut transpose_mut ViewMutConstructorPA2 { mut } { as_mut }
        { B::Applied: AsMut<[T]> }
    }
}

impl<T, B, B2, C, C2, S> PartialEq<Tensor<B2, T, S, C2>> for Tensor<B, T, S, C>
where
    T: PartialEq + 'static,
    S: Axes,
    B: KindTypeTypeType<T, S::Elem>,
    B::Applied: AsRef<[T]>,
    B2: KindTypeTypeType<T, S::Elem>,
    B2::Applied: AsRef<[T]>,
{
    fn eq(&self, other: &Tensor<B2, T, S, C2>) -> bool {
        assert_eq!(
            self.size.as_ref(),
            other.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other.size
        );

        let chunk_size = self.opt_chunk_size.min(other.opt_chunk_size);
        let it_a = self.chunks(chunk_size);
        let it_b = other.chunks(chunk_size);
        let mut it = it_a.streaming_zip(it_b);

        while let Some((chunk_a, chunk_b)) = it.next() {
            for (x, y) in chunk_a.iter().zip(chunk_b.iter()) {
                if x != y {
                    return false;
                }
            }
        }
        true
    }
}
impl<B, T, S, C> Eq for Tensor<B, T, S, C>
where
    T: Eq + 'static,
    S: Axes,
    B: KindTypeTypeType<T, S::Elem>,
    B::Applied: AsRef<[T]>,
{
}

impl<B, B2, T, S, C, C2> PartialOrd<Tensor<B2, T, S, C2>> for Tensor<B, T, S, C>
where
    T: PartialOrd + 'static,
    S: Axes,
    B: KindTypeTypeType<T, S::Elem>,
    B::Applied: AsRef<[T]>,
    B2: KindTypeTypeType<T, S::Elem>,
    B2::Applied: AsRef<[T]>,
{
    fn partial_cmp(&self, other: &Tensor<B2, T, S, C2>) -> Option<Ordering> {
        assert_eq!(
            self.size.as_ref(),
            other.size.as_ref(),
            "Elementwise ops require that operands have the same size. Got {:?} and {:?}.",
            self.size,
            other.size
        );

        let mut res = None;

        let chunk_size = self.opt_chunk_size.min(other.opt_chunk_size);
        let it_a = self.chunks(chunk_size);
        let it_b = other.chunks(chunk_size);
        let mut it = it_a.streaming_zip(it_b);

        while let Some((chunk_a, chunk_b)) = it.next() {
            for (x, y) in chunk_a.iter().zip(chunk_b.iter()) {
                match (res, x.partial_cmp(y)) {
                    (Some(x), Some(y)) => {
                        if x != y {
                            return None;
                        }
                    }
                    (None, r @ Some(_)) => res = r,
                    _ => {
                        return None;
                    }
                }
            }
        }
        res
    }
}

impl<T, S> TryFrom<Vec<T>> for Tensor<VecConstructor, T, S, Contiguous>
where
    S: StaticAxes,
{
    type Error = &'static str;
    fn try_from(buffer: Vec<T>) -> Result<Self, Self::Error> {
        if buffer.len() != S::Elem::USIZE {
            return Err("Input buffer has a length that is incompatible with the tensor size.");
        }

        let size = S::runtime();
        Ok(Tensor {
            buffer,
            size,
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size),
            opt_chunk_size: S::Elem::USIZE,
            _phantoms: PhantomData,
        })
    }
}

impl<T, S> TryFrom<(Vec<T>, <S::Len as StackBuffer<[usize; 1]>>::Buffer)> for Tensor<VecConstructor, T, S, Contiguous>
where
    S: Axes,
{
    type Error = &'static str;
    fn try_from((buffer, size): (Vec<T>, <S::Len as StackBuffer<[usize; 1]>>::Buffer)) -> Result<Self, Self::Error> {
        if buffer.len() != size.as_ref().iter().product() {
            return Err("Input buffer has a length that is incompatible with the tensor size.");
        }

        Ok(Tensor {
            buffer,
            size,
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size),
            opt_chunk_size: size.as_ref().iter().product(),
            _phantoms: PhantomData
        })
    }
}

impl<T, S> TryFrom<(<S::Elem as StackBuffer<[T; 1]>>::Buffer, <S::Len as StackBuffer<[usize; 1]>>::Buffer)> for Tensor<StackBufferConstructor, T, S, Contiguous>
where
    T: Copy,
    S: Axes,
    S::Elem: StackBuffer<[T; 1]>,
{
    type Error = &'static str;
    fn try_from((buffer, size): (<S::Elem as StackBuffer<[T; 1]>>::Buffer, <S::Len as StackBuffer<[usize; 1]>>::Buffer)) -> Result<Self, Self::Error> {
        if buffer.as_ref().len() != size.as_ref().iter().product() {
            return Err("Input buffer has a length that is incompatible with the tensor size.");
        }

        Ok(Tensor {
            buffer,
            size,
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size),
            opt_chunk_size: size.as_ref().iter().product(),
            _phantoms: PhantomData,
        })
    }
}

impl<T, S, L, R> From<Glue<L, R>> for Tensor<StackBufferConstructor, T, S, Contiguous>
where
    T: Copy,
    S: StaticAxes,
    S::Elem: StackBuffer<[T; 1], Buffer = Glue<L, R>>,
{
    fn from(buffer: Glue<L, R>) -> Self {
        let size = S::runtime();
        Tensor {
            buffer,
            size,
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size),
            opt_chunk_size: S::Elem::USIZE,
            _phantoms: PhantomData,
        }
    }
}
