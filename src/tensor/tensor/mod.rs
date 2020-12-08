use super::layout::{DynamicLayout, Layout, StaticLayout};
use super::shape::{Shape, Same, StaticShape, intrinsic_strides_in_place};
use super::strided_iterator::{StridedIter, StridedIterMut, StridedIterator, StridedIteratorMut, ChunksMut};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use typenum::bit::{B0, B1};
use crate::gat::RefMutGat;
use std::cmp::Ordering;
use typenum::Unsigned;
use super::index::Index;
use std::convert::TryFrom;

/// Type-level constant aliasing bit B0.
/// 
/// Its conterpart is [`Dynamic`](Dynamic)
pub type Static = B0;

/// Type-level constant aliasing bit B1.
/// 
/// Its conterpart is [`Static`](Static)
pub type Dynamic = B1;

/// Type-level constant aliasing bit B0.
/// 
/// Its conterpart is [`Strided`](Strided)
pub type Contiguous = B0;

/// Type-level constant aliasing bit B1.
/// 
/// Its conterpart is [`Contiguous`](Contiguous)
pub type Strided = B1;

/// Type-level constant aliasing bit B0.
/// 
/// Its conterpart is [`Transposed`](Transposed)
pub type Normal = B0;

/// Type-level constant aliasing bit B1.
/// 
/// Its conterpart is [`Normal`](Normal)
pub type Transposed = B1;

pub mod convert;
pub mod view;

/// Very abstract multidimensional array-like data container.
/// 
/// This is the core struct of this librabry, all vectorized
/// operations are defined on subsets of all types it encompasses.
/// 
/// It takes seven generic arguments:
/// - X, a type-level bit set to B0/B1 if the tensor is static/dynamic
/// respectively, see [`typenum`];
/// - Y, a type-level bit set to B0/B1 if the tensor is contiguous/strided
/// respectively, see [`typenum`];
/// - Z, a type-level bit set to B0/B1 if the tensor is normal/transposed
/// respectively, see [`typenum`];
/// - T, the scalar type of its elements;
/// - S, a type-level array of type-level unsigned integers representing
/// knowledge of its shape at compile time, see [`shape`](crate::tensor::shape);
/// - D, the type of container holding its data (e.g. Vec<T>);
/// - L, a type implementing [`Layout`] that holds the information
/// necessary to map the abstract layout of the tensor to how
/// data is really laid out in memory.
/// 
/// [`Layout`]: crate::tensor::layout::Layout
/// [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
#[derive(Debug)]
pub struct Tensor<X, Y, Z, T, S, D, L> {
    data: D,
    layout: L,
    _phantoms: PhantomData<(X, Y, Z, T, S)>,
}

impl<X, Y, Z, T, S, D, L> Tensor<X, Y, Z, T, S, D, L> {
    /// Returns an immutable slice to the tensor's raw data.
    pub fn as_raw_slice(&self) -> &[T]
    where
        D: Deref<Target = [T]>
    {
        &self.data
    }

    /// Returns an mutable slice to the tensor's raw data.
    pub fn as_raw_slice_mut(&mut self) -> &mut [T]
    where
        D: DerefMut<Target = [T]>
    {
        &mut self.data
    }
}

impl<T, S> Tensor<Static, Contiguous, Normal, T, S, Vec<T>, StaticLayout<S>> {
    /// Returns a static contiguous tensor stored on the heap filled with
    /// the given value.
    pub fn fill(value: T) -> Self
    where
        T: Copy,
        S: StaticShape,
    {
        Tensor {
            data: vec![value; S::NumElements::USIZE],
            layout: StaticLayout::new(),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S> Tensor<Dynamic, Contiguous, Normal, T, S, Vec<T>, DynamicLayout<S::Len>>
where
    T: Copy,
    S: Shape,
{
    /// Returns a static contiguous tensor stored on the heap filled with
    /// the given value.
    pub fn fill_dynamic(shape: Index<S::Len>, value: T) -> Self {
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
}

impl<X, T, S, D, L> Clone for Tensor<X, Contiguous, Normal, T, S, D, L>
where
    D: Clone,
    L: Clone,
{
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            layout: self.layout.clone(),
            _phantoms: PhantomData,
        }
    }
}

impl<X, Y, Z, T, S, D, L> Deref for Tensor<X, Y, Z, T, S, D, L> {
    type Target = L;
    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl<'a, X, Z, T, S, D, L> StridedIterator for &'a Tensor<X, Contiguous, Z, T, S, D, L>
where
    S: Shape,
    D: Deref<Target = [T]>,
{
    type Item = &'a [T];
    type StridedIter = std::slice::Chunks<'a, T>;
    fn strided_iter(self, chunk_size: usize) -> Self::StridedIter {
        self.data.chunks(chunk_size)
    }
}

impl<'a, X, Z, T, S, D, L> StridedIterator for &'a Tensor<X, Strided, Z, T, S, D, L>
where
    S: Shape,
    L: Layout<S::Len>,
    D: Deref<Target = [T]>,
{
    type Item = &'a [T];
    type StridedIter = StridedIter<'a, T, S::Len>;
    fn strided_iter(self, chunk_size: usize) -> Self::StridedIter {
        StridedIter::new(&self.data, &self.layout, chunk_size)
    }
}

impl<'a, X, Z, T, S, D, L> StridedIteratorMut for &'a mut Tensor<X, Contiguous, Z, T, S, D, L>
where
    T: 'static,
    D: DerefMut<Target = [T]>,
{
    type Item = RefMutGat<[T]>;
    type StridedIterMut = ChunksMut<'a, T>;
    fn strided_iter_mut(self, chunk_size: usize) -> Self::StridedIterMut {
        ChunksMut::new(&mut self.data, chunk_size)
    }
}

impl<'a, X, Z, T, S, D, L> StridedIteratorMut for &'a mut Tensor<X, Strided, Z, T, S, D, L>
where
    T: 'static,
    S: Shape,
    L: Layout<S::Len>,
    D: DerefMut<Target = [T]>,
{
    type Item = RefMutGat<[T]>;
    type StridedIterMut = StridedIterMut<'a, T, S::Len>;
    fn strided_iter_mut(self, chunk_size: usize) -> Self::StridedIterMut {
        StridedIterMut::new(&mut self.data, &self.layout, chunk_size)
    }
}

macro_rules! cmp_fn_impl {
    ($fn_name:ident $op:tt) => {
        fn $fn_name(&self, rhs: &Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Drhs, Lrhs>) -> bool {
            if self.shape() == rhs.shape() {
                let opt_chunk_size = self.opt_chunk_size().min(rhs.opt_chunk_size());
                for (self_chunk, rhs_chunk) in self.strided_iter(opt_chunk_size).zip(rhs.strided_iter(opt_chunk_size)) {
                    if self_chunk.iter().zip(rhs_chunk.iter()).any(|(x, y)| *x $op *y) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
    };
}

macro_rules! scalar_cmp_fn_impl {
    ($fn_name:ident $op:tt) => {
        fn $fn_name(&self, rhs: &T) -> bool {
            let opt_chunk_size = self.opt_chunk_size();
            for chunk in self.strided_iter(opt_chunk_size) {
                if chunk.iter().any(|x| *x $op *rhs) {
                    return false;
                }
            }
            true
        }
    };
}

impl<X, Y, Z, T, S, D, L, Xrhs, Yrhs, Zrhs, Srhs, Drhs, Lrhs> PartialEq<Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Drhs, Lrhs>> for Tensor<X, Y, Z, T, S, D, L>
where
    for<'a> &'a Self: StridedIterator<Item = &'a [T]>,
    for<'a> &'a Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Drhs, Lrhs>: StridedIterator<Item = &'a [T]>,
    T: PartialEq,
    S: Shape + Same<Srhs>,
    L: Layout<S::Len>,
    Lrhs: Layout<S::Len>,
{
    cmp_fn_impl! { eq != }
}

impl<X, Y, Z, T, S, D, L> PartialEq<T> for Tensor<X, Y, Z, T, S, D, L>
where
    for<'a> &'a Self: StridedIterator<Item = &'a [T]>,
    T: PartialEq,
    S: Shape,
    L: Layout<S::Len>,
{
    scalar_cmp_fn_impl! { eq != }
}

impl<X, Y, Z, T, S, D, L, Xrhs, Yrhs, Zrhs, Srhs, Drhs, Lrhs> PartialOrd<Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Drhs, Lrhs>> for Tensor<X, Y, Z, T, S, D, L>
where
    for<'a> &'a Self: StridedIterator<Item = &'a [T]>,
    for<'a> &'a Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Drhs, Lrhs>: StridedIterator<Item = &'a [T]>,
    T: PartialOrd,
    S: Shape + Same<Srhs>,
    L: Layout<S::Len>,
    Lrhs: Layout<S::Len>,
{
    #[inline]
    fn partial_cmp(&self, rhs: &Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Drhs, Lrhs>) -> Option<Ordering> {
        match (self <= rhs, self >= rhs) {
            (false, false) => None,
            (false, true) => Some(Ordering::Greater),
            (true, false) => Some(Ordering::Less),
            (true, true) => Some(Ordering::Equal),
        }
    }

    cmp_fn_impl! { lt >= }
    cmp_fn_impl! { gt <= }
    cmp_fn_impl! { le > }
    cmp_fn_impl! { ge < }
}

impl<X, Y, Z, T, S, D, L> PartialOrd<T> for Tensor<X, Y, Z, T, S, D, L>
where
    for<'a> &'a Self: StridedIterator<Item = &'a [T]>,
    T: PartialOrd,
    S: Shape,
    L: Layout<S::Len>,
{
    #[inline]
    fn partial_cmp(&self, rhs: &T) -> Option<Ordering> {
        match (self <= rhs, self >= rhs) {
            (false, false) => None,
            (false, true) => Some(Ordering::Greater),
            (true, false) => Some(Ordering::Less),
            (true, true) => Some(Ordering::Equal),
        }
    }

    scalar_cmp_fn_impl! { lt >= }
    scalar_cmp_fn_impl! { gt <= }
    scalar_cmp_fn_impl! { le > }
    scalar_cmp_fn_impl! { ge < }
}
