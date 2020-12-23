//! Contains mathematical reduction operations.
//!
//! These operations take one input tensor and output a new
//! tensor with a reduced shape,
//! i.e. a shape that can be broadcasted back to the input
//! tensor's shape.
//! This covers sum, product, max and min allong chosen axes.
//!
//! All those operations are implemented using the "reverse broadcasting"
//! trick: the output tensor is mutably broadcasted to the input
//! and the right computation is performed. Since output chunks
//! are yielded more than once, results can accumulate.
//!
//! Similarly to `core_ops`, `reductions` also rely on "chunks loops"
//! to optimize computation time.

use super::alloc::{DynamicAlloc, StaticAlloc};
use super::index::Index;
use super::layout::{DynamicLayout, Layout};
use super::shape::{BroadcastShape, PartialCopy, Same, Shape, StaticShape, TRUE};
use super::strided_iterator::StridedIterator;
use super::view::{BroadcastDynamicMut, BroadcastMut};
use super::{Dynamic, Static, Strided, Tensor};
use crate::ops::{MaxAssign, MinAssign};
use crate::scalar_traits::*;
use std::convert::TryFrom;
use std::ops::{AddAssign, MulAssign};

type BroadcastMutView<'a, Z, T, Sout: Shape, A> =
    Tensor<Static, Strided, Z, T, Sout, A, &'a mut [T], DynamicLayout<Sout::Len>>;
type BroadcastDynamicMutView<'a, Z, T, Sout: Shape, A> =
    Tensor<Dynamic, Strided, Z, T, Sout, A, &'a mut [T], DynamicLayout<Sout::Len>>;

/// Summation allong chosen axes of a static tensor.
///
/// The axes are automatically inferred from given or inferred
/// type-level output shape.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, -2, 1]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U1, U2>> = a.sum();
/// let c: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![-1, 3]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait Sum<Sout> {
    /// Output type.
    type Output;
    /// Returns a the tensor resulting from performing
    /// a summation along chosen axes.
    fn sum(self) -> Self::Output;
}

/// Product allong chosen axes of a static tensor.
///
/// The axes are automatically inferred from given or inferred
/// type-level output shape.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, -2, 1]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U1, U2>> = a.prod();
/// let c: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![-2, 2]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait Prod<Sout> {
    /// Output type.
    type Output;
    /// Returns a the tensor resulting from performing
    /// a summation along chosen axes.
    fn prod(self) -> Self::Output;
}

/// Maximum value allong chosen axes of a static tensor.
///
/// The axes are automatically inferred from given or inferred
/// type-level output shape.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, -2.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U1, U2>> = a.max_reduce();
/// let c: StaticTensor<f64, Shape2D<U1, U2>> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait MaxReduce<Sout> {
    /// Output type.
    type Output;
    /// Returns a the tensor resulting from performing
    /// a summation along chosen axes.
    fn max_reduce(self) -> Self::Output;
}

/// Minimum value allong chosen axes of a static tensor.
///
/// The axes are automatically inferred from given or inferred
/// type-level output shape.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, -2.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U1, U2>> = a.min_reduce();
/// let c: StaticTensor<f64, Shape2D<U1, U2>> = Tensor::try_from(vec![-2.0, 1.0]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait MinReduce<Sout> {
    /// Output type.
    type Output;
    /// Returns a the tensor resulting from performing
    /// a summation along chosen axes.
    fn min_reduce(self) -> Self::Output;
}

macro_rules! reduction_impls {
    (
        $($trait:ident, $trait_fn:ident, $reduction_trait:ident, $reduction_trait_fn:ident, $init_trait:ident, $init_value:expr);*
    ) => {$(
        impl<Y, Z, T, S, A, D, L, Sout> $trait<Sout> for &Tensor<Static, Y, Z, T, S, A, D, L>
        where
            T: $reduction_trait + $init_trait + Copy + 'static,
            S: StaticShape + Clone,
            Sout: StaticShape + Clone + BroadcastShape<S>,
            A: StaticAlloc<T, Sout>,
            for<'a> A::Alloc: BroadcastMut<'a, S, Output=BroadcastMutView<'a, Z, T, S, A>>,
            L: Layout<S::Len>,
            for<'a> &'a Tensor<Static, Y, Z, T, S, A, D, L>: StridedIterator<Item=&'a [T]>,
        {
            type Output = A::Alloc;
            fn $trait_fn(self) -> Self::Output {
                let mut out = A::fill($init_value);
                out.broadcast_mut().$reduction_trait_fn(self);
                out
            }
        }

        impl<Y, Z, T, S, A, D, L, Sout> $trait<Sout> for &Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            T: $reduction_trait + $init_trait + Copy + 'static,
            S: Shape + Same<S>,
            <S as Same<S>>::Output: TRUE,
            Sout: PartialCopy + Clone + BroadcastShape<S>,
            A: DynamicAlloc<T, Sout>,
            for<'a> A::Alloc: BroadcastDynamicMut<'a, S, Output=BroadcastDynamicMutView<'a, Z, T, S, A>>,
            L: Layout<S::Len>,
            for<'a> &'a Tensor<Dynamic, Y, Z, T, S, A, D, L>: StridedIterator<Item=&'a [T]>,
        {
            type Output = A::Alloc;
            fn $trait_fn(self) -> Self::Output {
                let mut output_shape = Vec::from(self.shape());
                Sout::partial_copy(&mut output_shape);

                let mut out: Self::Output = A::fill(Index::try_from(output_shape).unwrap(), $init_value);
                out.broadcast_dynamic_mut(self.shape()).$reduction_trait_fn(self);

                out
            }
        }
    )*};
}

reduction_impls! {
    Sum, sum, AddAssign, add_assign, Zero, T::ZERO;
    Prod, prod, MulAssign, mul_assign, One, T::ONE;
    MaxReduce, max_reduce, MaxAssign, max_assign, NegInfinity, T::NEG_INFINITY;
    MinReduce, min_reduce, MinAssign, min_assign, Infinity, T::INFINITY
}
