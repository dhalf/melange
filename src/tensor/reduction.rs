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

use super::{Tensor, Static, Dynamic, Strided};
use super::layout::{Layout, DynamicLayout};
use super::shape::{StaticShape, Shape, BroadcastShape, PartialCopy, Same, TRUE};
use super::view::{BroadcastMut, BroadcastDynamicMut};
use std::ops::{AddAssign, MulAssign};
use crate::ops::{MaxAssign, MinAssign};
use super::index::Index;
use std::convert::TryFrom;
use super::alloc::{StaticAlloc, DynamicAlloc};
use super::strided_iterator::StridedIterator;
use num_complex::{Complex, Complex32, Complex64};

type BroadcastMutView<'a, Z, T, Sout: Shape, A> = Tensor<Static, Strided, Z, T, Sout, A, &'a mut [T], DynamicLayout<Sout::Len>>;
type BroadcastDynamicMutView<'a, Z, T, Sout: Shape, A> = Tensor<Dynamic, Strided, Z, T, Sout, A, &'a mut [T], DynamicLayout<Sout::Len>>;

/// Initial values used when performing a
/// summation along given axes
/// for tensors of scalar type `Self`.
/// 
/// Implemented for all primitive numeric types
/// and [`num_complex`] `Complex64` and `Complex32`.
pub trait SumInit {
    const SUM: Self;
}

/// Initial values used when performing a
/// product along given axes
/// for tensors of scalar type `Self`.
/// 
/// Implemented for all primitive numeric types
/// and [`num_complex`] `Complex64` and `Complex32`.
pub trait ProdInit {
    const PROD: Self;
}

/// Initial values used when searching for
/// the maximum along given axes
/// for tensors of scalar type `Self`.
/// 
/// Implemented for all primitive numeric types.
pub trait MaxInit {
    const MAX: Self;
}

/// Initial values used when searching for
/// the minimum along given axes
/// for tensors of scalar type `Self`.
/// 
/// Implemented for all primitive numeric types.
pub trait MinInit {
    const MIN: Self;
}

macro_rules! init_values_impl_float {
    ($($t:ty)*) => ($(
        impl SumInit for $t {
            const SUM: Self = 0.0;
        }

        impl ProdInit for $t {
            const PROD: Self = 1.0;
        }

        impl MaxInit for $t {
            const MAX: Self = <$t>::NEG_INFINITY;
        }

        impl MinInit for $t {
            const MIN: Self = <$t>::INFINITY;
        }
    )*)
}

init_values_impl_float! { f64 f32 }

macro_rules! init_values_impl_integer {
    ($($t:ty)*) => ($(
        impl SumInit for $t {
            const SUM: Self = 0;
        }

        impl ProdInit for $t {
            const PROD: Self = 1;
        }

        impl MaxInit for $t {
            const MAX: Self = <$t>::MIN;
        }

        impl MinInit for $t {
            const MIN: Self = <$t>::MAX;
        }
    )*)
}

init_values_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

macro_rules! init_values_impl_complex {
    ($($t:ty)*) => ($(
        impl SumInit for $t {
            const SUM: Self = Complex::new(0.0, 0.0);
        }

        impl ProdInit for $t {
            const PROD: Self = Complex::new(1.0, 0.0);
        }
    )*)
}

init_values_impl_complex! { Complex64 Complex32 }

/// Summation allong chosen axes of a static tensor.
///
/// The axes are automatically inferred from given or inferred
/// type-level output shape.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
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
/// use melange_scratch::prelude::*;
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
/// use melange_scratch::prelude::*;
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
/// use melange_scratch::prelude::*;
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
    Sum, sum, AddAssign, add_assign, SumInit, T::SUM;
    Prod, prod, MulAssign, mul_assign, ProdInit, T::PROD;
    MaxReduce, max_reduce, MaxAssign, max_assign, MaxInit, T::MAX;
    MinReduce, min_reduce, MinAssign, min_assign, MinInit, T::MIN
}
