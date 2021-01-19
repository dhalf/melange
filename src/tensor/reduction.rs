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
use super::shape::{BroadcastShape, PartialCopy, Same, Shape, Shape1D, StaticShape, TRUE, UpsamplingStrides};
use super::strided_iterator::StridedIterator;
use super::view::{BroadcastDynamic, Broadcast, StrideDynamic, Stride};
use super::{AsRawSlice, Dynamic, Static, Strided, Tensor};
use crate::ops::{MaxAssign, MinAssign};
use crate::scalar_traits::*;
use num_complex::{Complex32, Complex64};
use std::convert::TryFrom;
use std::ops::{AddAssign, MulAssign};
use typenum::U1;

type BroadcastMutView<'a, Z, T, Sout: Shape, A> =
    Tensor<Static, Strided, Z, T, Sout, A, &'a mut [T], DynamicLayout<Sout::Len>>;
type BroadcastDynamicMutView<'a, Z, T, Sout: Shape, A> =
    Tensor<Dynamic, Strided, Z, T, Sout, A, &'a mut [T], DynamicLayout<Sout::Len>>;

/// Summation allong chosen axes.
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
    /// Returns the tensor resulting from performing
    /// a summation along chosen axes.
    fn sum(self) -> Self::Output;
}

/// Product allong chosen axes.
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
    /// Returns the tensor resulting from performing
    /// a product along chosen axes.
    fn prod(self) -> Self::Output;
}

/// Maximum value allong chosen axes.
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
    /// Returns the tensor resulting from performing
    /// a maximum operation along chosen axes.
    fn max_reduce(self) -> Self::Output;
}

/// Minimum value allong chosen axes.
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
    /// Returns the tensor resulting from performing
    /// a minimum operation along chosen axes.
    fn min_reduce(self) -> Self::Output;
}

/// Upsampling allong chosen axes.
///
/// The axes and number of inserted zeros
/// are automatically inferred from given or inferred
/// type-level output shape.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
///     1.0, 2.0,
///     3.0, 4.0,
/// ]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U4, U4>> = a.upsample();
/// let c: StaticTensor<f64, Shape2D<U4, U4>> = Tensor::try_from(vec![
///     1.0, 0.0, 2.0, 0.0,
///     0.0, 0.0, 0.0, 0.0,
///     3.0, 0.0, 4.0, 0.0,
///     0.0, 0.0, 0.0, 0.0,
/// ]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait Upsample<Sout> {
    /// Output type.
    type Output;
    /// Returns the upsampled tensor along chosen axes.
    fn upsample(self) -> Self::Output;
}

/// Upsampling allong chosen axes for dynamic tensors.
///
/// The axes and number of inserted zeros
/// are automatically inferred from given or inferred
/// type-level output shape and provided runtime output
/// shape.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: DynamicTensor<f64, Shape2D<Dyn, U2>> = Tensor::try_from(vec![
///     1.0, 2.0,
///     3.0, 4.0,
/// ]).unwrap();
/// let b: DynamicTensor<f64, Shape2D<Dyn, U4>> = a.upsample_dynamic(Index::try_from(vec![4, 4]).unwrap());
/// let c: StaticTensor<f64, Shape2D<U4, U4>> = Tensor::try_from(vec![
///     1.0, 0.0, 2.0, 0.0,
///     0.0, 0.0, 0.0, 0.0,
///     3.0, 0.0, 4.0, 0.0,
///     0.0, 0.0, 0.0, 0.0,
/// ]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait UpsampleDynamic<Sout>
where
    Sout: Shape,
{
    /// Output type.
    type Output;
    /// Returns the upsampled tensor along chosen axes.
    fn upsample_dynamic(self, runtime_shape: Index<Sout::Len>) -> Self::Output;
}

/// Gradient of dynamic broadcast.
/// 
/// In essence, this is just a summation
/// allong axes chosen at runtime.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U1, U2};
///
/// let a: DynamicTensor<i32, Shape2D<Dyn, U2>> = Tensor::try_from(vec![1, 2, -2, 1]).unwrap();
/// let b: DynamicTensor<i32, Shape2D<Dyn, U2>> = a.broadcast_dynamic_back(Index::try_from(vec![1, 2]).unwrap());
/// let c: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![-1, 3]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait BroadcastDynamicBack<Sout>
where
    Sout: Shape,
{
    /// Output type.
    type Output;
    /// Returns the gradient.
    fn broadcast_dynamic_back(self, runtime_shape: Index<Sout::Len>) -> Self::Output;
}

/// Gradient of striding.
///
/// In essence, this is just upsampling
/// using given strides.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U3, U4};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
///     1.0, 2.0,
///     3.0, 4.0,
/// ]).unwrap();
/// let b = StrideBack::<Shape2D<U4, U4>, Shape2D<U2, U3>>::stride_back(&a);
/// let c: StaticTensor<f64, Shape2D<U4, U4>> = Tensor::try_from(vec![
///     1.0, 0.0, 0.0, 2.0,
///     0.0, 0.0, 0.0, 0.0,
///     3.0, 0.0, 0.0, 4.0,
///     0.0, 0.0, 0.0, 0.0,
/// ]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait StrideBack<Sout, Strides> {
    /// Output type.
    type Output;
    /// Returns the gradient.
    fn stride_back(self) -> Self::Output;
}

/// Gradient of dynamic striding.
///
/// In essence, this is just upsampling
/// using given strides.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U3, U4};
///
/// let a: DynamicTensor<f64, Shape2D<Dyn, U2>> = Tensor::try_from(vec![
///     1.0, 2.0,
///     3.0, 4.0,
/// ]).unwrap();
/// let b = StrideDynamicBack::<Shape2D<Dyn, U4>, Shape2D<Dyn, U3>>::stride_dynamic_back(&a, Index::try_from(vec![4, 4]).unwrap(), Index::try_from(vec![2, 3]).unwrap());
/// let c: StaticTensor<f64, Shape2D<U4, U4>> = Tensor::try_from(vec![
///     1.0, 0.0, 0.0, 2.0,
///     0.0, 0.0, 0.0, 0.0,
///     3.0, 0.0, 0.0, 4.0,
///     0.0, 0.0, 0.0, 0.0,
/// ]).unwrap();
/// assert_eq!(b, c);
/// ```
pub trait StrideDynamicBack<Sout, Strides>
where
    Sout: Shape,
    Strides: Shape,
{
    /// Output type.
    type Output;
    /// Returns the gradient.
    fn stride_dynamic_back(self, runtime_shape: Index<Sout::Len>, runtime_strides: Index<Strides::Len>) -> Self::Output;
}

macro_rules! reduction_impls {
    (
        $($trait:ident, $trait_fn:ident, $reduction_trait:ident, $reduction_trait_fn:ident, $init_trait:ident, $init_value:expr);*
    ) => {$(
        impl<Y, Z, T, S, A, D, L, Sout> $trait<Sout> for &Tensor<Static, Y, Z, T, S, A, D, L>
        where
            T: $init_trait,
            S: Shape,
            Sout: StaticShape + BroadcastShape<S>,
            A: StaticAlloc<T, Sout>,
            for<'a> &'a mut A::Alloc: Broadcast<S, Output=Tensor<Static, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
            for<'a> Tensor<Static, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: $reduction_trait<Self>,
        {
            type Output = A::Alloc;
            fn $trait_fn(self) -> Self::Output {
                let mut out = A::fill($init_value);
                out.broadcast().$reduction_trait_fn(self);
                out
            }
        }

        impl<Y, Z, T, S, A, D, L, Sout> $trait<Sout> for &Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            T: $init_trait,
            S: Shape + Same<S>,
            <S as Same<S>>::Output: TRUE,
            Sout: PartialCopy + BroadcastShape<S>,
            A: DynamicAlloc<T, Sout>,
            L: Layout<S::Len>,
            for<'a> &'a mut A::Alloc: BroadcastDynamic<S, Output=Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
            for<'a> Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: $reduction_trait<Self>,
        {
            type Output = A::Alloc;
            fn $trait_fn(self) -> Self::Output {
                let mut output_shape = Vec::from(self.shape());
                Sout::partial_copy(&mut output_shape);

                let mut out: Self::Output = A::fill(Index::try_from(output_shape).unwrap(), $init_value);
                out.broadcast_dynamic(self.shape()).$reduction_trait_fn(self);

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

// Note: the orphan rules prevents a generic impl here
//
// These impls are needed to be able to backpropagate
// through a scalar in a tensor/scalar operation.
// This form of add_assign actually consists of a
// summation of all the elements of the tensor that
// is added to the scalar by delegating to its impl
// of AddAssign.
macro_rules! implicit_summation {
    ($($t:ty)*) => {$(
        impl<Y, Z, S, A, D, L> AddAssign<&Tensor<Static, Y, Z, $t, S, A, D, L>> for $t
        where
            $t: AddAssign,
            for<'a> &'a Tensor<Static, Y, Z, $t, S, A, D, L>: Sum<Shape1D<U1>, Output = A::Alloc>,
            A: StaticAlloc<$t, Shape1D<U1>>,
            A::Alloc: AsRawSlice<$t>,
        {
            fn add_assign(&mut self, rhs: &Tensor<Static, Y, Z, $t, S, A, D, L>) {
                *self += rhs.sum().as_raw_slice()[0]
            }
        }

        impl<Y, Z, S, A, D, L> AddAssign<&Tensor<Dynamic, Y, Z, $t, S, A, D, L>> for $t
        where
            $t: AddAssign,
            S: Shape,
            for<'a> &'a Tensor<Dynamic, Y, Z, $t, S, A, D, L>: Sum<Shape1D<U1>, Output = A::Alloc>,
            A: DynamicAlloc<$t, Shape1D<U1>>,
            A::Alloc: AsRawSlice<$t>,
        {
            fn add_assign(&mut self, rhs: &Tensor<Dynamic, Y, Z, $t, S, A, D, L>) {
                *self += rhs.sum().as_raw_slice()[0]
            }
        }
    )*};
}

implicit_summation! { f64 f32 Complex64 Complex32 }

impl<Y, Z, T, S, A, D, L, Sout> Upsample<Sout> for &Tensor<Static, Y, Z, T, S, A, D, L>
where
    T: Zero,
    S: Shape,
    Sout: StaticShape + UpsamplingStrides<S>,
    A: StaticAlloc<T, Sout>,
    for<'a> &'a mut A::Alloc: Stride<<Sout as UpsamplingStrides<S>>::Output, Output=Tensor<Static, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
    for<'a> Tensor<Static, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: AddAssign<Self>,
{
    type Output = A::Alloc;
    fn upsample(self) -> A::Alloc {
        let mut out = A::fill(T::ZERO);
        out.stride().add_assign(self);
        out
    }
}

impl<Y, Z, T, S, A, D, L, Sout> UpsampleDynamic<Sout> for &Tensor<Dynamic, Y, Z, T, S, A, D, L>
where
    T: Zero,
    S: Shape + Same<S>,
    <S as Same<S>>::Output: TRUE,
    Sout: Shape + UpsamplingStrides<S>,
    A: DynamicAlloc<T, Sout>,
    L: Layout<S::Len>,
    for<'a> &'a mut A::Alloc: StrideDynamic<<Sout as UpsamplingStrides<S>>::Output, Output=Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
    for<'a> Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: AddAssign<Self>,
{
    type Output = A::Alloc;
    fn upsample_dynamic(self, runtime_shape: Index<Sout::Len>) -> Self::Output {
        assert!(
            Sout::runtime_compat(&runtime_shape),
            "`runtime_shape` is incompatible with static shape `Sout`."
        );
        let runtime_strides: Vec<_> = runtime_shape.iter().zip(self.shape().iter()).map(|(o, i)| o / i).collect();
        let runtime_strides = Index::try_from(runtime_strides).unwrap();
        let mut out: Self::Output = A::fill(Index::try_from(runtime_shape).unwrap(), T::ZERO);
        out.stride_dynamic(runtime_strides).add_assign(self);

        out
    }
}

impl<Y, Z, T, S, A, D, L, Sout> BroadcastDynamicBack<Sout> for &Tensor<Dynamic, Y, Z, T, S, A, D, L>
where
    T: Zero,
    S: Shape + Same<S>,
    <S as Same<S>>::Output: TRUE,
    Sout: Shape + BroadcastShape<S>,
    A: DynamicAlloc<T, Sout>,
    L: Layout<S::Len>,
    for<'a> &'a mut A::Alloc: BroadcastDynamic<S, Output=Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
    for<'a> Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: AddAssign<Self>,
{
    type Output = A::Alloc;
    fn broadcast_dynamic_back(self, runtime_shape: Index<Sout::Len>) -> Self::Output {
        let mut out: Self::Output = A::fill(runtime_shape, T::ZERO);
        out.broadcast_dynamic(self.shape()).add_assign(self);

        out
    }
}

impl<Y, Z, T, S, A, D, L, Sout, Strides> StrideBack<Sout, Strides> for &Tensor<Static, Y, Z, T, S, A, D, L>
where
    T: Zero,
    S: Shape,
    Sout: StaticShape,
    A: StaticAlloc<T, Sout>,
    for<'a> &'a mut A::Alloc: Stride<Strides, Output=Tensor<Static, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
    for<'a> Tensor<Static, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: AddAssign<Self>,
{
    type Output = A::Alloc;
    fn stride_back(self) -> A::Alloc {
        let mut out = A::fill(T::ZERO);
        out.stride().add_assign(self);
        out
    }
}

impl<Y, Z, T, S, A, D, L, Sout, Strides> StrideDynamicBack<Sout, Strides> for &Tensor<Dynamic, Y, Z, T, S, A, D, L>
where
    T: Zero,
    S: Shape + Same<S>,
    <S as Same<S>>::Output: TRUE,
    Sout: Shape,
    Strides: Shape,
    A: DynamicAlloc<T, Sout>,
    L: Layout<S::Len>,
    for<'a> &'a mut A::Alloc: StrideDynamic<Strides, Output=Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>>,
    for<'a> Tensor<Dynamic, Strided, Z, T, S, A, &'a mut [T], DynamicLayout<S::Len>>: AddAssign<Self>,
{
    type Output = A::Alloc;
    fn stride_dynamic_back(self, runtime_shape: Index<Sout::Len>, runtime_strides: Index<Strides::Len>) -> Self::Output {
        assert!(
            Sout::runtime_compat(&runtime_shape),
            "`runtime_shape` is incompatible with static shape `Sout`."
        );
        let mut out: Self::Output = A::fill(Index::try_from(runtime_shape).unwrap(), T::ZERO);
        out.stride_dynamic(runtime_strides).add_assign(self);

        out
    }
}
