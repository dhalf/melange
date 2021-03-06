//! Contains algebra-specific operations.
//! It is currently limited to vector/vector, matrix/vector,
//! and matrix/matrix dot products. It is entirely backed by
//! openblas through C bindings.
//!
//! Note that only 1 dimmensional tensors are considered vectors
//! and that only two dimmensional tensors are considered matrices.

extern crate cblas;
extern crate openblas_src;

use super::alloc::{DynamicAlloc, StaticAlloc};
use super::index::Index;
use super::layout::Layout;
use super::shape::{Dim, Shape1D, Shape2D, StaticDim, StaticShape, TRUE};
use super::tensor::{Contiguous, Dynamic, Normal, Static, Tensor, Transposed};
use super::{AsRawSlice, AsRawSliceMut};
use crate::scalar_traits::{One, Zero};
use cblas::{
    cdotu_sub, cgemm, cgemv, ddot, dgemm, dgemv, sdot, sgemm, sgemv, zdotu_sub, zgemm, zgemv,
    Transpose,
};
use num_complex::{Complex32, Complex64};
use std::convert::TryFrom;
use std::ops::{Deref, DerefMut};
use typenum::{Eq, IsEqual, Unsigned, U1, U2};

/// Defines the constant that should be passed to BLAS operations.
pub trait BLASTranspose {
    /// Constant to be passed to BLAS.
    const BLAS_TRANSPOSE: Transpose;
}

impl BLASTranspose for Contiguous {
    const BLAS_TRANSPOSE: Transpose = Transpose::None;
}

impl BLASTranspose for Transposed {
    const BLAS_TRANSPOSE: Transpose = Transpose::Ordinary;
}

/// Dot product operator.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// `Rhs` should have a shape compatible with the shape of `Self`.
/// For dot product, this means the last axis of `Self` and the
/// first of `Rhs` must be the same (i.e they are the same type-level
/// integer or one is `Dyn`).
///
/// Implemented for all tensors whose scalar type is `f64` or `f32`
/// (this limitation originates from BLAS) if one of the following
/// statements holds:
/// * `self` is 1D (vector) and `rhs` is 1D (vector)
/// * `self` is 2D (matrix) and `rhs` is 1D (vector)
/// * `self` is 2D (matrix) and `rhs` is 2D (matrix)
///
/// # Panics
/// `dot` panics with dynamic tensors if contracted dimensions are
/// not equal.
///
/// # Examples
/// ```no_run
/// use melange::prelude::*;
/// use std::f64::consts::FRAC_PI_2;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
///     FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
///     FRAC_PI_2.sin(), FRAC_PI_2.cos()
/// ]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![0.0, 1.0]).unwrap();
/// assert!(a.dot(&b).sub(&c) < f64::EPSILON);
/// ```
///
/// ```no_run
/// use melange::prelude::*;
/// use std::f64::consts::FRAC_PI_2;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
///     FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
///     FRAC_PI_2.sin(), FRAC_PI_2.cos()
/// ]).unwrap();
/// let b: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
/// let c: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![0.0, 1.0]).unwrap();
/// assert!(a.dot(&b).sub(&c) < f64::EPSILON);
/// ```
///
/// ```no_run
/// use melange::prelude::*;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, -4.0]).unwrap();
/// let b: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![6.0, 2.0]).unwrap();
/// assert_eq!(a.dot(&b), -2.0);
/// ```
pub trait Dot<Rhs> {
    /// Output type.
    type Output;

    /// Performs the dot product of `self` and `rhs`.
    fn dot(self, rhs: Rhs) -> Self::Output;
}

/// Fused dot product and addition operator.
///
/// Returns `self . rhs0 + rhs1`.
/// This is particularly useful for
/// Fully Connected layers in Deep Neural Networks.
///
/// Note that `Rhs0` and `Rhs1 are `Self` by default,
/// but this is not mandatory.
///
/// `Rhs0` and `Rhs1 should have a shape compatible with
/// the shape of `Self`.
/// `Rhs0` should be compatible for dot product, this means
/// the last axis of `Self` and the first of `Rhs0` must be
/// the same (i.e they are the same type-level integer or one
/// is `Dyn`).
/// `Rhs1` should be compatible for addition, its shape must be
/// compatible with the output of the dot product.
///
/// Implemented for all tensors whose scalar type is `f64` or `f32`
/// (this limitation originates from BLAS) if one of the following
/// statements holds:
/// * `self` is 2D (matrix), `rhs0` is 1D (vector) and `rhs1` is 1D (vector)
/// * `self` is 2D (matrix), `rhs0` is 2D (matrix) and `rhs1` is 2D (matrix)
///
/// # Panics
/// `dot_add` panics with dynamic tensors if contracted dimensions are
/// not equal or if non contracted dimensions and `rhs1` shape are not
/// compatible.
///
/// # Examples
/// ```no_run
/// use melange::prelude::*;
/// use std::f64::consts::FRAC_PI_2;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
///     FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
///     FRAC_PI_2.sin(), FRAC_PI_2.cos()
/// ]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 1.0]).unwrap();
/// let d: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
/// assert!(a.dot_add(&b, &c).sub(&d) < f64::EPSILON);
/// ```
///
/// ```no_run
/// use melange::prelude::*;
/// use std::f64::consts::FRAC_PI_2;
/// use typenum::{U1, U2};
///
/// let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
///     FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
///     FRAC_PI_2.sin(), FRAC_PI_2.cos()
/// ]).unwrap();
/// let b: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
/// let c: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 1.0]).unwrap();
/// let d: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
/// assert!(a.dot_add(&b, &c).sub(&d) < f64::EPSILON);
/// ```
pub trait DotAdd<Rhs0, Rhs1> {
    /// Output type.
    type Output;

    /// Returns `self . rhs0 + rhs1`.
    fn dot_add(self, rhs0: Rhs0, rhs1: Rhs1) -> Self::Output;
}

macro_rules! mmdot_impl {
    ($t:ty; $blas_fn:ident) => {
        impl<Z, M, K, A, D, L, Zrhs, N, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Static, Contiguous, Zrhs, $t, Shape2D<K, N>, Arhs, Drhs, Lrhs>>
            for &Tensor<Static, Contiguous, Z, $t, Shape2D<M, K>, A, D, L>
        where
            Z: BLASTranspose,
            Zrhs: BLASTranspose,
            M: StaticDim,
            N: StaticDim,
            K: StaticDim,
            Shape2D<M, N>: StaticShape,
            A: StaticAlloc<$t, Shape2D<M, N>>,
            A::Alloc: AsRawSliceMut<$t>,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U2>,
            Lrhs: Layout<U2>,
        {
            type Output = A::Alloc;
            fn dot(
                self,
                other: &Tensor<Static, Contiguous, Zrhs, $t, Shape2D<K, N>, Arhs, Drhs, Lrhs>,
            ) -> Self::Output {
                let mut out = A::fill(<$t>::ZERO);

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        Zrhs::BLAS_TRANSPOSE,
                        M::I32,
                        N::I32,
                        K::I32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        K::I32,
                        other.as_raw_slice(),
                        N::I32,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        N::I32,
                    );
                }

                out
            }
        }

        impl<Z, M, K, A, D, L, Zrhs, Krhs, N, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Dynamic, Contiguous, Zrhs, $t, Shape2D<Krhs, N>, Arhs, Drhs, Lrhs>>
            for &Tensor<Dynamic, Contiguous, Z, $t, Shape2D<M, K>, A, D, L>
        where
            Z: BLASTranspose,
            Zrhs: BLASTranspose,
            M: Dim,
            N: Dim,
            K: IsEqual<Krhs>,
            Eq<K, Krhs>: TRUE,
            A: DynamicAlloc<$t, Shape2D<M, N>>,
            A::Alloc: AsRawSliceMut<$t>,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U2>,
            Lrhs: Layout<U2>,
        {
            type Output = A::Alloc;
            fn dot(
                self,
                other: &Tensor<Dynamic, Contiguous, Zrhs, $t, Shape2D<Krhs, N>, Arhs, Drhs, Lrhs>,
            ) -> Self::Output {
                let self_shape = self.shape();
                let other_shape = other.shape();
                assert_eq!(
                    self_shape[1], other_shape[0],
                    "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
                    self_shape[1], other_shape[0], self_shape, other_shape,
                );

                let mut out = A::fill(
                    Index::try_from(vec![self_shape[0], other_shape[1]]).unwrap(),
                    <$t>::ZERO,
                );

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        Zrhs::BLAS_TRANSPOSE,
                        self_shape[0] as i32,
                        other_shape[1] as i32,
                        self_shape[1] as i32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        self_shape[1] as i32,
                        other.as_raw_slice(),
                        other_shape[1] as i32,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        other_shape[1] as i32,
                    );
                }

                out
            }
        }
    };
}

mmdot_impl! { f64; dgemm }
mmdot_impl! { f32; sgemm }
mmdot_impl! { Complex64; zgemm }
mmdot_impl! { Complex32; cgemm }

macro_rules! mvdot_impl {
    ($t:ty; $blas_fn:ident) => {
        impl<Z, M, N, A, D, L, Zrhs, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Static, Contiguous, Zrhs, $t, Shape1D<N>, Arhs, Drhs, Lrhs>>
            for &Tensor<Static, Contiguous, Z, $t, Shape2D<M, N>, A, D, L>
        where
            Z: BLASTranspose,
            M: StaticDim,
            N: StaticDim,
            Shape1D<M>: StaticShape,
            A: StaticAlloc<$t, Shape1D<M>>,
            A::Alloc: AsRawSliceMut<$t>,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U2>,
            Lrhs: Layout<U1>,
        {
            type Output = A::Alloc;

            fn dot(
                self,
                other: &Tensor<Static, Contiguous, Zrhs, $t, Shape1D<N>, Arhs, Drhs, Lrhs>,
            ) -> Self::Output {
                let mut out: Self::Output = A::fill(<$t>::ZERO);

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        M::I32,
                        N::I32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        N::I32,
                        other.as_raw_slice(),
                        1,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        1,
                    );
                }

                out
            }
        }

        impl<Z, M, N, A, D, L, Zrhs, Nrhs, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Dynamic, Contiguous, Zrhs, $t, Shape1D<Nrhs>, Arhs, Drhs, Lrhs>>
            for &Tensor<Dynamic, Contiguous, Z, $t, Shape2D<M, N>, A, D, L>
        where
            Z: BLASTranspose,
            M: Dim,
            N: IsEqual<Nrhs>,
            Eq<N, Nrhs>: TRUE,
            A: DynamicAlloc<$t, Shape1D<M>>,
            A::Alloc: AsRawSliceMut<$t>,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U2>,
            Lrhs: Layout<U1>,
        {
            type Output = A::Alloc;

            fn dot(
                self,
                other: &Tensor<Dynamic, Contiguous, Zrhs, $t, Shape1D<Nrhs>, Arhs, Drhs, Lrhs>,
            ) -> Self::Output {
                let self_shape = self.shape();
                let other_shape = other.shape();
                assert_eq!(
                    self_shape[1], other_shape[0],
                    "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
                    self_shape[1], other_shape[0], self_shape, other_shape,
                );

                let mut out: Self::Output =
                    A::fill(Index::try_from(vec![self_shape[0]]).unwrap(), <$t>::ZERO);

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        self_shape[0] as i32,
                        self_shape[1] as i32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        self_shape[1] as i32,
                        other.as_raw_slice(),
                        1,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        1,
                    );
                }

                out
            }
        }
    };
}

mvdot_impl! { f64; dgemv }
mvdot_impl! { f32; sgemv }
mvdot_impl! { Complex64; zgemv }
mvdot_impl! { Complex32; cgemv }

macro_rules! vvdot_impl {
    ($t:ty; $blas_fn:ident) => {
        impl<Z, N, A, D, L, Zrhs, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Static, Contiguous, Zrhs, $t, Shape1D<N>, Arhs, Drhs, Lrhs>>
            for &Tensor<Static, Contiguous, Z, $t, Shape1D<N>, A, D, L>
        where
            Z: BLASTranspose,
            N: Unsigned,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U1>,
            Lrhs: Layout<U1>,
        {
            type Output = $t;

            fn dot(
                self,
                other: &Tensor<Static, Contiguous, Zrhs, $t, Shape1D<N>, Arhs, Drhs, Lrhs>,
            ) -> $t {
                unsafe { $blas_fn(N::I32, self.as_raw_slice(), 1, other.as_raw_slice(), 1) }
            }
        }

        impl<Z, N, A, D, L, Zrhs, Nrhs, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Dynamic, Contiguous, Zrhs, $t, Shape1D<Nrhs>, Arhs, Drhs, Lrhs>>
            for &Tensor<Dynamic, Contiguous, Z, $t, Shape1D<N>, A, D, L>
        where
            Z: BLASTranspose,
            N: IsEqual<Nrhs>,
            Eq<N, Nrhs>: TRUE,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U1>,
            Lrhs: Layout<U1>,
        {
            type Output = $t;

            fn dot(
                self,
                other: &Tensor<Dynamic, Contiguous, Zrhs, $t, Shape1D<Nrhs>, Arhs, Drhs, Lrhs>,
            ) -> $t {
                let self_shape = self.shape();
                let other_shape = other.shape();
                assert_eq!(
                    self_shape[0], other_shape[0],
                    "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
                    self_shape[0], other_shape[0], self_shape, other_shape,
                );

                unsafe {
                    $blas_fn(
                        self_shape[0] as i32,
                        self.as_raw_slice(),
                        1,
                        other.as_raw_slice(),
                        1,
                    )
                }
            }
        }
    };
}

vvdot_impl! { f64; ddot }
vvdot_impl! { f32; sdot }

macro_rules! array_vvdot_impl {
    ($t:ty; $blas_fn:ident) => {
        impl<Z, N, A, D, L, Zrhs, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Static, Contiguous, Zrhs, $t, Shape1D<N>, Arhs, Drhs, Lrhs>>
            for &Tensor<Static, Contiguous, Z, $t, Shape1D<N>, A, D, L>
        where
            Z: BLASTranspose,
            N: Unsigned,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U1>,
            Lrhs: Layout<U1>,
        {
            type Output = $t;

            fn dot(
                self,
                other: &Tensor<Static, Contiguous, Zrhs, $t, Shape1D<N>, Arhs, Drhs, Lrhs>,
            ) -> $t {
                let mut out: [$t; 1] = [<$t>::ZERO];
                unsafe {
                    $blas_fn(
                        N::I32,
                        self.as_raw_slice(),
                        1,
                        other.as_raw_slice(),
                        1,
                        &mut out,
                    );
                }
                out[0]
            }
        }

        impl<Z, N, A, D, L, Zrhs, Nrhs, Arhs, Drhs, Lrhs>
            Dot<&Tensor<Dynamic, Contiguous, Zrhs, $t, Shape1D<Nrhs>, Arhs, Drhs, Lrhs>>
            for &Tensor<Dynamic, Contiguous, Z, $t, Shape1D<N>, A, D, L>
        where
            Z: BLASTranspose,
            N: IsEqual<Nrhs>,
            Eq<N, Nrhs>: TRUE,
            D: Deref<Target = [$t]>,
            Drhs: Deref<Target = [$t]>,
            L: Layout<U1>,
            Lrhs: Layout<U1>,
        {
            type Output = $t;

            fn dot(
                self,
                other: &Tensor<Dynamic, Contiguous, Zrhs, $t, Shape1D<Nrhs>, Arhs, Drhs, Lrhs>,
            ) -> $t {
                let self_shape = self.shape();
                let other_shape = other.shape();
                assert_eq!(
                    self_shape[0], other_shape[0],
                    "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
                    self_shape[0], other_shape[0], self_shape, other_shape,
                );

                let mut out: [$t; 1] = [<$t>::ZERO];
                unsafe {
                    $blas_fn(
                        self_shape[0] as i32,
                        self.as_raw_slice(),
                        1,
                        other.as_raw_slice(),
                        1,
                        &mut out,
                    )
                }
                out[0]
            }
        }
    };
}

array_vvdot_impl! { Complex64; zdotu_sub }
array_vvdot_impl! { Complex32; cdotu_sub }

// NOTE: inconsistency in cblas crate API
// FIX NEEDED
// vvdot_impl! { Complex64; zdotu_sub }
// vvdot_impl! { Complex32; cdotu_sub }

macro_rules! mmdot_add_impl {
    ($t:ty; $blas_fn:ident) => {
        impl<Z, M, K, A, D, L, Zrhs0, N, Arhs0, Drhs0, Lrhs0, Arhs1, Drhs1, Lrhs1> DotAdd<&Tensor<Static, Contiguous, Zrhs0, $t, Shape2D<K, N>, Arhs0, Drhs0, Lrhs0>, &Tensor<Static, Contiguous, Normal, $t, Shape2D<M, N>, Arhs1, Drhs1, Lrhs1>> for &Tensor<Static, Contiguous, Z, $t, Shape2D<M, K>, A, D, L>
        where
            Z: BLASTranspose,
            Zrhs0: BLASTranspose,
            M: StaticDim,
            N: StaticDim,
            K: StaticDim,
            Shape2D<M, N>: StaticShape,
            D: Deref<Target=[$t]>,
            Drhs0: Deref<Target=[$t]>,
            Drhs1: DerefMut<Target=[$t]> + Clone,
            L: Layout<U2>,
            Lrhs0: Layout<U2>,
            Lrhs1: Clone,
        {
            type Output = Tensor<Static, Contiguous, Normal, $t, Shape2D<M, N>, Arhs1, Drhs1, Lrhs1>;
            fn dot_add(
                self,
                rhs0: &Tensor<Static, Contiguous, Zrhs0, $t, Shape2D<K, N>, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Static, Contiguous, Normal, $t, Shape2D<M, N>, Arhs1, Drhs1, Lrhs1>,
            ) -> Self::Output {
                let mut out: Self::Output = (*rhs1).clone();

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        Zrhs0::BLAS_TRANSPOSE,
                        M::I32,
                        N::I32,
                        K::I32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        K::I32,
                        rhs0.as_raw_slice(),
                        N::I32,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        N::I32,
                    );
                }

                out
            }
        }

        impl<Z, M, K, A, D, L, Zrhs0, Krhs, N, Arhs0, Drhs0, Lrhs0, Arhs1, Drhs1, Lrhs1> DotAdd<&Tensor<Dynamic, Contiguous, Zrhs0, $t, Shape2D<Krhs, N>, Arhs0, Drhs0, Lrhs0>, &Tensor<Dynamic, Contiguous, Normal, $t, Shape2D<M, N>, Arhs1, Drhs1, Lrhs1>> for &Tensor<Dynamic, Contiguous, Z, $t, Shape2D<M, K>, A, D, L>
        where
            Z: BLASTranspose,
            Zrhs0: BLASTranspose,
            M: Dim,
            N: Dim,
            K: IsEqual<Krhs>,
            Eq<K, Krhs>: TRUE,
            D: Deref<Target=[$t]>,
            Drhs0: Deref<Target=[$t]>,
            Drhs1: DerefMut<Target=[$t]> + Clone,
            L: Layout<U2>,
            Lrhs0: Layout<U2>,
            Lrhs1: Layout<U2> + Clone,
        {
            type Output = Tensor<Dynamic, Contiguous, Normal, $t, Shape2D<M, N>, Arhs1, Drhs1, Lrhs1>;
            fn dot_add(
                self,
                rhs0: &Tensor<Dynamic, Contiguous, Zrhs0, $t, Shape2D<Krhs, N>, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Dynamic, Contiguous, Normal, $t, Shape2D<M, N>, Arhs1, Drhs1, Lrhs1>,
            ) -> Self::Output {
                let self_shape = self.shape();
                let rhs0_shape = rhs0.shape();
                let rhs1_shape = rhs1.shape();
                assert_eq!(
                    self_shape[1], rhs0_shape[0],
                    "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
                    self_shape[1], rhs0_shape[0], self_shape, rhs0_shape,
                );
                assert_eq!(
                    (self_shape[0], rhs0_shape[1]), (rhs0_shape[0], rhs0_shape[1]),
                    "Non-contracted dimmensions {} and {} must be compatible with `rhs1` shape {:?}.",
                    self_shape[0], rhs0_shape[1], rhs1_shape,
                );

                let mut out: Self::Output = (*rhs1).clone();

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        Zrhs0::BLAS_TRANSPOSE,
                        self_shape[0] as i32,
                        rhs0_shape[1] as i32,
                        self_shape[1] as i32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        self_shape[1] as i32,
                        rhs0.as_raw_slice(),
                        rhs0_shape[1] as i32,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        rhs0_shape[1] as i32,
                    );
                }

                out
            }
        }
    };
}

mmdot_add_impl! { f64; dgemm }
mmdot_add_impl! { f32; sgemm }
mmdot_add_impl! { Complex64; zgemm }
mmdot_add_impl! { Complex32; cgemm }

macro_rules! mvdot_add_impl {
    ($t:ty; $blas_fn:ident) => {
        impl<Z, M, N, A, D, L, Zrhs0, Arhs0, Drhs0, Lrhs0, Arhs1, Drhs1, Lrhs1>
            DotAdd<
                &Tensor<Static, Contiguous, Zrhs0, $t, Shape1D<N>, Arhs0, Drhs0, Lrhs0>,
                &Tensor<Static, Contiguous, Normal, $t, Shape1D<M>, Arhs1, Drhs1, Lrhs1>,
            > for &Tensor<Static, Contiguous, Z, $t, Shape2D<M, N>, A, D, L>
        where
            Z: BLASTranspose,
            M: StaticDim,
            N: StaticDim,
            Shape1D<M>: StaticShape,
            D: Deref<Target = [$t]>,
            Drhs0: Deref<Target = [$t]>,
            Drhs1: DerefMut<Target = [$t]> + Clone,
            L: Layout<U2>,
            Lrhs0: Layout<U1>,
            Lrhs1: Clone,
        {
            type Output = Tensor<Static, Contiguous, Normal, $t, Shape1D<M>, Arhs1, Drhs1, Lrhs1>;

            fn dot_add(
                self,
                rhs0: &Tensor<Static, Contiguous, Zrhs0, $t, Shape1D<N>, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Static, Contiguous, Normal, $t, Shape1D<M>, Arhs1, Drhs1, Lrhs1>,
            ) -> Self::Output {
                let mut out: Self::Output = (*rhs1).clone();

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        M::I32,
                        N::I32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        N::I32,
                        rhs0.as_raw_slice(),
                        1,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        1,
                    );
                }

                out
            }
        }

        impl<Z, M, N, A, D, L, Zrhs0, Nrhs, Arhs0, Drhs0, Lrhs0, Arhs1, Drhs1, Lrhs1>
            DotAdd<
                &Tensor<Dynamic, Contiguous, Zrhs0, $t, Shape1D<Nrhs>, Arhs0, Drhs0, Lrhs0>,
                &Tensor<Dynamic, Contiguous, Normal, $t, Shape1D<M>, Arhs1, Drhs1, Lrhs1>,
            > for &Tensor<Dynamic, Contiguous, Z, $t, Shape2D<M, N>, A, D, L>
        where
            Z: BLASTranspose,
            M: Dim,
            N: IsEqual<Nrhs>,
            Eq<N, Nrhs>: TRUE,
            D: Deref<Target = [$t]>,
            Drhs0: Deref<Target = [$t]>,
            Drhs1: DerefMut<Target = [$t]> + Clone,
            L: Layout<U2>,
            Lrhs0: Layout<U1>,
            Lrhs1: Layout<U1> + Clone,
        {
            type Output = Tensor<Dynamic, Contiguous, Normal, $t, Shape1D<M>, Arhs1, Drhs1, Lrhs1>;

            fn dot_add(
                self,
                rhs0: &Tensor<Dynamic, Contiguous, Zrhs0, $t, Shape1D<Nrhs>, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Dynamic, Contiguous, Normal, $t, Shape1D<M>, Arhs1, Drhs1, Lrhs1>,
            ) -> Self::Output {
                let self_shape = self.shape();
                let rhs0_shape = rhs0.shape();
                let rhs1_shape = rhs1.shape();
                assert_eq!(
                    self_shape[1], rhs0_shape[0],
                    "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
                    self_shape[1], rhs0_shape[0], self_shape, rhs0_shape,
                );
                assert_eq!(
                    self_shape[0], rhs1_shape[0],
                    "Non-contracted dimmension {} must be compatible with `rhs1` shape {:?}.",
                    self_shape[0], rhs1_shape,
                );

                let mut out: Self::Output = (*rhs1).clone();

                unsafe {
                    $blas_fn(
                        cblas::Layout::RowMajor,
                        Z::BLAS_TRANSPOSE,
                        self_shape[0] as i32,
                        self_shape[1] as i32,
                        <$t>::ONE,
                        self.as_raw_slice(),
                        self_shape[1] as i32,
                        rhs0.as_raw_slice(),
                        1,
                        <$t>::ONE,
                        out.as_raw_slice_mut(),
                        1,
                    );
                }

                out
            }
        }
    };
}

mvdot_add_impl! { f64; dgemv }
mvdot_add_impl! { f32; sgemv }
mvdot_add_impl! { Complex64; zgemv }
mvdot_add_impl! { Complex32; cgemv }
