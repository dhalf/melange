//! Contains basic mathematical operations at the tensor level.
//!
//! All operations are wrapped in their own trait which eases reuse.
//! See the [`ops`](crate::ops) module.
//!
//! There are two families of operations: functionnal operations
//! and in-place operations.
//!
//! Functionnal operations are methods that immutably borrow `self` (and
//! optionnaly borrow other tensors) and return a new tensor. They are
//! actually defined with traits that move their inputs to be compatible
//! with traits in [`std::ops`](`std::ops`). The trick is to implement
//! those traits for references.
//!
//! Conversely, in-place operations mutably borrow `self` (and optionnaly
//! immutably borrow other tensors) to directly mutate its data. The actual
//! implementation, in this case, does borrow `self` mutably but moves other
//! parameters. Once again the trick is to implement for references.
//!
//! Functionnal operations are interesting when backpropagating because they
//! preserve operands whereas in-place operations reduce the memory footprint.
//!
//! Both families of operations perform ad-hoc parallel computation
//! acording to how data is stored. Note that operations on more than one
//! tensor require all tensors to have compatible shapes (all dimensions
//! must be equal or `Dyn`). If this is not the case consider broadcasting.
//!
//! Note that functionnal operations actually use in-place operations
//! under the hood.
//!
//! Note that some operations are only available for tensors based on float
//! types `f32` and `f64` or primitive integers.
//! This is inherent to how numeric types are treated in rust.
//!
//! Those operations rely on the scalar operations implemented for the underlying
//! scalar data type `T` of the tensor via the implementation of ops traits on
//! scalars in the [`ops`](crate::ops) module.
//! Please refer to the relevant methods defined on primitive types for further
//! details.

use super::layout::Layout;
use super::shape::{Same, Shape, TRUE};
use super::strided_iterator::{StridedIterator, StridedIteratorMut};
use crate::gat::{RefMutGat, StreamingIterator};
use crate::ops::*;
use crate::tensor::alloc::{AllocLike, AllocSameShape};
use crate::tensor::{Dynamic, Static, Tensor};
use num_complex::{Complex32, Complex64};
use std::ops::*;
use crate::scalar_traits::Cast;

macro_rules! assert_shape_eq {
    ($lhs:expr, $rhs:expr) => {
        assert_eq!(
            $lhs,
            $rhs,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            $lhs,
            $rhs
        );
    };
}

// --------------------
// Binary inplace ops
// --------------------

macro_rules! binary_inplace_op_impls {
    (
        $($trait:ident, $trait_fn:ident);*
    ) => {$(
        macro_rules! op_unchecked {
            ($self:ident $rhs:ident) => {
                let chunk_size = $self.opt_chunk_size().min($rhs.opt_chunk_size());

                let mut it = $self.strided_iter_mut(chunk_size).streaming_zip($rhs.strided_iter(chunk_size));
                while let Some((self_chunk, rhs_chunk)) = it.next() {
                    for (x, y) in self_chunk.iter_mut().zip(rhs_chunk.iter()) {
                        x.$trait_fn(*y);
                    }
                }
            };
        }

        impl<Y, Z, T, S, A, D, L, Yrhs, Zrhs, Arhs, Drhs, Lrhs> $trait<&Tensor<Static, Yrhs, Zrhs, T, S, Arhs, Drhs, Lrhs>> for Tensor<Static, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            for<'a> &'a Tensor<Static, Yrhs, Zrhs, T, S, Arhs, Drhs, Lrhs>: StridedIterator<Item=&'a [T]>,
            T: $trait + Copy + 'static,
            S: Shape,
            L: Layout<S::Len>,
            Lrhs: Layout<S::Len>,
        {
            fn $trait_fn(&mut self, rhs: &Tensor<Static, Yrhs, Zrhs, T, S, Arhs, Drhs, Lrhs>) {
                op_unchecked! { self rhs }
            }
        }

        impl<Y, Z, T, S, A, D, L, Yrhs, Zrhs, Srhs, Arhs, Drhs, Lrhs> $trait<&Tensor<Dynamic, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>> for Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            for<'a> &'a Tensor<Dynamic, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>: StridedIterator<Item=&'a [T]>,
            T: $trait + Copy + 'static,
            S: Shape + Same<Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            Srhs: Shape,
            L: Layout<S::Len>,
            Lrhs: Layout<Srhs::Len>,
        {
            fn $trait_fn(&mut self, rhs: &Tensor<Dynamic, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>) {
                assert_shape_eq!(self.shape().deref(), rhs.shape().deref());
                op_unchecked! { self rhs }
            }
        }
    )*};
}

binary_inplace_op_impls! {
    AddAssign, add_assign;
    SubAssign, sub_assign;
    MulAssign, mul_assign;
    DivAssign, div_assign;
    RemAssign, rem_assign;
    MaxAssign, max_assign;
    MinAssign, min_assign;
    ArgmaxAssign, argmax_assign;
    ArgminAssign, argmin_assign;
    Atan2Assign, atan2_assign;
    CopysignAssign, copysign_assign;
    DivEuclidAssign, div_euclid_assign;
    RemEuclidAssign, rem_euclid_assign
}

// ---------------------------
// one-parameter inplace ops
// ---------------------------

macro_rules! one_param_inplace_op_impls {
    (
        $($trait:ident$(<$param_type:ty>)?, $trait_fn:ident);*
    ) => {$(
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { T };
        }

        impl<X, Y, Z, T, S, A, D, L> $trait<isset_or_default!($($param_type)?)> for Tensor<X, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            T: $trait<isset_or_default!($($param_type)?)> + Copy + 'static,
            S: Shape,
            L: Layout<S::Len>,
        {
            fn $trait_fn(&mut self, rhs: isset_or_default!($($param_type)?)) {
                let chunk_size = self.opt_chunk_size();

                let mut it = self.strided_iter_mut(chunk_size);
                while let Some(chunk) = it.next() {
                    for x in chunk.iter_mut() {
                        x.$trait_fn(rhs);
                    }
                }
            }
        }
    )*};
}

one_param_inplace_op_impls! {
    AddAssign, add_assign;
    SubAssign, sub_assign;
    MulAssign, mul_assign;
    DivAssign, div_assign;
    RemAssign, rem_assign;
    MaxAssign, max_assign;
    MinAssign, min_assign;
    ArgmaxAssign, argmax_assign;
    ArgminAssign, argmin_assign;
    Atan2Assign, atan2_assign;
    CopysignAssign, copysign_assign;
    DivEuclidAssign, div_euclid_assign;
    RemEuclidAssign, rem_euclid_assign;
    PowAssign<i32>, pow_assign;
    PowAssign<u32>, pow_assign;
    PowAssign<f32>, pow_assign;
    PowAssign<f64>, pow_assign;
    PowAssign<Complex32>, pow_assign;
    PowAssign<Complex64>, pow_assign
}

// ------------
// inplace fn
// ------------

macro_rules! inplace_fn_impls {
    (
        $($trait:ident, $trait_fn:ident);*
    ) => {$(
        impl<X, Y, Z, T, S, A, D, L> $trait for Tensor<X, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            T: $trait + Copy + 'static,
            S: Shape,
            L: Layout<S::Len>,
        {
            fn $trait_fn(&mut self) {
                let chunk_size = self.opt_chunk_size();

                let mut it = self.strided_iter_mut(chunk_size);
                while let Some(chunk) = it.next() {
                    for x in chunk.iter_mut() {
                        x.$trait_fn();
                    }
                }
            }
        }
    )*};
}

inplace_fn_impls! {
    ExpAssign, exp_assign;
    Exp2Assign, exp2_assign;
    ExpM1Assign, exp_m1_assign;
    LnAssign, ln_assign;
    Ln1pAssign, ln_1p_assign;
    Log2Assign, log2_assign;
    Log10Assign, log10_assign;
    SinAssign, sin_assign;
    CosAssign, cos_assign;
    TanAssign, tan_assign;
    SinhAssign, sinh_assign;
    CoshAssign, cosh_assign;
    TanhAssign, tanh_assign;
    AsinAssign, asin_assign;
    AcosAssign, acos_assign;
    AtanAssign, atan_assign;
    AsinhAssign, asinh_assign;
    AcoshAssign, acosh_assign;
    AtanhAssign, atanh_assign;
    SqrtAssign, sqrt_assign;
    CbrtAssign, cbrt_assign;
    AbsAssign, abs_assign;
    SignumAssign, signum_assign;
    CeilAssign, ceil_assign;
    FloorAssign, floor_assign;
    RoundAssign, round_assign;
    RecipAssign, recip_assign;
    ToDegreesAssign, to_degrees_assign;
    ToRadiansAssign, to_radians_assign;
    ConjAssign, conj_assign;
    ZeroOut, zero_out
}

// ---------------------------
// two-parameter inplace ops
// ---------------------------

macro_rules! two_param_inplace_op_impls {
    (
        $($trait:ident$(<$param_type0:ty, $param_type1:ty>)?, $trait_fn:ident);*
    ) => {$(
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { T };
        }
        impl<X, Y, Z, T, S, A, D, L> $trait<isset_or_default!($($param_type0)?), isset_or_default!($($param_type1)?)> for Tensor<X, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            T: $trait<isset_or_default!($($param_type0)?), isset_or_default!($($param_type1)?)> + Copy + 'static,
            S: Shape,
            L: Layout<S::Len>,
        {
            fn $trait_fn(&mut self, rhs0: isset_or_default!($($param_type0)?), rhs1: isset_or_default!($($param_type1)?)) {
                let chunk_size = self.opt_chunk_size();

                let mut it = self.strided_iter_mut(chunk_size);
                while let Some(chunk) = it.next() {
                    for x in chunk.iter_mut() {
                        x.$trait_fn(rhs0, rhs1);
                    }
                }
            }
        }
    )*};
}

two_param_inplace_op_impls! {
    MulAddAssign, mul_add_assign
}

// ---------------------
// ternary inplace ops
// ---------------------

macro_rules! ternary_inplace_op_impls {
    (
        $($trait:ident, $trait_fn:ident);*
    ) => {$(
        macro_rules! op_unchecked {
            ($self:ident $rhs0:ident $rhs1:ident) => {
                let chunk_size = $self.opt_chunk_size().min($rhs0.opt_chunk_size().min($rhs1.opt_chunk_size()));

                let mut it = $self.strided_iter_mut(chunk_size).streaming_zip($rhs0.strided_iter(chunk_size)).streaming_zip($rhs1.strided_iter(chunk_size));
                while let Some(((self_chunk, rhs0_chunk), rhs1_chunk)) = it.next() {
                    for ((x, y0), y1) in self_chunk.iter_mut().zip(rhs0_chunk.iter()).zip(rhs1_chunk.iter()) {
                        x.$trait_fn(*y0, *y1);
                    }
                }
            };
        }

        impl<Y, Z, T, S, A, D, L, Yrhs0, Zrhs0, Arhs0, Drhs0, Lrhs0, Yrhs1, Zrhs1, Arhs1, Drhs1, Lrhs1> $trait<&Tensor<Static, Yrhs0, Zrhs0, T, S, Arhs0, Drhs0, Lrhs0>, &Tensor<Static, Yrhs1, Zrhs1, T, S, Arhs1, Drhs1, Lrhs1>> for Tensor<Static, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            for<'a> &'a Tensor<Static, Yrhs0, Zrhs0, T, S, Arhs0, Drhs0, Lrhs0>: StridedIterator<Item=&'a [T]>,
            for<'a> &'a Tensor<Static, Yrhs1, Zrhs1, T, S, Arhs1, Drhs1, Lrhs1>: StridedIterator<Item=&'a [T]>,
            T: $trait + Copy + 'static,
            S: Shape,
            L: Layout<S::Len>,
            Lrhs0: Layout<S::Len>,
            Lrhs1: Layout<S::Len>,
        {
            fn $trait_fn(
                &mut self,
                rhs0: &Tensor<Static, Yrhs0, Zrhs0, T, S, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Static, Yrhs1, Zrhs1, T, S, Arhs1, Drhs1, Lrhs1>,
            ) {
                op_unchecked! { self rhs0 rhs1 }
            }
        }

        impl<Y, Z, T, S, A, D, L, Yrhs0, Zrhs0, Srhs0, Arhs0, Drhs0, Lrhs0, Yrhs1, Zrhs1, Arhs1, Srhs1, Drhs1, Lrhs1> $trait<&Tensor<Dynamic, Yrhs0, Zrhs0, T, Srhs0, Arhs0, Drhs0, Lrhs0>, &Tensor<Dynamic, Yrhs1, Zrhs1, T, Srhs1, Arhs1, Drhs1, Lrhs1>> for Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[T]>>,
            for<'a> &'a Tensor<Dynamic, Yrhs0, Zrhs0, T, Srhs0, Arhs0, Drhs0, Lrhs0>: StridedIterator<Item=&'a [T]>,
            for<'a> &'a Tensor<Dynamic, Yrhs1, Zrhs1, T, Srhs1, Arhs1, Drhs1, Lrhs1>: StridedIterator<Item=&'a [T]>,
            T: $trait + Copy + 'static,
            S: Shape + Same<Srhs0> + Same<Srhs1>,
            <S as Same<Srhs0>>::Output: TRUE,
            <S as Same<Srhs1>>::Output: TRUE,
            Srhs0: Shape,
            Srhs1: Shape,
            L: Layout<S::Len>,
            Lrhs0: Layout<Srhs0::Len>,
            Lrhs1: Layout<Srhs1::Len>,
        {
            fn $trait_fn(
                &mut self,
                rhs0: &Tensor<Dynamic, Yrhs0, Zrhs0, T, Srhs0, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Dynamic, Yrhs1, Zrhs1, T, Srhs1, Arhs1, Drhs1, Lrhs1>,
            ) {
                assert_shape_eq!(self.shape().deref(), rhs0.shape().deref());
                assert_shape_eq!(self.shape().deref(), rhs1.shape().deref());
                op_unchecked! { self rhs0 rhs1 }
            }
        }
    )*};
}

ternary_inplace_op_impls! {
    MulAddAssign, mul_add_assign
}

// -------------------
// Functional traits
// -------------------

macro_rules! op_impls {
    (
        $($trait:ident, $trait_fn:ident, $inplace_trait:ident, $inplace_trait_fn:ident);*
    ) => {$(
        impl<'a, X, Y, Z, T, S, A, D, L, Rhs> $trait<Rhs> for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Tensor<X, Y, Z, T, S, A, D, L>: AllocLike,
            <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc: $inplace_trait<Rhs>,
        {
            type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc;
            fn $trait_fn(
                self,
                rhs: Rhs,
            ) -> Self::Output {
                let mut out = self.to_contiguous();
                out.$inplace_trait_fn(rhs);

                out
            }
        }
    )*};
}

op_impls! {
    Add, add, AddAssign, add_assign;
    Sub, sub, SubAssign, sub_assign;
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
    Rem, rem, RemAssign, rem_assign;
    Atan2, atan2, Atan2Assign, atan2_assign;
    Copysign, copysign, CopysignAssign, copysign_assign;
    DivEuclid, div_euclid, DivEuclidAssign, div_euclid_assign;
    Max, max, MaxAssign, max_assign;
    Min, min, MinAssign, min_assign;
    Argmax, argmax, ArgmaxAssign, argmax_assign;
    Argmin, argmin, ArgminAssign, argmin_assign;
    RemEuclid, rem_euclid, RemEuclidAssign, rem_euclid_assign;
    Pow, pow, PowAssign, pow_assign
}

macro_rules! fn_impls {
    (
        $($trait:ident, $trait_fn:ident, $inplace_trait:ident, $inplace_trait_fn:ident);*
    ) => {$(
        impl<'a, X, Y, Z, T, S, A, D, L> $trait for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Tensor<X, Y, Z, T, S, A, D, L>: AllocLike,
            <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc: $inplace_trait,
        {
            type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc;
            fn $trait_fn(self) -> Self::Output {
                let mut out = self.to_contiguous();
                out.$inplace_trait_fn();

                out
            }
        }
    )*};
}

fn_impls! {
    Exp, exp, ExpAssign, exp_assign;
    Exp2, exp2, Exp2Assign, exp2_assign;
    ExpM1, exp_m1, ExpM1Assign, exp_m1_assign;
    Ln, ln, LnAssign, ln_assign;
    Ln1p, ln_1p, Ln1pAssign, ln_1p_assign;
    Log2, log2, Log2Assign, log2_assign;
    Log10, log10, Log10Assign, log10_assign;
    Sin, sin, SinAssign, sin_assign;
    Cos, cos, CosAssign, cos_assign;
    Tan, tan, TanAssign, tan_assign;
    Sinh, sinh, SinhAssign, sinh_assign;
    Cosh, cosh, CoshAssign, cosh_assign;
    Tanh, tanh, TanhAssign, tanh_assign;
    Asin, asin, AsinAssign, asin_assign;
    Acos, acos, AcosAssign, acos_assign;
    Atan, atan, AtanAssign, atan_assign;
    Asinh, asinh, AsinhAssign, asinh_assign;
    Acosh, acosh, AcoshAssign, acosh_assign;
    Atanh, atanh, AtanhAssign, atanh_assign;
    Sqrt, sqrt, SqrtAssign, sqrt_assign;
    Cbrt, cbrt, CbrtAssign, cbrt_assign;
    Abs, abs, AbsAssign, abs_assign;
    Signum, signum, SignumAssign, signum_assign;
    Ceil, ceil, CeilAssign, ceil_assign;
    Floor, floor, FloorAssign, floor_assign;
    Round, round, RoundAssign, round_assign;
    Recip, recip, RecipAssign, recip_assign;
    ToDegrees, to_degrees, ToDegreesAssign, to_degrees_assign;
    ToRadians, to_radians, ToRadiansAssign, to_radians_assign;
    Conj, conj, ConjAssign, conj_assign
}

macro_rules! ternary_op_impls {
    (
        $($trait:ident, $trait_fn:ident, $inplace_trait:ident, $inplace_trait_fn:ident);*
    ) => {$(
        impl<'a, X, Y, Z, T, S, A, D, L, Rhs0, Rhs1> $trait<Rhs0, Rhs1> for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Tensor<X, Y, Z, T, S, A, D, L>: AllocLike,
            <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc: $inplace_trait<Rhs0, Rhs1>,
    {
        type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc;
            fn $trait_fn(
                self,
                rhs0: Rhs0,
                rhs1: Rhs1,
            ) -> Self::Output {
                let mut out = self.to_contiguous();
                out.$inplace_trait_fn(rhs0, rhs1);

                out
            }
        }
    )*};
}

ternary_op_impls! {
    MulAdd, mul_add, MulAddAssign, mul_add_assign
}

// ----------------
// Conversion ops
// ----------------

macro_rules! conversion_op_impls {
    (
        $($trait:ident, $trait_fn:ident);*
    ) => {$(
        impl<'a, X, Y, Z, T, S, A, D, L> $trait for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            T: $trait + Copy + 'static,
            <T as $trait>::Output: Default,
            S: Shape,
            L: Layout<S::Len>,
            Tensor<X, Y, Z, T, S, A, D, L>: AllocSameShape<<T as $trait>::Output>,
            for<'b> &'b mut <Tensor<X, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait>::Output>>::Alloc: StridedIteratorMut<Item=RefMutGat<[<T as $trait>::Output]>>,
            Self: StridedIterator<Item=&'a [T]>,
        {
            type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait>::Output>>::Alloc;
            fn $trait_fn(self) -> Self::Output {
                let mut out = self.fill_same_shape(<<T as $trait>::Output>::default());
                let chunk_size = self.opt_chunk_size();

                {
                    let mut it = out.strided_iter_mut(chunk_size).streaming_zip(self.strided_iter(chunk_size));
                    while let Some((out_chunk, self_chunk)) = it.next() {
                        for (x, y) in out_chunk.iter_mut().zip(self_chunk.iter()) {
                            *x = y.$trait_fn();
                        }
                    }
                }

                out
            }
        }
    )*};
}

conversion_op_impls! {
    Re, re;
    Im, im;
    Norm, norm;
    NormSqr, norm_sqr;
    Arg, arg;
    J, j;
    EPowJ, e_pow_j
}

macro_rules! binary_conversion_op_impls {
    (
        $($trait:ident, $trait_fn:ident);*
    ) => {$(
        macro_rules! op_unchecked {
            ($self:ident $rhs:ident) => {
                let mut out = $self.fill_same_shape(<<T as $trait>::Output>::default());
                let chunk_size = $self.opt_chunk_size().min($rhs.opt_chunk_size());

                {
                    let mut it = out.strided_iter_mut(chunk_size).streaming_zip($self.strided_iter(chunk_size)).streaming_zip($rhs.strided_iter(chunk_size));
                    while let Some(((out_chunk, self_chunk), rhs_chunk)) = it.next() {
                        for ((x, y), z) in out_chunk.iter_mut().zip(self_chunk.iter()).zip(rhs_chunk.iter()) {
                            *x = (*y).$trait_fn(*z);
                        }
                    }
                }

                out
            };
        }

        impl<'a, Y, Z, T, S, A, D, L, Yrhs, Zrhs, Arhs, Drhs, Lrhs> $trait<&Tensor<Static, Yrhs, Zrhs, T, S, Arhs, Drhs, Lrhs>> for &'a Tensor<Static, Y, Z, T, S, A, D, L>
        where
            Self: StridedIterator<Item=&'a [T]>,
            for<'b> &'b Tensor<Static, Yrhs, Zrhs, T, S, Arhs, Drhs, Lrhs>: StridedIterator<Item=&'b [T]>,
            Tensor<Static, Y, Z, T, S, A, D, L>: AllocSameShape<<T as $trait>::Output>,
            for<'b> &'b mut <Tensor<Static, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait>::Output>>::Alloc: StridedIteratorMut<Item=RefMutGat<[<T as $trait>::Output]>>,
            T: $trait + Copy + 'static,
            <T as $trait>::Output: Default,
            S: Shape,
            L: Layout<S::Len>,
            Lrhs: Layout<S::Len>,
        {
            type Output = <Tensor<Static, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait>::Output>>::Alloc;
            fn $trait_fn(self, rhs: &Tensor<Static, Yrhs, Zrhs, T, S, Arhs, Drhs, Lrhs>) -> Self::Output {
                op_unchecked! { self rhs }
            }
        }

        impl<'a, Y, Z, T, S, A, D, L, Yrhs, Zrhs, Srhs, Arhs, Drhs, Lrhs> $trait<&Tensor<Dynamic, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>> for &'a Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            Self: StridedIterator<Item=&'a [T]>,
            for<'b> &'b Tensor<Dynamic, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>: StridedIterator<Item=&'b [T]>,
            Tensor<Dynamic, Y, Z, T, S, A, D, L>: AllocSameShape<<T as $trait>::Output>,
            for<'b> &'b mut <Tensor<Dynamic, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait>::Output>>::Alloc: StridedIteratorMut<Item=RefMutGat<[<T as $trait>::Output]>>,
            T: $trait + Copy + 'static,
            <T as $trait>::Output: Default,
            S: Shape + Same<Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            Srhs: Shape,
            L: Layout<S::Len>,
            Lrhs: Layout<Srhs::Len>,
        {
            type Output = <Tensor<Dynamic, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait>::Output>>::Alloc;
            fn $trait_fn(self, rhs: &Tensor<Dynamic, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>) -> Self::Output {
                assert_shape_eq!(self.shape().deref(), rhs.shape().deref());
                op_unchecked! { self rhs }
            }
        }
    )*};
}

binary_conversion_op_impls! {
    AddJ, add_j;
    MulEPowJ, mul_e_pow_j
}

macro_rules! one_param_conversion_op_impls {
    (
        $($trait:ident$(<$param_type:ty>)?, $trait_fn:ident);*
    ) => {$(
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { T };
        }

        impl<'a, X, Y, Z, T, S, A, D, L> $trait<isset_or_default!($($param_type)?)> for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Self: StridedIterator<Item=&'a [T]>,
            Tensor<X, Y, Z, T, S, A, D, L>: AllocSameShape<<T as $trait<isset_or_default!($($param_type)?)>>::Output>,
            for<'b> &'b mut <Tensor<X, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait<isset_or_default!($($param_type)?)>>::Output>>::Alloc: StridedIteratorMut<Item=RefMutGat<[<T as $trait>::Output]>>,
            T: $trait<isset_or_default!($($param_type)?)> + Copy + 'static,
            <T as $trait<isset_or_default!($($param_type)?)>>::Output: Default,
            S: Shape,
            L: Layout<S::Len>,
        {
            type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocSameShape<<T as $trait<isset_or_default!($($param_type)?)>>::Output>>::Alloc;
            fn $trait_fn(self, rhs: isset_or_default!($($param_type)?)) -> Self::Output {
                let mut out = self.fill_same_shape(<<T as $trait>::Output>::default());
                let chunk_size = self.opt_chunk_size();

                {
                    let mut it = out.strided_iter_mut(chunk_size).streaming_zip(self.strided_iter(chunk_size));
                    while let Some((out_chunk, self_chunk)) = it.next() {
                        for (x, y) in out_chunk.iter_mut().zip(self_chunk.iter()) {
                            *x = (*y).$trait_fn(rhs);
                        }
                    }
                }

                out
            }
        }
    )*};
}

one_param_conversion_op_impls! {
    AddJ, add_j;
    MulEPowJ, mul_e_pow_j
}

// -------
// Casts
// -------

/// Elementwise cast of a tensor.
/// 
/// Cast all its elements into `U`.
pub trait TensorCast<U> {
    /// Output type.
    type Output;
    /// Cast all elements into `U`.
    fn as_(self) -> Self::Output;
}

impl<'a, X, Y, Z, T, S, A, D, L, U> TensorCast<U> for &'a Tensor<X, Y, Z, T, S, A, D, L>
where
    T: Cast<U> + Copy,
    U: Default + 'static,
    S: Shape,
    L: Layout<S::Len>,
    Tensor<X, Y, Z, T, S, A, D, L>: AllocSameShape<U>,
    for<'b> &'b mut <Tensor<X, Y, Z, T, S, A, D, L> as AllocSameShape<U>>::Alloc: StridedIteratorMut<Item=RefMutGat<[U]>>,
    Self: StridedIterator<Item=&'a [T]>,
{
    type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocSameShape<U>>::Alloc;
    fn as_(self) -> Self::Output {
        let mut out = self.fill_same_shape(U::default());
        let chunk_size = self.opt_chunk_size();

        {
            let mut it = out.strided_iter_mut(chunk_size).streaming_zip(self.strided_iter(chunk_size));
            while let Some((out_chunk, self_chunk)) = it.next() {
                for (x, y) in out_chunk.iter_mut().zip(self_chunk.iter()) {
                    *x = y.as_();
                }
            }
        }

        out
    }
}
