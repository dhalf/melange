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

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Rem, RemAssign, Neg};
use num_complex::{Complex32, Complex64, Complex};
use crate::scalar_traits::Cast;
use crate::ops::*;
use super::*;

// A: Any tensor
// B: Owned tensor
// C: Mutable tensor
// K: any scalar type

// ----------------------------------------
// unary ops: C (-> K | &K)* -> inplace ()
// ----------------------------------------

macro_rules! args_call {
    ($e:ty, $e2:ty; $x:ident.$fn:ident($rhs0:ident, $rhs1:ident)) => { $x.$fn($rhs0, $rhs1) };
    ($e:ty; $x:ident.$fn:ident($rhs0:ident, $rhs1:ident)) => { $x.$fn($rhs0) };
    (; $x:ident.$fn:ident($rhs0:ident, $rhs1:ident)) => { $x.$fn() };
}

macro_rules! inplace_unary_op_impl_tensor {
    ($($trait:ident$(<$t:ty$(, $s:ty)?>)? $fn:ident $(for<$($gens:ident),*>)? $(for<$($ls:lifetime),*>)?);*) => {$(
        impl<$($($ls,)*)?B, T, S, C$($(, $gens)*)?> $trait$(<$t$(, $s)?>)? for Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem>,
            B::Applied: AsMut<[T]>,
            S: Axes,
            T: $trait$(<$t$(, $s)?>)? + Copy + 'static,
            $($($gens: Copy,)*)?
        {
            fn $fn(&mut self$(, rhs0: $t$(, rhs1: $s)?)?) {
                self.for_each(|x| args_call!($($t$(, $s)?)?; x.$fn(rhs0, rhs1)));
            }
        }
    )*};
}

inplace_unary_op_impl_tensor! {
    AddAssign<T> add_assign;
    SubAssign<T> sub_assign;
    MulAssign<T> mul_assign;
    DivAssign<T> div_assign;
    RemAssign<T> rem_assign;
    Atan2Assign<T> atan2_assign;
    HypotAssign<T> hypot_assign;
    CopysignAssign<T> copysign_assign;
    DivEuclidAssign<T> div_euclid_assign;
    RemEuclidAssign<T> rem_euclid_assign;
    MaxAssign<T> max_assign;
    MinAssign<T> min_assign;
    MaxMaskAssign<T> max_mask_assign;
    MinMaskAssign<T> min_mask_assign;
    AddAssign<&'a T> add_assign for<'a>;
    SubAssign<&'a T> sub_assign for<'a>;
    MulAssign<&'a T> mul_assign for<'a>;
    DivAssign<&'a T> div_assign for<'a>;
    RemAssign<&'a T> rem_assign for<'a>;
    Atan2Assign<&'a T> atan2_assign for<'a>;
    HypotAssign<&'a T> hypot_assign for<'a>;
    CopysignAssign<&'a T> copysign_assign for<'a>;
    DivEuclidAssign<&'a T> div_euclid_assign for<'a>;
    RemEuclidAssign<&'a T> rem_euclid_assign for<'a>;
    MaxAssign<&'a T> max_assign for<'a>;
    MinAssign<&'a T> min_assign for<'a>;
    MaxMaskAssign<&'a T> max_mask_assign for<'a>;
    MinMaskAssign<&'a T> min_mask_assign for<'a>;
    PowAssign<U> pow_assign for<U>;
    LogAssign<U> log_assign for<U>;
    ExpAssign exp_assign;
    Exp2Assign exp2_assign;
    ExpM1Assign exp_m1_assign;
    LnAssign ln_assign;
    Ln1pAssign ln_1p_assign;
    Log2Assign log2_assign;
    Log10Assign log10_assign;
    SinAssign sin_assign;
    CosAssign cos_assign;
    TanAssign tan_assign;
    SinhAssign sinh_assign;
    CoshAssign cosh_assign;
    TanhAssign tanh_assign;
    AsinAssign asin_assign;
    AcosAssign acos_assign;
    AtanAssign atan_assign;
    AsinhAssign asinh_assign;
    AcoshAssign acosh_assign;
    AtanhAssign atanh_assign;
    SqrtAssign sqrt_assign;
    CbrtAssign cbrt_assign;
    AbsAssign abs_assign;
    SignumAssign signum_assign;
    CeilAssign ceil_assign;
    FloorAssign floor_assign;
    RoundAssign round_assign;
    TruncAssign trunc_assign;
    FractAssign fract_assign;
    RecipAssign recip_assign;
    ToDegreesAssign to_degrees_assign;
    ToRadiansAssign to_radians_assign;
    ConjAssign conj_assign;
    ZeroOut zero_out;
    MulAddAssign<T, T> mul_add_assign
}

// ---------------------------------
// binary ops C -> &A -> inplace ()
//            C -> A -> inplace ()
// ---------------------------------

macro_rules! inplace_binary_op_impl_tensor {
    ($($trait:ident $fn:ident);*) => {$(
        impl<B, B2, T, S, C, C2> $trait<&Tensor<B2, T, S, C2>> for Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem>,
            B2: KindTypeTypeType<T, S::Elem>,
            B::Applied: AsMut<[T]>,
            B2::Applied: AsRef<[T]>,
            S: Axes,
            T: $trait + Copy + 'static,
        {
            fn $fn(&mut self, rhs: &Tensor<B2, T, S, C2>) {
                self.zip_with_mut(rhs, |x, &y| x.$fn(y));
            }
        }
        impl<B, B2, T, S, C, C2> $trait<Tensor<B2, T, S, C2>> for Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem>,
            B2: KindTypeTypeType<T, S::Elem>,
            B::Applied: AsMut<[T]>,
            B2::Applied: AsRef<[T]>,
            S: Axes,
            T: $trait + Copy + 'static,
        {
            fn $fn(&mut self, rhs: Tensor<B2, T, S, C2>) {
                self.zip_with_mut(&rhs, |x, &y| x.$fn(y));
            }
        }
    )*};
}

inplace_binary_op_impl_tensor! {
    AddAssign add_assign;
    SubAssign sub_assign;
    MulAssign mul_assign;
    DivAssign div_assign;
    RemAssign rem_assign;
    Atan2Assign atan2_assign;
    HypotAssign hypot_assign;
    CopysignAssign copysign_assign;
    DivEuclidAssign div_euclid_assign;
    RemEuclidAssign rem_euclid_assign;
    MaxAssign max_assign;
    MinAssign min_assign;
    MaxMaskAssign max_mask_assign;
    MinMaskAssign min_mask_assign
}

// -----------------------------------------
// ternary ops: C -> &A -> &A -> inplace ()
// -----------------------------------------

impl<B, B2, B3, T, S, C, C2, C3> MulAddAssign<&Tensor<B2, T, S, C2>, &Tensor<B3, T, S, C3>> for Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem>,
    B2: KindTypeTypeType<T, S::Elem>,
    B3: KindTypeTypeType<T, S::Elem>,
    B::Applied: AsMut<[T]>,
    B2::Applied: AsRef<[T]>,
    B3::Applied: AsRef<[T]>,
    S: Axes,
    T: MulAddAssign + Copy + 'static,
{
    fn mul_add_assign(&mut self, rhs0: &Tensor<B2, T, S, C2>, rhs1: &Tensor<B3, T, S, C3>) {
        self.zip2_with_mut(rhs0, rhs1, |x, &y, &z| x.mul_add_assign(y, z));
    }
}

// --------------------------------
// unary ops: &A (-> K | &K)* -> B
// --------------------------------

macro_rules! functional_unary_op_impl_ref_tensor {
    ($($trait:ident$(<$t:ty$(, $s:ty)?>)? $fn:ident $(for<$($gens:ident),*>)? $(for<$($ls:lifetime),*>)?);*) => {$(
        impl<$($($ls,)*)?B, T, S, C$($(, $gens)*)?> $trait$(<$t$(, $s)?>)? for &Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem> + Realloc<T, S::Elem>,
            B::Applied: AsRef<[T]>,
            <B::Buffer as KindTypeTypeType<T, S::Elem>>::Applied: AsMut<[T]>,
            S: Axes,
            T: $trait<$($t$(, $s)?,)? Output = T> + Default + Copy + 'static,
            $($($gens: Copy,)*)?
        {
            type Output = Tensor<B::Buffer, T, S, Contiguous>;
            fn $fn(self$(, rhs0: $t$(, rhs1: $s)?)?) -> Self::Output {
                let mut res = Tensor::alloc(T::default(), self.size);
                res.zip_with_mut(self, |x, &y| *x = args_call!($($t$(, $s)?)?; y.$fn(rhs0, rhs1)));
                res
            }
        }
    )*};
}

functional_unary_op_impl_ref_tensor! {
    Add<T> add;
    Sub<T> sub;
    Mul<T> mul;
    Div<T> div;
    Rem<T> rem;
    Atan2<T> atan2;
    Hypot<T> hypot;
    Copysign<T> copysign;
    DivEuclid<T> div_euclid;
    RemEuclid<T> rem_euclid;
    Max<T> max;
    Min<T> min;
    MaxMask<T> max_mask;
    MinMask<T> min_mask;
    Add<&'a T> add for<'a>;
    Sub<&'a T> sub for<'a>;
    Mul<&'a T> mul for<'a>;
    Div<&'a T> div for<'a>;
    Rem<&'a T> rem for<'a>;
    Atan2<&'a T> atan2 for<'a>;
    Hypot<&'a T> hypot for<'a>;
    Copysign<&'a T> copysign for<'a>;
    DivEuclid<&'a T> div_euclid for<'a>;
    RemEuclid<&'a T> rem_euclid for<'a>;
    Max<&'a T> max for<'a>;
    Min<&'a T> min for<'a>;
    MaxMask<&'a T> max_mask for<'a>;
    MinMask<&'a T> min_mask for<'a>;
    Pow<U> pow for<U>;
    Log<U> log for<U>;
    Neg neg;
    Exp exp;
    Exp2 exp2;
    ExpM1 exp_m1;
    Ln ln;
    Ln1p ln_1p;
    Log2 log2;
    Log10 log10;
    Sin sin;
    Cos cos;
    Tan tan;
    Sinh sinh;
    Cosh cosh;
    Tanh tanh;
    Asin asin;
    Acos acos;
    Atan atan;
    Asinh asinh;
    Acosh acosh;
    Atanh atanh;
    Sqrt sqrt;
    Cbrt cbrt;
    Abs abs;
    Signum signum;
    Ceil ceil;
    Floor floor;
    Round round;
    Trunc trunc;
    Fract fract;
    Recip recip;
    ToDegrees to_degrees;
    ToRadians to_radians;
    Conj conj;
    MulAdd<T, T> mul_add
}

// -------------------------
// binary ops &A -> &A -> B
// -------------------------

macro_rules! functional_binary_op_impl_ref_tensor {
    ($($trait:ident $fn:ident);*) => {$(
        impl<B, B2, T, S, C, C2> $trait<&Tensor<B2, T, S, C2>> for &Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem> + Realloc<T, S::Elem>,
            B2: KindTypeTypeType<T, S::Elem>,
            <B::Buffer as KindTypeTypeType<T, S::Elem>>::Applied: AsMut<[T]>,
            B::Applied: AsRef<[T]>,
            B2::Applied: AsRef<[T]>,
            S: Axes,
            T: $trait<Output = T> + Default + Copy + 'static,
        {
            type Output = Tensor<B::Buffer, T, S, Contiguous>;
            fn $fn(self, rhs: &Tensor<B2, T, S, C2>) -> Self::Output {
                let mut res = Tensor::alloc(T::default(), self.size);
                res.zip2_with_mut(self, rhs, |x, &y, &z| *x = y.$fn(z));
                res
            }
        }
    )*};
}

functional_binary_op_impl_ref_tensor! {
    Add add;
    Sub sub;
    Mul mul;
    Div div;
    Rem rem;
    Atan2 atan2;
    Hypot hypot;
    Copysign copysign;
    DivEuclid div_euclid;
    RemEuclid rem_euclid;
    Max max;
    Min min;
    MaxMask max_mask;
    MinMask min_mask
}

// ---------------------------------
// ternary ops: &A -> &A -> &A -> B
// ---------------------------------

impl<B, B2, B3, T, S, C, C2, C3> MulAdd<&Tensor<B2, T, S, C2>, &Tensor<B3, T, S, C3>> for &Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem> + Realloc<T, S::Elem>,
    B2: KindTypeTypeType<T, S::Elem>,
    B3: KindTypeTypeType<T, S::Elem>,
    <B::Buffer as KindTypeTypeType<T, S::Elem>>::Applied: AsMut<[T]>,
    B::Applied: AsRef<[T]>,
    B2::Applied: AsRef<[T]>,
    B3::Applied: AsRef<[T]>,
    S: Axes,
    T: MulAdd<Output = T> + Default + Copy + 'static,
{
    type Output = Tensor<B::Buffer, T, S, Contiguous>;
    fn mul_add(self, rhs0: &Tensor<B2, T, S, C2>, rhs1: &Tensor<B3, T, S, C3>) -> Self::Output {
        let mut res = Tensor::alloc(T::default(), self.size);
        res.zip3_with_mut(self, rhs0, rhs1, |x, &y, &z, &w| *x = y.mul_add(z, w));
        res
    }
}

// ----------------------------------------------
// unary ops:   B (-> K | &K)* -> move,inplace B
// binary ops:  B -> &A -> move,inplace B
//              B -> A -> move,inplace B
// ternary ops: B -> &A -> &A -> move,inplace B
// ----------------------------------------------

macro_rules! functional_ops_impl_tensor {
    ($($trait:ident$(<$t:ident$(, $s:ident)?>)? $fn:ident $delegate_trait:ident $delegate_fn:ident);*) => {$(
        impl<B, T, S, C$(, $t$(, $s)?)?> $trait$(<$t$(, $s)?>)? for Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem> + Realloc<T, S::Elem, Buffer = B>,
            S: Axes,
            Self: $delegate_trait$(<$t$(, $s)?>)?,
        {
            type Output = Self;
            fn $fn(mut self$(, rhs0: $t$(, rhs1: $s)?)?) -> Self::Output {
                args_call!($($t$(, $s)?)?; self.$delegate_fn(rhs0, rhs1));
                self
            }
        }
    )*};
}

functional_ops_impl_tensor! {
    Add<Rhs> add AddAssign add_assign;
    Sub<Rhs> sub SubAssign sub_assign;
    Mul<Rhs> mul MulAssign mul_assign;
    Div<Rhs> div DivAssign div_assign;
    Rem<Rhs> rem RemAssign rem_assign;
    Atan2<Rhs> atan2 Atan2Assign atan2_assign;
    Hypot<Rhs> hypot HypotAssign hypot_assign;
    Copysign<Rhs> copysign CopysignAssign copysign_assign;
    DivEuclid<Rhs> div_euclid DivEuclidAssign div_euclid_assign;
    RemEuclid<Rhs> rem_euclid RemEuclidAssign rem_euclid_assign;
    Max<Rhs> max MaxAssign max_assign;
    Min<Rhs> min MinAssign min_assign;
    MaxMask<Rhs> max_mask MaxMaskAssign max_mask_assign;
    MinMask<Rhs> min_mask MinMaskAssign min_mask_assign;
    Pow<Rhs> pow PowAssign pow_assign;
    Log<Rhs> log LogAssign log_assign;
    Exp exp ExpAssign exp_assign;
    Exp2 exp2 Exp2Assign exp2_assign;
    ExpM1 exp_m1 ExpM1Assign exp_m1_assign;
    Ln ln LnAssign ln_assign;
    Ln1p ln_1p Ln1pAssign ln_1p_assign;
    Log2 log2 Log2Assign log2_assign;
    Log10 log10 Log10Assign log10_assign;
    Sin sin SinAssign sin_assign;
    Cos cos CosAssign cos_assign;
    Tan tan TanAssign tan_assign;
    Sinh sinh SinhAssign sinh_assign;
    Cosh cosh CoshAssign cosh_assign;
    Tanh tanh TanhAssign tanh_assign;
    Asin asin AsinAssign asin_assign;
    Acos acos AcosAssign acos_assign;
    Atan atan AtanAssign atan_assign;
    Asinh asinh AsinhAssign asinh_assign;
    Acosh acosh AcoshAssign acosh_assign;
    Atanh atanh AtanhAssign atanh_assign;
    Sqrt sqrt SqrtAssign sqrt_assign;
    Cbrt cbrt CbrtAssign cbrt_assign;
    Abs abs AbsAssign abs_assign;
    Signum signum SignumAssign signum_assign;
    Ceil ceil CeilAssign ceil_assign;
    Floor floor FloorAssign floor_assign;
    Round round RoundAssign round_assign;
    Trunc trunc TruncAssign trunc_assign;
    Fract fract FractAssign fract_assign;
    Recip recip RecipAssign recip_assign;
    ToDegrees to_degrees ToDegreesAssign to_degrees_assign;
    ToRadians to_radians ToRadiansAssign to_radians_assign;
    Conj conj ConjAssign conj_assign;
    MulAdd<Rhs0, Rhs1> mul_add MulAddAssign mul_add_assign
}

impl<B, T, S, C> Neg for Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem> + Realloc<T, S::Elem, Buffer = B>,
    B::Applied: AsMut<[T]>,
    S: Axes,
    Self: MulAssign<T>,
    T: Neg<Output = T> + Copy + 'static,
{
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.for_each(|x| *x = -*x);
        self
    }
}

// -----------------------------------------
// unary ops: K | &K -> &A -> B
//            K | &K -> B -> move,inplace B
// -----------------------------------------

macro_rules! for_types {
    ($t:ty, $($u:ty),+: $($tail:tt)*) => {
        for_types!($t: $($tail)*);
        for_types!($($u),+: $($tail)*);
    };
    ($t:ty: $macro:ident $($tail:tt)*) => {
        $macro! { $($tail)* for $t }
    };
}

macro_rules! for_scalar_types {
    ($($tail:tt)*) => {
        for_types!(f64, f32, Complex64, Complex32,
            i128, i64, i32, i16, i8,
            u128, u64, u32, u16, u8: $($tail)*);
    };
}

macro_rules! for_scalar_types_no_complex {
    ($($tail:tt)*) => {
        for_types!(f64, f32,
            i128, i64, i32, i16, i8,
            u128, u64, u32, u16, u8: $($tail)*);
    };
}

macro_rules! for_floats {
    ($($tail:tt)*) => {
        for_types!(f64, f32: $($tail)*);
    };
}

macro_rules! reversed_functional_unary_ops_impl {
    ($trait:ident $fn:ident for $t:ty) => {
        impl<B, S, C> $trait<&Tensor<B, $t, S, C>> for $t
        where
            B: KindTypeTypeType<$t, S::Elem> + Realloc<$t, S::Elem>,
            <B::Buffer as KindTypeTypeType<$t, S::Elem>>::Applied: AsMut<[$t]>,
            B::Applied: AsRef<[$t]>,
            S: Axes,
            $t: $trait<Output = $t>,
        {
            type Output = Tensor<B::Buffer, $t, S, Contiguous>;
            fn $fn(self, rhs: &Tensor<B, $t, S, C>) -> Self::Output {
                let mut res = Tensor::alloc(<$t>::default(), rhs.size);
                res.zip_with_mut(&rhs, |x, &y| *x = <$t as $trait>::$fn(self, y));
                res
            }
        }

        impl<'a, B, S, C> $trait<&'a Tensor<B, $t, S, C>> for &'a $t
        where
            B: KindTypeTypeType<$t, S::Elem>,
            S: Axes,
            $t: $trait<&'a Tensor<B, $t, S, C>>,
        {
            type Output = <$t as $trait<&'a Tensor<B, $t, S, C>>>::Output;
            fn $fn(self, rhs: &'a Tensor<B, $t, S, C>) -> Self::Output {
                <$t as $trait<&'a Tensor<B, $t, S, C>>>::$fn(*self, rhs)
            }
        }

        impl<B, S, C> $trait<Tensor<B, $t, S, C>> for $t
        where
            B: KindTypeTypeType<$t, S::Elem> + Realloc<$t, S::Elem, Buffer = B>,
            B::Applied: AsMut<[$t]>,
            S: Axes,
            $t: $trait<Output = $t>,
        {
            type Output = Tensor<B, $t, S, C>;
            fn $fn(self, mut rhs: Tensor<B, $t, S, C>) -> Self::Output {
                rhs.for_each(|x| *x = <$t as $trait>::$fn(self, *x));
                rhs
            }
        }

        impl<B, S, C> $trait<Tensor<B, $t, S, C>> for &$t
        where
            B: KindTypeTypeType<$t, S::Elem>,
            S: Axes,
            $t: $trait<Tensor<B, $t, S, C>>,
        {
            type Output = <$t as $trait<Tensor<B, $t, S, C>>>::Output;
            fn $fn(self, rhs: Tensor<B, $t, S, C>) -> Self::Output {
                <$t as $trait<Tensor<B, $t, S, C>>>::$fn(*self, rhs)
            }
        }
    };
}

macro_rules! reversed_functional_unary_ops_impl_scalar_types {
    ($($trait:ident $fn:ident $macro:ident);*) => {$(
        $macro! { reversed_functional_unary_ops_impl $trait $fn }
    )*};
}

reversed_functional_unary_ops_impl_scalar_types! {
    Add add for_scalar_types;
    Sub sub for_scalar_types;
    Mul mul for_scalar_types;
    Div div for_scalar_types;
    Rem rem for_scalar_types;
    Atan2 atan2 for_floats;
    Hypot hypot for_floats;
    Copysign copysign for_floats;
    DivEuclid div_euclid for_scalar_types_no_complex;
    RemEuclid rem_euclid for_scalar_types_no_complex;
    Max max for_scalar_types_no_complex;
    Min min for_scalar_types_no_complex;
    MaxMask max_mask for_scalar_types_no_complex;
    MinMask min_mask for_scalar_types_no_complex
}

// -------------------------------------
// binary ops &A -> B -> move,inplace B
// -------------------------------------

macro_rules! reversed_functional_binary_op_impl_ref_tensor {
    ($($trait:ident $fn:ident);*) => {$(
        impl<B, B2, T, S, C, C2> $trait<Tensor<B2, T, S, C2>> for &Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem>,
            B2: KindTypeTypeType<T, S::Elem> + Realloc<T, S::Elem, Buffer = B2>,
            B::Applied: AsRef<[T]>,
            B2::Applied: AsMut<[T]>,
            S: Axes,
            T: $trait<Output = T> + Copy + 'static,
        {
            type Output = Tensor<B2, T, S, C2>;
            fn $fn(self, mut rhs: Tensor<B2, T, S, C2>) -> Self::Output {
                rhs.zip_with_mut(self, |x, &y| *x = y.$fn(*x));
                rhs
            }
        }
    )*};
}

reversed_functional_binary_op_impl_ref_tensor! {
    Add add;
    Sub sub;
    Mul mul;
    Div div;
    Rem rem;
    Atan2 atan2;
    Hypot hypot;
    Copysign copysign;
    DivEuclid div_euclid;
    RemEuclid rem_euclid;
    Max max;
    Min min;
    MaxMask max_mask;
    MinMask min_mask
}

// -----------------------------------
// unary conversions: &A (-> K)* -> B
// -----------------------------------

macro_rules! unary_conversions_impl_ref_tensor {
    ($($trait:ident$(<$t:ty$(, $s:ty)?>)? $fn:ident);*) => {$(
        impl<B, T, S, C> $trait$(<$t$(, $s)?>)? for &Tensor<B, T, S, C>
        where
            T: $trait$(<$t$(, $s)?>)? + Copy + 'static,
            <T as $trait$(<$t$(, $s)?>)?>::Output: Default + 'static,
            B: KindTypeTypeType<T, S::Elem> + Realloc<<T as $trait$(<$t$(, $s)?>)?>::Output, S::Elem>,
            <B::Buffer as KindTypeTypeType<<T as $trait$(<$t$(, $s)?>)?>::Output, S::Elem>>::Applied: AsMut<[<T as $trait$(<$t$(, $s)?>)?>::Output]>,
            B::Applied: AsRef<[T]>,
            S: Axes,
        {
            type Output = Tensor<B::Buffer, <T as $trait$(<$t$(, $s)?>)?>::Output, S, Contiguous>;
            fn $fn(self$(, rhs0: $t$(, rhs1: $s)?)?) -> Self::Output {
                let mut res = Tensor::alloc(<<T as $trait$(<$t$(, $s)?>)?>::Output>::default(), self.size);
                res.zip_with_mut(self, |x, &y| *x = args_call!($($t$(, $s)?)?; y.$fn(rhs0, rhs1)));
                res
            }
        }
    )*};
}

unary_conversions_impl_ref_tensor! {
    Re re;
    Im im;
    Norm norm;
    NormSqr norm_sqr;
    Arg arg;
    J j;
    EPowJ e_pow_j;
    AddJ<T> add_j;
    MulEPowJ<T> mul_e_pow_j
}

// -----------------------------------------
// reversed unary conversions: K -> &A -> B
// -----------------------------------------

macro_rules! reversed_unary_conversions_impl {
    ($trait:ident $fn:ident for $t:ty) => {
        impl<B, S, C> $trait<&Tensor<B, $t, S, C>> for $t
        where
            B: KindTypeTypeType<$t, S::Elem> + Realloc<Complex<$t>, S::Elem>,
            <B::Buffer as KindTypeTypeType<Complex<$t>, S::Elem>>::Applied: AsMut<[Complex<$t>]>,
            B::Applied: AsRef<[$t]>,
            S: Axes,
        {
            type Output = Tensor<B::Buffer, Complex<$t>, S, Contiguous>;
            fn $fn(self, rhs: &Tensor<B, $t, S, C>) -> Self::Output {
                let mut res = Tensor::alloc(Complex::<$t>::default(), rhs.size);
                res.zip_with_mut(rhs, |x, &y| *x = y.$fn(self));
                res
            }
        }
    };
}

macro_rules! reversed_unary_conversions_impl_floats {
    ($($trait:ident $fn:ident);*) => {$(
        for_floats! { reversed_unary_conversions_impl $trait $fn }
    )*};
}

reversed_unary_conversions_impl_floats! {
    AddJ add_j;
    MulEPowJ mul_e_pow_j
}

// ----------------------------------
// binary conversions: &A -> &A -> B
// ----------------------------------

macro_rules! binary_conversions_impl_tensor {
    ($($trait:ident $fn:ident);*) => {$(
        impl<B, B2, T, S, C, C2> $trait<&Tensor<B2, T, S, C2>> for &Tensor<B, T, S, C>
        where
            T: $trait + Copy + 'static,
            <T as $trait>::Output: Default + 'static,
            B: KindTypeTypeType<T, S::Elem> + Realloc<<T as $trait>::Output, S::Elem>,
            <B::Buffer as KindTypeTypeType<<T as $trait>::Output, S::Elem>>::Applied: AsMut<[<T as $trait>::Output]>,
            B::Applied: AsRef<[T]>,
            B2: KindTypeTypeType<T, S::Elem>,
            B2::Applied: AsRef<[T]>,
            S: Axes,
        {
            type Output = Tensor<B::Buffer, <T as $trait>::Output, S, Contiguous>;
            fn $fn(self, rhs: &Tensor<B2, T, S, C2>) -> Self::Output {
                let mut res = Tensor::alloc(<<T as $trait>::Output>::default(), self.size);
                res.zip2_with_mut(self, rhs, |x, &y, &z| *x = y.$fn(z));
                res
            }
        }
    )*};
}

binary_conversions_impl_tensor! {
    AddJ add_j;
    MulEPowJ mul_e_pow_j
}

pub trait TensorCast<U> {
    /// Output type.
    type Output;
    /// Cast all elements into `U`.
    fn as_(self) -> Self::Output;
}

impl<B, T, U, S, C> TensorCast<U> for &Tensor<B, T, S, C>
where
    T: Cast<U> + Copy + 'static,
    U: Default + 'static,
    B: KindTypeTypeType<T, S::Elem> + Realloc<U, S::Elem>,
    <B::Buffer as KindTypeTypeType<U, S::Elem>>::Applied: AsMut<[U]>,
    B::Applied: AsRef<[T]>,
    S: Axes,
{
    type Output = Tensor<B::Buffer, U, S, Contiguous>;
    fn as_(self) -> Self::Output {
        let mut res = Tensor::alloc(U::default(), self.size);
        res.zip_with_mut(self, |x, y| *x = y.as_());
        res
    }
}
