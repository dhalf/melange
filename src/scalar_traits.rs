//! Defines traits that provide useful features
//! for numeric types.
//!
//! This includes identity elements.
//!
//! Relevant traits are implemented for all primitive
//! numeric types.

use num_complex::{Complex, Complex32, Complex64};

/// Defines the additive identity.
pub trait Zero {
    /// Additive identity element.
    const ZERO: Self;
}

macro_rules! zero_impl_float {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            const ZERO: $t = 0.0;
        }
    )*)
}

zero_impl_float! { f64 f32 }

macro_rules! zero_impl_integer {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            const ZERO: $t = 0;
        }
    )*)
}

zero_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

macro_rules! zero_impl_complex {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            const ZERO: $t = Complex::new(0.0, 0.0);
        }
    )*)
}

zero_impl_complex! { Complex64 Complex32 }

/// Defines the multiplicative identity.
pub trait One {
    /// Multiplicative identity element.
    const ONE: Self;
}

macro_rules! one_impl_float {
    ($($t:ty)*) => ($(
        impl One for $t {
            const ONE: $t = 1.0;
        }
    )*)
}

one_impl_float! { f64 f32 }

macro_rules! one_impl_integer {
    ($($t:ty)*) => ($(
        impl One for $t {
            const ONE: $t = 1;
        }
    )*)
}

one_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

macro_rules! zero_impl_complex {
    ($($t:ty)*) => ($(
        impl One for $t {
            const ONE: $t = Complex::new(1.0, 0.0);
        }
    )*)
}

zero_impl_complex! { Complex64 Complex32 }

/// Defines the identity element of minimum operation.
pub trait Infinity {
    /// Multiplicative identity element.
    const INFINITY: Self;
}

macro_rules! infinity_impl_float {
    ($($t:ty)*) => ($(
        impl Infinity for $t {
            const INFINITY: $t = <$t>::INFINITY;
        }
    )*)
}

infinity_impl_float! { f64 f32 }

macro_rules! infinity_impl_integer {
    ($($t:ty)*) => ($(
        impl Infinity for $t {
            const INFINITY: $t = <$t>::MAX;
        }
    )*)
}

infinity_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

/// Defines the identity element of minimum operation.
pub trait NegInfinity {
    /// Multiplicative identity element.
    const NEG_INFINITY: Self;
}

macro_rules! neg_infinity_impl_float {
    ($($t:ty)*) => ($(
        impl NegInfinity for $t {
            const NEG_INFINITY: $t = <$t>::NEG_INFINITY;
        }
    )*)
}

neg_infinity_impl_float! { f64 f32 }

macro_rules! neg_infinity_impl_integer {
    ($($t:ty)*) => ($(
        impl NegInfinity for $t {
            const NEG_INFINITY: $t = <$t>::MIN;
        }
    )*)
}

neg_infinity_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

/// Provides a natural logarithm of 2 constant.
pub trait Ln2 {
    /// Natural logarithm of 2.
    const LN_2: Self;
}

impl Ln2 for f64 {
    const LN_2: f64 = std::f64::consts::LN_2;
}

impl Ln2 for f32 {
    const LN_2: f32 = std::f32::consts::LN_2;
}

impl Ln2 for Complex64 {
    const LN_2: Complex64 = Complex::new(std::f64::consts::LN_2, 0.0);
}

impl Ln2 for Complex32 {
    const LN_2: Complex32 = Complex::new(std::f32::consts::LN_2, 0.0);
}

/// Provides a natural logarithm of 10 constant.
pub trait Ln10 {
    /// Natural logarithm of 10.
    const LN_10: Self;
}

impl Ln10 for f64 {
    const LN_10: f64 = std::f64::consts::LN_10;
}

impl Ln10 for f32 {
    const LN_10: f32 = std::f32::consts::LN_10;
}

impl Ln10 for Complex64 {
    const LN_10: Complex64 = Complex::new(std::f64::consts::LN_10, 0.0);
}

impl Ln10 for Complex32 {
    const LN_10: Complex32 = Complex::new(std::f32::consts::LN_10, 0.0);
}

/// Provides radians/degrees conversion constants.
pub trait RadDeg {
    /// Radians to degrees constant.
    const RAD_TO_DEG: Self;
    /// Degrees to radians constant.
    const DEG_TO_RAD: Self;
}

impl RadDeg for f64 {
    const RAD_TO_DEG: f64 = 180.0f64 / std::f64::consts::PI;
    const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0f64;
}

impl RadDeg for f32 {
    const RAD_TO_DEG: f32 = 57.2957795130823208767981548141051703_f32;
    const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0f32;
}

//

/// Extends numeric cast to [`Complex`](num_complex::Complex)
/// types.
///
/// Casts from `T` to `Complex<U>` create a complex
/// number whose real portion is the cast of the `T` value
/// and whose imaginary portion is [`U::ZERO`](Zero::ZERO) from the
/// [`Zero`] trait. The cast for the real part is as exact
/// as a cast from `T` to `U` is.
///
/// Casts from `Complex<T>` to `U` drop the imaginary
/// portion and cast the real portion to `U`. The real portion
/// cast is as exact as a cast from `T` to `U` is.
///
/// Cast from `Complex<T>` to `Complex<U>` cast both
/// the real and imaginary portions. Those two casts
/// are as exact as a cast from `T` to `U` is.
pub trait Cast<U> {
    /// Casts `self` as `U`.
    fn as_(self) -> U;
}

macro_rules! cast_impl {
    ($($t:ty as $u:ty)*) => {$(
        impl Cast<$u> for $t {
            fn as_(self) -> $u {
                self as $u
            }
        }

        impl Cast<Complex<$u>> for $t
        where
            $u: Zero,
        {
            fn as_(self) -> Complex<$u> {
                Complex::new(self as $u, <$u>::ZERO)
            }
        }

        impl Cast<$u> for Complex<$t> {
            fn as_(self) -> $u {
                self.re as $u
            }
        }

        impl Cast<Complex<$u>> for Complex<$t> {
            fn as_(self) -> Complex<$u> {
                Complex::new(self.re as $u, self.im as $u)
            }
        }
    )*};
}

macro_rules! expand_right {
    ($callback:ident! $u:ty as $($t:ty)*) => {
        $callback! { $($u as $t)* }
    };
}

macro_rules! list_to_pairs {
    ($callback:ident! $($t:ty,)*; $u:ty; $v:ty, $($w:ty,)*) => {
        expand_right! { $callback! $u as $($t)* $v $($w)* }
        list_to_pairs! { $callback! $($t,)* $u,; $v; $($w,)* }
    };
    ($callback:ident! $($t:ty,)*; $u:ty;) => {
        expand_right! { $callback! $u as $($t)* }
    };
    ($callback:ident! ;;$v:ty, $($w:ty,)*) => {
        list_to_pairs! { $callback! ; $v; $($w,)* }
    };
    ($callback:ident! $($t:ty)*) => {
        list_to_pairs! {$callback! ;;$($t,)*}
    };
}

list_to_pairs! { cast_impl! u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 f64 f32 }

/// Marker traits for scalar types that can be used with
/// backpropagation.
pub trait Differentiable {}

macro_rules! differentiable_impl {
    ($($t:ty)*) => ($(
        impl Differentiable for $t {}
    )*)
}

differentiable_impl! { f64 f32 Complex64 Complex32 }
