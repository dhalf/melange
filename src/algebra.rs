//! Defines traits that provide useful features
//! of common algebraic structures.
//! 
//! This include identity elements
//! and multiplicative inversion.
//! 
//! Relevant traits are implemented for all primitive
//! numeric types.

use std::ops::*;

/// Algebraic ring features.
/// 
/// It is implemented for floats and signed itegers.
/// But more surprisingly, since it only defines
/// the identity elements (additive inverse is
/// a primitive operation in Rust), we also choose
/// to imlement it for unsigned itegers as well though
/// they aren't part of a ring (their structure
/// is a monoid for both addition and multiplication).
pub trait Ring {
    /// Additive identity element.
    const ZERO: Self;

    /// Multiplicative identity element.
    const ONE: Self;
}

/// Algebraic field features.
/// 
/// This extends the [`Ring`](Ring)
/// trait with multiplicative inversion.
pub trait Field: Ring {
    /// Multiplicative inverse function.
    #[inline]
    fn minv(self) -> Self
    where
        Self: Sized + Div<Output = Self>,
    {
        Self::ONE / self
    }
}

macro_rules! ring_impl_float {
    ($($t:ty)*) => ($(
        impl Ring for $t {
            const ZERO: $t = 0.0;
            const ONE: $t = 1.0;
        }
    )*)
}

ring_impl_float! { f64 f32 }

macro_rules! field_impl_float {
    ($($t:ty)*) => ($(
        impl Field for $t {
            fn minv(self) -> Self {
                self.recip()
            }
        }
    )*)
}

field_impl_float! { f64 f32 }

macro_rules! ring_impl_integer {
    ($($t:ty)*) => ($(
        impl Ring for $t {
            const ZERO: $t = 0;
            const ONE: $t = 1;
        }
    )*)
}

ring_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }
