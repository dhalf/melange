//! Defines traits that provide useful features
//! of common algebraic structures.
//! 
//! This include identity elements.
//! 
//! Relevant traits are implemented for all primitive
//! numeric types.

/// Defines the additive identity of the implementor.
pub trait AdditiveIdentity {
    /// Additive identity element.
    const ZERO: Self;
}

macro_rules! add_id_impl_float {
    ($($t:ty)*) => ($(
        impl AdditiveIdentity for $t {
            const ZERO: $t = 0.0;
        }
    )*)
}

add_id_impl_float! { f64 f32 }

macro_rules! add_id_impl_integer {
    ($($t:ty)*) => ($(
        impl AdditiveIdentity for $t {
            const ZERO: $t = 0;
        }
    )*)
}

add_id_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

/// Defines the multiplicative identity of the implementor.
pub trait MultiplicativeIdentity {
    /// Multiplicative identity element.
    const ONE: Self;
}

macro_rules! mul_id_impl_float {
    ($($t:ty)*) => ($(
        impl MultiplicativeIdentity for $t {
            const ONE: $t = 1.0;
        }
    )*)
}

mul_id_impl_float! { f64 f32 }

macro_rules! mul_id_impl_integer {
    ($($t:ty)*) => ($(
        impl MultiplicativeIdentity for $t {
            const ONE: $t = 1;
        }
    )*)
}

mul_id_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

/// Marker trait for types having a field structure.
pub trait Field: AdditiveIdentity + MultiplicativeIdentity {}

macro_rules! field_impl_float {
    ($($t:ty)*) => ($(
        impl Field for $t {}
    )*)
}

field_impl_float! { f64 f32 }
