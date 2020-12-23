//! `tensor` is a collection of tools to interact with multidimensional arrays
//! that are at the core of ML pipelines.
//!
//! It defines various ways to store data and optimized mathematical operations.
//!
//! Contrary to other ndarray modules such as numpy in Python or ndarray in Rust,
//! this module allows full size checking at compile time thanks to type level
//! integers from the [`typenum`] crate.
//!
//! # Examples
//! ```
//! use melange::prelude::*;
//! use typenum::{U1, U2};
//!
//! let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![
//!     1, 0,
//!     0, 1
//! ]).unwrap();
//! let b: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![
//!     1, 1
//! ]).unwrap();
//! let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![
//!     2, 1,
//!     1, 2
//! ]).unwrap();
//! assert_eq!(a.add(&b.broadcast()), c);
//! ```
//!
//! [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html

use layout::{DynamicLayout, StaticLayout};
use shape::Shape;

#[doc(inline)]
pub use tensor::{
    AsRawSlice, AsRawSliceMut, Contiguous, Dynamic, Normal, Static, Strided, Tensor, Transposed,
};

pub mod elementwise_ops;
pub mod index;
pub mod layout;
pub mod linear_algebra;
#[doc(hidden)]
pub mod prelude;
pub mod reduction;
pub mod shape;
pub mod strided_iterator;
#[doc(hidden)]
pub mod tensor;

#[doc(inplace)]
pub use tensor::view;

#[doc(inplace)]
pub use tensor::alloc;

use alloc::DefaultAllocator;

/// Type alias for static, contiguous, non-transposed, heap-stored tensors.
pub type StaticTensor<T, S> =
    Tensor<Static, Contiguous, Normal, T, S, DefaultAllocator, Vec<T>, StaticLayout<S>>;

/// Type alias for dynamic, contiguous, non-transposed, heap-stored tensors.
pub type DynamicTensor<T, S> = Tensor<
    Dynamic,
    Contiguous,
    Normal,
    T,
    S,
    DefaultAllocator,
    Vec<T>,
    DynamicLayout<<S as Shape>::Len>,
>;
