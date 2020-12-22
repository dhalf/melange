//! Reexports commonly used features.

pub use super::index::Index;
pub use super::layout::Layout;
pub use super::strided_iterator::{StridedIterator, StridedIteratorMut};
pub use std::convert::TryFrom;
pub use super::shape::*;
pub use super::{Tensor, Static, Contiguous, Normal, Dynamic, Strided, Transposed, DynamicTensor, StaticTensor, AsRawSlice, AsRawSliceMut};
pub use std::ops::{Add, Sub, Mul, Div, Rem};
pub use super::tensor::view::*;
pub use super::alloc::*;
pub use super::reduction::*;
pub use super::linear_algebra::*;
