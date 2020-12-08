//! Reexports commonly used features.

pub use super::index::Index;
pub use super::layout::Layout;
pub use super::strided_iterator::{StridedIterator, StridedIteratorMut};
pub use std::convert::TryFrom;
pub use super::shape::*;
pub use super::{Tensor, Static, Contiguous, Normal, Dynamic, Strided, Transposed, ToContiguous, DynamicTensor, StaticTensor};
pub use super::core_ops::*;
pub use std::ops::{Add, Sub, Mul, Div, Rem};
pub use super::tensor::view::*;
pub use super::reduction::*;
pub use super::linear_algebra::*;
