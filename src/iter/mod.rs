//! Provides tools to properly iterate over potentially broadcasted, strided,
//! transposed, ... tensors.

pub mod streaming_iterator;
pub mod strided_iter;

pub use streaming_iterator::StreamingIterator;
