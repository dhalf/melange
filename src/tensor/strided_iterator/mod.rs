//! Provides utilities to iterate over contiguous
//! chunks of data from tensors.
//! 
//! The most efficient way to iterate over the elements of a tensor
//! is to use two nested loops, the outer loop being over
//! chunks and the inner over contiguous elements in those chunks.
//! This approach forces the compiler to optimize the inner loop
//! knowing that elements are contiguous. This strategy is used
//! throughout the library for non BLAS backed operations.
//! 
//! The usage of those iterators is very classic except for those that
//! implement [`StreamingIterator`] that cannot be used in for loops.
//! See [`StreamingIterator`] documentation.
//! 
//! [`StreamingIterator`]: crate::gat::StreamingIterator

#[doc(hidden)]
pub mod strided_iter;

#[doc(hidden)]
pub mod strided_iter_mut;

#[doc(hidden)]
pub mod strided_iterator;

#[doc(hidden)]
pub mod strided_iterator_mut;

#[doc(hidden)]
pub mod chunks_mut;

#[doc(inline)]
pub use strided_iter::StridedIter;

#[doc(inline)]
pub use strided_iter_mut::StridedIterMut;

#[doc(inline)]
pub use strided_iterator::StridedIterator;

#[doc(inline)]
pub use strided_iterator_mut::StridedIteratorMut;

#[doc(inline)]
pub use chunks_mut::ChunksMut;
