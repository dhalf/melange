use crate::gat::{Gat, StreamingIterator};

/// Defines how to get a mutable streaming iterator yielding
/// chunks of data from a tensor following its abstract
/// layout.
/// 
/// The abstract layout is the layout defined by the
/// [`Layout`](crate::tensor::layout::Layout)
/// associated with the tensor. It is distinct from
/// the actual real layout of its data in memory.
/// 
/// This trait is similar to IntoIter but the tensor is
/// borrowed instead of being moved.
pub trait StridedIteratorMut {
    /// [`GAT`](crate::gat) used to define the actual
    /// yielded types.
    type Item: for<'a> Gat<'a>;

    /// Type of iterator.
    type StridedIterMut: StreamingIterator<Item = Self::Item>;
    
    /// The actual function that outputs an iterator
    /// over the chunks.
    fn strided_iter_mut(self, chunk_size: usize) -> Self::StridedIterMut;
}
