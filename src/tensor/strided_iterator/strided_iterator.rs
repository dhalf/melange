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
pub trait StridedIterator {
    /// Yielded type.
    type Item;

    /// Iterator type.
    type StridedIter: Iterator<Item = Self::Item>;

    /// The actual function that outputs an iterator
    /// over the chunks.
    fn strided_iter(self, chunk_size: usize) -> Self::StridedIter;
}
