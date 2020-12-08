use super::Layout;
use crate::tensor::index::Index;
use std::fmt;

/// Layout type suitable for dynamic or strided tensors.
/// 
/// It stores the runtime shape, strides, num_elements and
/// optimal_chunk_size of the tensor.
#[derive(Clone)]
pub struct DynamicLayout<U> {
    pub(in crate::tensor) shape: Index<U>,
    pub(in crate::tensor) strides: Index<U>,
    pub(in crate::tensor) num_elements: usize,
    pub(in crate::tensor) opt_chunk_size: usize,
}

impl<U> Layout<U> for DynamicLayout<U>
where
    U: Clone,
{
    #[inline]
    fn linear_index(&self, index: &Index<U>) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(x, y)| *x * *y)
            .sum()
    }

    #[inline]
    fn shape(&self) -> Index<U> {
        self.shape.clone()
    }

    #[inline]
    fn strides(&self) -> Index<U> {
        self.strides.clone()
    }

    #[inline]
    fn num_elements(&self) -> usize {
        self.num_elements
    }

    #[inline]
    fn opt_chunk_size(&self) -> usize {
        self.opt_chunk_size
    }
}

impl<U> fmt::Debug for DynamicLayout<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicLayout")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("num_elements", &self.num_elements)
            .field("opt_chunk_size", &self.opt_chunk_size)
            .finish()
    }
}
