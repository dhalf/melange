use crate::tensor::index::Index;

/// Common interface for layout types.
pub trait Layout<U>: Clone {
    /// Computes the linear index (data pointer offset) corresponding
    /// to the given [`Index`](Index).
    ///
    /// This takes into account both intrinsic and extrinsic strides.
    ///
    /// # Examples
    /// ```
    /// use std::convert::TryFrom;
    /// use typenum::U2;
    /// use melange::tensor::shape::Shape2D;
    /// use melange::tensor::layout::{Layout, StaticLayout};
    /// use melange::tensor::index::Index;
    ///
    /// let layout: StaticLayout<Shape2D<U2, U2>> = StaticLayout::new();
    /// let coord = Index::try_from(vec![1, 0]).unwrap();
    /// assert_eq!(layout.linear_index(&coord), 2);
    /// ```
    fn linear_index(&self, index: &Index<U>) -> usize;

    /// Returns an [`Index`](Index) representing the shape of the tensor.
    ///
    /// # Examples
    /// ```
    /// use typenum::U2;
    /// use melange::tensor::shape::Shape2D;
    /// use melange::tensor::layout::{Layout, StaticLayout};
    ///
    /// let layout: StaticLayout<Shape2D<U2, U2>> = StaticLayout::new();
    /// assert_eq!(Vec::from(layout.shape()), vec![2, 2]);
    /// ```
    fn shape(&self) -> Index<U>;
    /// Returns an [`Index`](Index) representing the strides of the tensor.
    ///
    /// Returned strides are the product of intrinsic strides
    /// (see [`intrinsic_strides_in_place`](crate::tensor::shape::intrinsic_strides_in_place))
    /// and extrinsic strides.
    /// Extrinsic strides are what is usually simply called strides.
    ///
    /// For instance, when one element every two elements is retained
    /// along an axis, the extrinsic stride is two.
    /// If that axis is the second to last and the last axis has
    /// a dimension of 5, then the instrinsic stride along that axis
    /// is 5 and the stride is 10.
    ///
    /// # Examples
    /// ```
    /// use typenum::U2;
    /// use melange::tensor::shape::Shape2D;
    /// use melange::tensor::layout::{Layout, StaticLayout};
    ///
    /// let layout: StaticLayout<Shape2D<U2, U2>> = StaticLayout::new();
    /// assert_eq!(Vec::from(layout.strides()), vec![2, 1]);
    /// ```
    fn strides(&self) -> Index<U>;
    /// Returns an [`Index`](Index) representing the offset of the tensor.
    ///
    /// This offset is the collection of the stating index of all the
    /// axes of a tensor. For most tensors, the offset is just full
    /// of zeros. This feature is manly useful for subviews on tensors.
    ///
    /// # Examples
    /// ```
    /// use typenum::U2;
    /// use melange::tensor::shape::Shape2D;
    /// use melange::tensor::layout::{Layout, StaticLayout};
    ///
    /// let layout: StaticLayout<Shape2D<U2, U2>> = StaticLayout::new();
    /// assert_eq!(Vec::from(layout.offset()), vec![0, 0]);
    /// ```
    fn offset(&self) -> Index<U>;
    /// Returns the number of elements in the tensor.
    ///
    /// # Examples
    /// ```
    /// use typenum::U2;
    /// use melange::tensor::shape::Shape2D;
    /// use melange::tensor::layout::{Layout, StaticLayout};
    ///
    /// let layout: StaticLayout<Shape2D<U2, U2>> = StaticLayout::new();
    /// assert_eq!(layout.num_elements(), 4);
    /// ```
    fn num_elements(&self) -> usize;

    /// Returns the optimal chunk size i.e. the number of elements
    /// of the largest contiguous chunks in the tensor.
    ///
    /// This is useful when iterating over the elements of the tensor
    /// to write two nested loops, the outer loop being over maximal
    /// chunks and the inner over contiguous elements in those chunks.
    /// This approach forces the compiler to optimize the inner loop
    /// knowing that elements are contiguous. This strategy is used
    /// throughout the library for non BLAS backed operations.
    ///
    /// # Examples
    /// ```
    /// use typenum::U2;
    /// use melange::tensor::shape::Shape2D;
    /// use melange::tensor::layout::{Layout, StaticLayout};
    ///
    /// let layout: StaticLayout<Shape2D<U2, U2>> = StaticLayout::new();
    /// assert_eq!(layout.opt_chunk_size(), 4);
    /// ```
    fn opt_chunk_size(&self) -> usize;
}
