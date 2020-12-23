use super::Layout;
use crate::tensor::index::Index;
use crate::tensor::shape::StaticShape;
use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;
use typenum::Unsigned;

/// Layout type suitable for static and contiguous tensors.
///
/// It has zero overhead since for those tensors the layout
/// is known at compile time.
#[derive(Clone)]
pub struct StaticLayout<S> {
    _phantoms: PhantomData<S>,
}

impl<S> StaticLayout<S> {
    /// Default constructor.
    pub fn new() -> Self {
        StaticLayout {
            _phantoms: PhantomData,
        }
    }
}

impl<S> Layout<S::Len> for StaticLayout<S>
where
    S: StaticShape + Clone,
{
    #[inline]
    fn linear_index(&self, index: &Index<S::Len>) -> usize {
        index
            .iter()
            .zip(S::strides().into_iter())
            .map(|(x, y)| *x * y)
            .sum()
    }

    #[inline]
    fn shape(&self) -> Index<S::Len> {
        Index::try_from(S::to_vec()).unwrap()
    }

    #[inline]
    fn strides(&self) -> Index<S::Len> {
        Index::try_from(S::strides()).unwrap()
    }

    #[inline]
    fn num_elements(&self) -> usize {
        S::NumElements::USIZE
    }

    #[inline]
    fn opt_chunk_size(&self) -> usize {
        S::NumElements::USIZE
    }
}

impl<S> fmt::Debug for StaticLayout<S>
where
    S: StaticShape,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tuple_output = f.debug_tuple("StaticLayout");
        for ax in S::to_vec() {
            tuple_output.field(&ax);
        }
        tuple_output.finish()
    }
}
