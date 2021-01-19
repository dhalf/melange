use crate::gat::{Gat, RefMutGat, StreamingIterator};
use crate::tensor::index::Index;
use crate::tensor::layout::Layout;
use std::convert::TryFrom;
use std::marker::PhantomData;
use typenum::Unsigned;

/// Streaming iterator that yields (overlapping) mutable chunks
/// of data from a strided tensor.
///
/// Chunks correspond to the abstract layout of the tensor
/// as defined in its [`Layout`](crate::tensor::layout::Layout)
/// and not to the underlying real layout of its data.
/// For instance, it will repeatedly yield the same actual
/// slices in the case of a broadcasted tensor.
///
/// `StridedIterMut` is the output of
/// [`StridedIteratorMut::strided_iter_mut`](super::StridedIteratorMut::strided_iter_mut)
/// on a strided [`Tensor`](crate::tensor::Tensor)
///
/// # Safety
/// `StridedIterMut` internally uses raw poiters to speed up
/// iteration. Proper bound checking is done to avoid potential
/// buffer overflows.
/// *This struct and its [`Iterator`](std::iter::Iterator)
/// implementation need proper auditing.*
pub struct StridedIterMut<'a, T: 'a, U> {
    ptr: *mut T,
    len: usize,
    bounds: Index<U>,
    strides: Index<U>,
    step_sizes: Index<U>,
    initial_state: Index<U>,
    state: Index<U>,
    chunk_size: usize,
    offset: usize,
    dead: bool,
    _phantoms: PhantomData<&'a mut T>,
}

impl<'a, T, U> StridedIterMut<'a, T, U>
where
    U: Unsigned,
{
    pub(in crate::tensor) fn new<L>(data: &'a mut [T], layout: &L, chunk_size: usize) -> Self
    where
        L: Layout<U>,
    {
        let mut step_sizes = layout.shape();
        // Contiguous chunks cannot be greater then opt_chunk_size.
        let chunk_size = chunk_size.min(layout.opt_chunk_size());
        let mut step = chunk_size;

        for x in step_sizes.iter_mut().rev() {
            if step < *x {
                *x = 1;
            } else {
                step /= *x;
            }
        }

        StridedIterMut {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            bounds: layout.shape(),
            strides: layout.strides(),
            step_sizes,
            initial_state: Index::try_from(layout.offset()).unwrap(),
            state: Index::try_from(layout.offset()).unwrap(),
            chunk_size: chunk_size / step, // Chunks must match innermost axes.
            offset: layout.linear_index(&layout.offset()),
            dead: false,
            _phantoms: PhantomData,
        }
    }
}

impl<'a, T, U> StreamingIterator for StridedIterMut<'a, T, U>
where
    T: 'static,
{
    type Item = RefMutGat<[T]>;

    fn next<'b>(&'b mut self) -> Option<<Self::Item as Gat<'b>>::Output> {
        if self.dead {
            return None;
        }

        // Buffer overflow guard
        assert!(
            self.offset + self.chunk_size <= self.len,
            "Buffer overflow detected in StridedIterMut. Aborting!"
        );

        let chunk = unsafe {
            let data = self.ptr.offset(self.offset as isize);
            std::slice::from_raw_parts_mut(data, self.chunk_size)
        };

        for ((((digit, step_size), bound), stride), initial_digit) in self
            .state
            .iter_mut()
            .zip(self.step_sizes.iter())
            .zip(self.bounds.iter())
            .zip(self.strides.iter())
            .zip(self.initial_state.iter())
            .rev()
        {
            if *digit + step_size >= *bound {
                self.offset -= (*digit - *initial_digit) * step_size * stride;
                *digit = *initial_digit;
            } else {
                self.offset += step_size * stride;
                *digit += step_size;
                return Some(chunk);
            }
        }

        self.dead = true;
        Some(chunk)
    }
}
