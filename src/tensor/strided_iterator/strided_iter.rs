use crate::tensor::index::Index;
use crate::tensor::layout::Layout;
use std::convert::TryFrom;
use std::marker::PhantomData;
use typenum::Unsigned;

/// Iterator that yields immutable chunks of data from a
/// strided tensor.
///
/// Chunks correspond to the abstract layout of the tensor
/// as defined in its [`Layout`](crate::tensor::layout::Layout)
/// and not to the underlying real layout of its data.
/// For instance, it will repeatedly yield the same actual
/// slices in the case of a broadcasted tensor.
///
/// `StridedIter` is the output of
/// [`StridedIterator::strided_iter`](super::StridedIterator::strided_iter)
/// on a strided [`Tensor`](crate::tensor::Tensor)
///
/// # Safety
/// `StridedIter` internally uses raw poiters to speed up
/// iteration. Proper bound checking is done to avoid potential
/// buffer overflows.
/// *This struct and its [`Iterator`](std::iter::Iterator)
/// implementation need proper auditing.*
pub struct StridedIter<'a, T: 'a, U> {
    ptr: *const T,
    len: usize,
    hint: usize,
    bounds: Index<U>,
    strides: Index<U>,
    step_sizes: Index<U>,
    state: Index<U>,
    chunk_size: usize,
    offset: usize,
    dead: bool,
    _phantoms: PhantomData<&'a T>,
}

impl<'a, T, U> StridedIter<'a, T, U>
where
    U: Unsigned,
{
    pub(in crate::tensor) fn new<L>(data: &'a [T], layout: &L, chunk_size: usize) -> Self
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

        StridedIter {
            ptr: data.as_ptr(),
            len: data.len(),
            hint: layout.num_elements(),
            bounds: layout.shape(),
            strides: layout.strides(),
            step_sizes,
            state: Index::try_from(vec![0; U::USIZE]).unwrap(),
            chunk_size: chunk_size / step, // Chunks must match innermost axes.
            offset: 0,
            dead: false,
            _phantoms: PhantomData,
        }
    }
}

impl<'a, T, U> Iterator for StridedIter<'a, T, U>
where
    T: 'a,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.dead {
            return None;
        }

        // Buffer overflow guard
        assert!(
            self.offset + self.chunk_size <= self.len,
            "Buffer overflow detected in StridedIter. Aborting! {:?} {:?} {:?} {:?}",
            self.offset,
            self.chunk_size,
            self.len,
            self.dead,
        );

        let chunk = unsafe {
            let data = self.ptr.offset(self.offset as isize);
            std::slice::from_raw_parts(data, self.chunk_size)
        };

        for (((digit, step_size), bound), stride) in self
            .state
            .iter_mut()
            .zip(self.step_sizes.iter())
            .zip(self.bounds.iter())
            .zip(self.strides.iter())
            .rev()
        {
            if *digit + step_size >= *bound {
                self.offset -= *digit * step_size * stride;
                *digit = 0;
            } else {
                self.offset += step_size * stride;
                *digit += step_size;
                return Some(chunk);
            }
        }

        self.dead = true;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.hint / self.chunk_size;
        (exact, Some(exact))
    }
}
