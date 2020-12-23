use crate::gat::{Gat, RefMutGat, StreamingIterator};
use std::marker::PhantomData;

/// Streaming iterator that yields mutable chunks of data from a
/// contiguous tensor.
///
/// Chunks correspond to the underlying real layout of the
/// tensor's data because it is contiguous (real and abstract
/// layouts do match).
///
/// `ChunksMut` is the output of
/// [`StridedIteratorMut::strided_iter_mut`](super::StridedIteratorMut::strided_iter_mut)
/// on a contiguous [`Tensor`](crate::tensor::Tensor)
///
/// Note: slices' ChunksMut could have been use instead since
/// chunks never overlap. However, the choice of rewritting
/// ChunksMut as a streaming iterator made sense for the
/// sake of consistency in the
/// [`StridedIteratorMut`](super::StridedIteratorMut) trait.
///
/// # Safety
/// `ChunksMut` internally uses raw poiters to speed up
/// iteration. Proper bound checking is done to avoid potential
/// buffer overflows.
/// *This struct and its [`Iterator`](std::iter::Iterator)
/// implementation need proper auditing.*
pub struct ChunksMut<'a, T: 'a> {
    ptr: *mut T,
    offset: usize,
    len: usize,
    chunk_size: usize,
    _phantoms: PhantomData<&'a mut T>,
}

impl<'a, T> ChunksMut<'a, T> {
    pub(in crate::tensor) fn new(data: &'a mut [T], chunk_size: usize) -> Self {
        ChunksMut {
            ptr: data.as_mut_ptr(),
            offset: 0,
            len: data.len(),
            chunk_size: chunk_size,
            _phantoms: PhantomData,
        }
    }
}

impl<'a, T> StreamingIterator for ChunksMut<'a, T>
where
    T: 'static,
{
    type Item = RefMutGat<[T]>;

    fn next<'b>(&'b mut self) -> Option<<Self::Item as Gat<'b>>::Output> {
        if self.offset + self.chunk_size <= self.len {
            // Buffer overflow guard
            let chunk = unsafe {
                let data = self.ptr.offset(self.offset as isize);
                std::slice::from_raw_parts_mut(data, self.chunk_size)
            };
            self.offset += self.chunk_size;
            Some(chunk)
        } else {
            None
        }
    }
}
