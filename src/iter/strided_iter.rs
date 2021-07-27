//! Provides iterators designed to iterate over tensors while taking broadcasting,
//! striding, blocks and transposition into account.

use super::StreamingIterator;
use crate::hkt::{KindLifetimeType, RefConstructorPA1, RefMutConstructorPA1};
use std::marker::PhantomData;

/// Streaming iterator that yields (overlapping) chunks
/// of data.
///
/// Chunks correspond to the abstract layout of the tensor
/// defined in a specific tensor type
/// and not to the underlying real layout of its data.
/// For instance, it will repeatedly yield the same actual
/// slices in the case of a broadcasted tensor.
///
/// `StridedIter` is used internally by the functions in the [`tensor`](crate::tensor)
/// module that implicitly iterate over tensors (e.g.
/// [`Tensor::zip_with_mut`](crate::tensor::Tensor::zip_with_mut)).
///
/// # Safety
/// `StridedIter` internally uses raw poiters to speed up
/// iteration. Proper bound checking is done to avoid potential
/// buffer overflows.
/// *This struct and its
/// implementations need proper auditing.*
pub struct StridedIter<'a, T: 'a, A> {
    ptr: *const T,
    len: usize,
    bounds: A,
    strides: A,
    step_sizes: A,
    initial_state: A,
    state: A,
    chunk_size: usize,
    offset: usize,
    dead: bool,
    _phantoms: PhantomData<&'a T>,
}

/// Streaming iterator that yields (overlapping) mutable chunks
/// of data from a strided tensor.
///
/// Chunks correspond to the abstract layout of the tensor
/// defined in a specific tensor type
/// and not to the underlying real layout of its data.
/// For instance, it will repeatedly yield the same actual
/// slices in the case of a broadcasted tensor.
///
/// `StridedIterMut` is used internally by the functions in the [`tensor`](crate::tensor)
/// module that implicitly iterate over tensors (e.g.
/// [`Tensor::for_each`](crate::tensor::Tensor::for_each)).
///
/// # Safety
/// `StridedIterMut` internally uses raw poiters to speed up
/// iteration. Proper bound checking is done to avoid potential
/// buffer overflows.
/// *This struct and its
/// implementations need proper auditing.*
pub struct StridedIterMut<'a, T: 'a, A> {
    ptr: *mut T,
    len: usize,
    bounds: A,
    strides: A,
    step_sizes: A,
    initial_state: A,
    state: A,
    chunk_size: usize,
    offset: usize,
    dead: bool,
    _phantoms: PhantomData<&'a mut T>,
}

macro_rules! inherent_impl {
    ($($iter:ident $cast_fn:ident {$($_mut:tt)?});*) => {$(
        impl<'a, T, A> $iter<'a, T, A>
        where
            A: Copy + AsRef<[usize]> + AsMut<[usize]>,
        {
            pub(crate) fn new(buffer: &'a $($_mut)? [T], size: A, stride: A, offset: A, chunk_size: usize) -> Self {
                let mut step_sizes = size;
                let mut step = chunk_size;

                for x in step_sizes.as_mut().iter_mut() {
                    if step < *x {
                        *x = 1;
                    } else {
                        step /= *x;
                    }
                }

                let linear_offset = stride.as_ref().iter().zip(offset.as_ref().iter()).map(|(x, y)| x * y).sum();

                $iter {
                    ptr: buffer.$cast_fn(),
                    len: buffer.len(),
                    bounds: size,
                    strides: stride,
                    step_sizes,
                    initial_state: offset,
                    state: offset,
                    chunk_size: chunk_size / step, // Chunks must match innermost axes.
                    offset: linear_offset,
                    dead: false,
                    _phantoms: PhantomData,
                }
            }
        }
    )*};
}

inherent_impl! {
    StridedIter as_ptr {/* no mut */};
    StridedIterMut as_mut_ptr { mut }
}

macro_rules! streaming_iterator_impl {
    ($($iter:ident $cast_fn:ident $gat:ident);*) => {$(
        impl<'a, T, A> StreamingIterator for $iter<'a, T, A>
        where
            T: 'static,
            A: Copy + AsRef<[usize]> + AsMut<[usize]>,
        {
            type Item = $gat<[T]>;

            fn next<'b>(&'b mut self) -> Option<<Self::Item as KindLifetimeType<'b>>::Applied> {
                if self.dead {
                    return None;
                }

                // Buffer overflow guard
                assert!(
                    self.offset + self.chunk_size <= self.len,
                    "Buffer overflow detected in strided iterator. Aborting!"
                );

                let chunk = unsafe {
                    let data = self.ptr.offset(self.offset as isize);
                    std::slice::$cast_fn(data, self.chunk_size)
                };

                for ((((digit, step_size), bound), stride), initial_digit) in self
                    .state.as_mut()
                    .iter_mut()
                    .zip(self.step_sizes.as_ref().iter())
                    .zip(self.bounds.as_ref().iter())
                    .zip(self.strides.as_ref().iter())
                    .zip(self.initial_state.as_ref().iter())
                {
                    if *digit - *initial_digit + step_size >= *bound {
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
    )*};
}

streaming_iterator_impl! {
    StridedIter from_raw_parts RefConstructorPA1;
    StridedIterMut from_raw_parts_mut RefMutConstructorPA1
}
