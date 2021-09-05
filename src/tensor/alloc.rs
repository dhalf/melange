//! Utility traits that define the alloction properties of buffers whether their
//! length is known at compile time or at runtime.

use super::*;
use crate::stack_buffer::Buffer;
use crate::hkt::{VecConstructor, StackBufferConstructor};
use linalg::Contiguous;

/// Unifies satic and dynamic allocation strategies.
///
/// The [`VecConstructor`] implementation disregards the constant size whereas
/// the [`ArrayConstructor`] implementation disregards the dynamic size.
pub trait Alloc<T, N>: KindTypeTypeType<T, N> {
    /// Returns a buffer filled with value of size `N` or `len`.
    ///
    /// The choice of len is up to the implementor. However, buffers that
    /// support dynamic allocation should take `len` into account whereas
    /// buffers that only support static allocation should favor `N`.
    fn fill(value: T, len: usize) -> <Self as KindTypeTypeType<T, N>>::Applied;
}

impl<T: Clone, N> Alloc<T, N> for VecConstructor {
    fn fill(value: T, len: usize) -> Vec<T> {
        vec![value; len]
    }
}

impl<T: Copy, N> Alloc<T, N> for StackBufferConstructor
where
    N: StackBuffer<[T; 1]>,
{
    fn fill(value: T, _: usize) -> <N as StackBuffer<[T; 1]>>::Buffer {
        <N as StackBuffer<[T; 1]>>::Buffer::fill(value)
    }
}

pub trait Realloc<T, N> {
    type Buffer: Alloc<T, N>;
}

impl<T, N> Realloc<T, N> for VecConstructor
where
    T: Clone,
{
    type Buffer = VecConstructor;
}

impl<T, N> Realloc<T, N> for StackBufferConstructor
where
    T: Copy,
    N: StackBuffer<[T; 1]>,
{
    type Buffer = StackBufferConstructor;
}

impl<'a, C, T, N> Realloc<T, N> for ViewConstructorPA2<'a, C>
where
    C: Alloc<T, N>,
{
    type Buffer = C;
}

impl<'a, C, T, N> Realloc<T, N> for ViewMutConstructorPA2<'a, C>
where
    C: Alloc<T, N>,
{
    type Buffer = C;
}

/// Marker trait implemented for buffer type constructors that support dynamic
/// allocation.
pub trait DynamicBuffer {}

impl DynamicBuffer for VecConstructor {}

impl<B, T, S> Tensor<B, T, S, Contiguous>
where
    B: KindTypeTypeType<T, S::Elem>,
    S: Axes,
{
    pub fn alloc(value: T, size: <S::Len as StackBuffer<[usize; 1]>>::Buffer) -> Self
    where
        B: Alloc<T, S::Elem>,
    {
        let len = size.as_ref().iter().product();
        Tensor {
            buffer: B::fill(value, len),
            offset: <S::Len as StackBuffer<[usize; 1]>>::Buffer::fill(0),
            stride: contiguous_stride::<S::Len>(&size), // fix this: contiguous_stride(size)
            size: size,
            opt_chunk_size: len,
            _phantoms: PhantomData,
        }
    }
}
