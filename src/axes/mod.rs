use std::marker::PhantomData;
use typenum::Unsigned;
use typenum::bit::B1;

pub mod static_axes;
pub mod broadcast;
pub mod stride;
pub mod block;
pub mod transpose;

pub trait True {}
impl True for B1 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ax<A, As>(pub PhantomData<*const (A, As)>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StatAx<N>(PhantomData<*const N>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynAx<T>(PhantomData<*const T>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ax0;

pub trait Axes {
    const LEN: usize;
    const ELEM: usize;
}

impl<N, As> Axes for Ax<StatAx<N>, As>
where
    N: Unsigned,
    As: Axes,
{
    const LEN: usize = As::LEN + 1;
    const ELEM: usize = As::ELEM * N::USIZE;
}

impl<As, S> Axes for Ax<DynAx<S>, As>
where
    As: Axes,
{
    const LEN: usize = As::LEN + 1;
    const ELEM: usize = 0;
}

impl Axes for Ax0 {
    const LEN: usize = 0;
    const ELEM: usize = 1;
}

// /// Utility function similar to `Vec::collect`.
// ///
// /// # Examples
// /// ```
// /// # use melange::axes::*;
// /// assert_eq!(collect([1, 2, 3].iter().map(|x| x * x)), [1, 4, 9]);
// /// ```
// pub fn collect<'a, T, I, const N: usize>(iter: I) -> [T; N]
// where
//     T: Copy + Default + 'a,
//     I: Iterator<Item = T>,
// {
//     let mut res = [T::default(); N];
//     for (r, x) in res.iter_mut().zip(iter) {
//         *r = x;
//     }
//     res
// }

// /// Computes the contiguous stride associated with the given size.
// ///
// /// The contiguous stride is the total stride of a constiguous tensor that has
// /// the given size (refer to [`total_stride`]). A tensor is contiguous if it is
// /// not a broadcasted, strided, sub- or transposed view on another tensor.
// ///
// /// The contiguous stride corresponds with the amount of shift in the linear buffer
// /// that contains the contiguous tensor's data when incrementing a coordianate by
// /// the unit. For instance the following 2x2 contiguous matrix `[[1, 2], [3, 4]]`
// /// is represented in memory as [1, 3, 2, 4] (column major storage).
// /// To access the element located at coordinates `[i, j]`, the inner product of
// /// the coordinates and of the total stride `[1, 2]` is computed which is
// /// `i + 2 * j`. The element at `[1, 0]` is indeed the elemnt at index 1 in the
// /// buffer.
// ///
// /// When the tensor is not contiguous the total stride can be obtained from
// /// the size and partial stride instead using [`total_stride`].
// ///
// /// # Examples
// /// ```
// /// # use melange::axes::*;
// /// assert_eq!(contiguous_stride([3, 2, 4]), [1, 3, 6]);
// /// ```
// pub fn contiguous_stride<const N: usize>(size: [usize; N]) -> [usize; N] {
//     let mut stride = [0; N];
//     stride[0] = 1;
//     for i in 1..N {
//         stride[i] = stride[i - 1] * size[i - 1];
//     }
//     stride
// }

// /// Computes the partial stride associated with the given size and total stride.
// ///
// /// The partial stride is defined such that the following holds:
// /// For all valid axis index `i`, `total[i] = partial[i] * contiguous[i]`.
// /// The contiguous stride depends on the size only and can be obtained with
// /// [`contiguous_stride`].
// ///
// /// The total stride corresponds with the amount of shift in the linear buffer
// /// that contains the tensor's data when incrementing a coordianate by the unit
// /// (refer to [`total_stride`]).
// ///
// /// When the tensor is contiguous (i.e. not a broadcasted, strided, sub- or
// /// transposed view on another tensor), the total and contiguous strides are
// /// the same and the partial stride is a vector of ones.
// ///
// /// When the tensor is not contiguous the partial stride accounts for the
// /// modifications that should be applied to the contiguous strides to obtain
// /// the right amount of shift in the underlying buffer. Consider the following
// /// contiguous 1x2 matrix and a broadcasted 2x2 view of it: `[[1, 2]]` and
// /// `[[1, 2], [1, 2]]`. Their three stride are given in the table below:
// /// `matrix | total | partial | contiguous
// ///  1x2    | [1, 1]| [1, 1]  | [1, 1]
// ///  2x2    | [1, 0]| [1, 0]  | [1, 1]`
// ///
// /// # Examples
// /// ```
// /// # use melange::axes::*;
// /// assert_eq!(partial_stride([3, 2, 4, 1], [1, 6, 18, 0]), [1, 2, 3, 0]);
// /// ```
// pub fn partial_stride<const N: usize>(size: [usize; N], total: [usize; N]) -> [usize; N] {
//     collect(
//         contiguous_stride(size)
//             .iter()
//             .zip(total.iter())
//             .map(|(&s, &z)| z / s),
//     )
// }

// /// Computes the total stride associated with the given size and partial stride.
// ///
// /// This stride is correspond with the amount of shift in the linear buffer
// /// that contains the tensor's data when incrementing a coordianate by the unit.
// /// In the case of contiguous tensors it is equal to the [`contiguous_stride`].
// /// In the case of non-contiguous tensors, it captures the modifications dictated
// /// by the [`partial_stride`].
// ///
// /// # Examples
// /// ```
// /// # use melange::axes::*;
// /// assert_eq!(total_stride([3, 2, 4, 1], [1, 2, 3, 0]), [1, 6, 18, 0]);
// /// ```
// pub fn total_stride<const N: usize>(size: [usize; N], partial: [usize; N]) -> [usize; N] {
//     collect(
//         contiguous_stride(size)
//         .iter()
//         .zip(partial.iter())
//         .map(|(&s, &z)| s * z)
//     )
// }

// /// Computes the optimal (maximum) chunk size that can be used with a tensor
// /// of the given size and total strides.
// ///
// /// The optimal chunk size of a tensor is the size (in number of elements) of
// /// the biggest contiguous portion in the linear buffer that represents the
// /// tensor in memory that contains adjacent elements. Striding and broadcasting
// /// operations typically change the optimal chunk size as adjacent element in
// /// the resulting tensor are not necessarily adjacent in memory anymore. The
// /// optimal chunk size is of paramount importance to implement
// /// [`StridedIter`](crate::iter::strided_iter::StridedIter)
// /// and [`StridedIterMut`](crate::iter::strided_iter::StridedIterMut) that are
// /// the primary mean to iterate over tensors by taking striding and broadcasting
// /// into account.
// ///
// /// # Examples
// /// ```
// /// # use melange::axes::*;
// /// // A contiguous tensor of size 3 x 2 x 4 x 1 has total strides 1, 3, 6, 24.
// /// // Here the tensor is strided along its third dim (12 = 6 * 2), thus maximum
// /// // contiguous blocks have size 3 x 2 and contain 6 elements.
// /// assert_eq!(opt_chunk_size([3, 2, 4, 1], [1, 3, 12, 24]), 6);
// /// ```
// pub fn opt_chunk_size<const N: usize>(size: [usize; N], total: [usize; N]) -> usize {
//     contiguous_stride(size)
//         .iter()
//         .zip(total.iter())
//         .skip_while(|(&s, &z)| s == z)
//         .map(|(&s, _)| s)
//         .nth(0)
//         .unwrap_or(size.iter().product())
// }

// /// Takes an array of size N and returns an array of size M obtained by cutting
// /// or padding the input array with the given padding value.
// pub fn pad<T: Copy, const N: usize, const M: usize>(arr: [T; N], padding_value: T) -> [T; M] {
//     let mut padded = [padding_value; M];
//     for (x, &y) in padded.iter_mut().zip(arr.iter()) {
//         *x = y
//     }
//     padded
// }
