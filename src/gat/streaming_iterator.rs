use super::*;

/// Trait implemented by iterators that adapt the lifetime of the
/// items they yield to their usage.
///
/// This allows the implementation of iterators that would be
/// impossible to express with the classical Iterator trait
/// such as iterators that yield overlapping mutable references.
///
/// Note that because streaming iterators don't implement
/// the IntoIterator trait (that would defeat their purpose),
/// they cannot be used with for loops. Please use while let
/// loops instead (see examples).
///
/// StreamingIterator is automatically implemented on
/// all types that implement Iterator. This allows
/// easy zipping with the two kinds of iterators by using
/// the [`streaming_zip`](StreamingIterator::streaming_zip)
/// provided function.
///
/// # Examples
/// Iterating over overlapping mutable references becomes possible.
/// ```no_run
/// use melange::prelude::*;
/// use melange::gat::StreamingIterator;
/// use typenum::{U1, U2};
///
/// let mut a: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![1, 2]).unwrap();
/// let mut a = BroadcastMut::<Shape2D<U2, U2>>::broadcast_mut(&mut a);
/// let mut streaming_iteartor = a.strided_iter_mut(2);
///
/// while let Some(chunk) = streaming_iteartor.next() {
///     println!("{:?}", chunk);
/// }
/// ```
///
/// But borrowing rules are still enforced properly.
/// ```compile_fail
/// # use melange::prelude::*;
/// # use melange::gat::StreamingIterator;
/// # use typenum::{U1, U2};
/// #
/// # let mut a: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![1, 2]).unwrap();
/// # let mut a = a.broadcast_mut::<Shape2D<U2, U2>>();
/// # let mut streaming_iteartor = a.strided_iter_mut(2);
/// #
/// let r1 = streaming_iteartor.next();
/// let r2 = streaming_iteartor.next();
///
/// println!("{:?}", r1);
/// ```
pub trait StreamingIterator {
    type Item: for<'b> Gat<'b>;
    /// Similar to Iterator's next function.
    ///
    /// Needs to be explicitly called in while let loops.
    /// See trait-level documentation.
    fn next<'a>(&'a mut self) -> Option<<Self::Item as Gat<'a>>::Output>;
    /// Similar to Iterator's zip function.
    fn streaming_zip<U>(self, other: U) -> StreamingZip<Self, U>
    where
        Self: Sized,
        U: StreamingIterator,
    {
        StreamingZip {
            iter_a: self,
            iter_b: other,
        }
    }
}

/// Ouput of [`StreamingIterator::streaming_zip`](StreamingIterator::streaming_zip).
pub struct StreamingZip<A, B> {
    iter_a: A,
    iter_b: B,
}

impl<A, B> StreamingIterator for StreamingZip<A, B>
where
    A: StreamingIterator,
    B: StreamingIterator,
{
    type Item = (A::Item, B::Item);
    #[inline]
    fn next<'a>(&'a mut self) -> Option<<Self::Item as Gat<'a>>::Output> {
        let item_a = self.iter_a.next()?;
        let item_b = self.iter_b.next()?;
        Some((item_a, item_b))
    }
}

impl<I> StreamingIterator for I
where
    Self: Iterator,
{
    type Item = NoGat<<Self as Iterator>::Item>;
    fn next<'a>(&'a mut self) -> Option<<Self::Item as Gat<'a>>::Output> {
        let item = self.next()?;
        Some(item)
    }
}
