//! Provides the [`StreamingIterator`] trait that alows for expressing iterators
//! that yield items whose lifetime adapt to the context. See trait-level doc.

use crate::hkt::KindLifetimeType;

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
/// # #![allow(incomplete_features)]
/// # #![feature(const_generics)]
/// # #![feature(const_evaluatable_checked)]
/// # #![feature(never_type)]
/// use melange::prelude::*;
/// use melange::iter::StreamingIterator;
/// use std::convert::TryFrom;
/// 
/// let mut a: VecTensor2D<f64, 2, 1> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
/// let mut view = (&mut a).broadcast(Size2D::<{[2, 2]}>::new());
/// let mut streaming_iterator = view.chunks_mut(2);
///
/// while let Some(chunk) = streaming_iterator.next() {
///     println!("{:?}", chunk);
/// }
/// ```
///
/// But borrowing rules are still enforced properly.
/// ```compile_fail
/// # #![allow(incomplete_features)]
/// # #![feature(const_generics)]
/// # #![feature(const_evaluatable_checked)]
/// # #![feature(never_type)]
/// # use melange::prelude::*;
/// # use melange::iter::StreamingIterator;
/// # use std::convert::TryFrom;
/// # 
/// # let mut a: VecTensor2D<f64, 2, 1> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
/// # let mut view = (&mut a).broadcast(Size2D::<{[2, 2]}>::new());
/// # let mut streaming_iterator = view.chunks_mut(2);
/// #
/// let r1 = streaming_iterator.next();
/// let r2 = streaming_iterator.next();
///
/// println!("{:?}", r1);
/// ```
pub trait StreamingIterator {
    /// Type of yielded items.
    ///
    /// `Item` must be an instance of Gat to allow for lifetime adaptation.
    type Item: for<'b> KindLifetimeType<'b>;
    /// Similar to Iterator's next function.
    ///
    /// Needs to be explicitly called in while let loops.
    /// See trait-level documentation.
    fn next<'a>(&'a mut self) -> Option<<Self::Item as KindLifetimeType<'a>>::Applied>;
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
    fn next<'a>(&'a mut self) -> Option<<Self::Item as KindLifetimeType<'a>>::Applied> {
        let item_a = self.iter_a.next()?;
        let item_b = self.iter_b.next()?;
        Some((item_a, item_b))
    }
}
