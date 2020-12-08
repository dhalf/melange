//! Provides a way to simulate Generic Associated Types
//! (GATs) in stable Rust.
//! 
//! Thanks to the richness of Rust's type system
//! (Turing completeness), it is possible to simulate
//! Higher Kinded Types (HKT) that are a superset of GATs
//! just by using type operators (traits that act like
//! "functions" on types) à la [`typenum`]. This idea
//! is developed in [`this blogpost`].
//! 
//! GATs are ideed a specific subgroup of HKTs that consume
//! a lifetime and output a type. This can be denoted as
//! '* -> * in a Haskel-like fashion where '* represents
//! a lifetime.
//! 
//! The approach implemented in this module relies on three
//! main pieces.
//! 1. Given a type resulting from the application of a lifetime,
//! to a GAT, we can obtain another type corresponding to the
//! application of another lifetime just by using the [`Gat`](Gat)
//! type operator.
//! 2. We still miss a way to talk about the HKT itself (unapplied).
//! This cannot be achieved with a special lifetime we would apply
//! to the HKT because we cannot create lifetimes from nothing in Rust.
//! However, we can create some specific structs that represent the
//! HKTs we wish to use at the type level (kind lowering).
//! 3. We also need a way to enforce that Gat is implemented on
//! any input GAT for any lifetime. This is possible with current
//! HRTBs since we only care about lifetimes here.
//! 
//! The conjunction of those three ideas makes the simulation
//! of GAT work and also makes it usable in practice.
//! 
//! Some stadard HKTs are defined in this module:
//! * RefGat<T> that represents &'* T,
//! * RefMutGat<T> that represents &'* mut T,
//! * NoGat<T> that represents T (lifetime invariant).
//! 
//! One of the biggest motivation for the GAT feature in
//! Rust over the past years has been the ability to write
//! streaming iterators that are a class of iterators that
//! yield their item with adaptive lifetimes dictated
//! by the actual usage of these items. This contrasts with
//! classical iterators that yield items with a lifetime
//! corresponding to their own lifetime.
//! 
//! One direct consequence of this is that it is notoriously
//! difficult to write iterators that yield overlapping
//! mutable references in Rust. Because those references all
//! have the same lifetime and alias each other, this constitutes
//! a violation of the borrowing rules.
//! 
//! We here propose an implementation of streaming iterators that
//! use GATs to adapt the lifetime of yielded items.
//! 
//! [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
//! [`this blogpost`]: https://gist.github.com/edmundsmith/855fcf0cb35dd467c29a9350481f0ecf
use std::marker::PhantomData;

#[doc(hidden)]
pub mod streaming_iterator;

#[doc(inline)]
pub use streaming_iterator::StreamingIterator;

/// Type operator that changes the lifetime applied to a GAT.
/// See module documentation.
pub trait Gat<'rhs> {
    /// Output type.
    type Output;
}

/// Representation of the immutable reference GAT given some type `T`
/// at the type level.
pub struct RefGat<T: ?Sized + 'static> {
    _phantoms: PhantomData<&'static T>
}

/// Representation of the mutable reference GAT given some type `T`
/// at the type level.
pub struct RefMutGat<T: ?Sized + 'static> {
    _phantoms: PhantomData<&'static mut T>
}

/// Representation of the lifetime invariant GAT given some type `T`
/// at the type level.
/// 
/// Used to implement StreamingIterators on types that implement
/// Iterator. To keep the same behavior in the streaming version,
/// lifetime adaptation needs not be performed at all.
pub struct NoGat<T>(T);

impl<'rhs, T> Gat<'rhs> for RefGat<T>
where
    T: ?Sized + 'rhs,
{
    type Output = &'rhs T;
}

impl<'rhs, T> Gat<'rhs> for RefMutGat<T>
where
    T: ?Sized + 'rhs,
{
    type Output = &'rhs mut T;
}

impl<'rhs, T> Gat<'rhs> for NoGat<T> {
    type Output = T;
}

impl<'rhs, A, B> Gat<'rhs> for (A, B)
where
    A: Gat<'rhs>,
    B: Gat<'rhs>,
{
    type Output = (<A as Gat<'rhs>>::Output, <B as Gat<'rhs>>::Output);
}
