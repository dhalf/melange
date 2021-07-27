//! Provides a way to simulate Higher Kinded Types (HKTs)
//! in stable Rust (and provides better GATs in nightly!)
//!
//! Thanks to the richness of Rust's type system
//! (Turing completeness), it is possible to simulate Higher Kinded Types (HKT) 
//! by just using type operators (traits that act like
//! "functions" on types) à la [`typenum`]. This idea
//! is developed in [`this blogpost`].
//!
//! As Rust lacks a way to manipulate unapplied type constructors (i.e.
//! kinded types), we rely on lowered representations (types of kind `*` that
//! simulate HKTs) that can be applied later on by using type operators (i.e.
//! traits that act like type-level functions thanks to associated types).
//!
//! This module contains two "kind" traits to be used with constructors of kind
//! `'* -> *` and `* -> * -> *` or any constructor that has these kinds
//! when partially applied. Partial application is simulated by using generic
//! types as lowered representation. The first kind that is useful to manipulate
//! references of a variable lifetime corresponds with the `KindLifetimeType`
//! trait. The second, implemented through the `KindTypeTypeType` trait is useful
//! to represent staically-sized buffers (with type level integers, see [`typenum`]).
//! 
//! One limitation is the lack of Higher Ranked Trait Bounds (HRTBs) working
//! with types or constants that makes HKT parametrized by types or constants
//! less uasable in practice. For instance, using such a kinded type constructor
//! as associated type in a trait forces the trait declaration to have the same
//! parameter. Type constructors parametrized by lifetimes only such as `'* -> *`
//! do not have this limitation as lifetime-based HRTBs are already implemented.
//!
//! We define some stadard HKTs in this module:
//! * RefConstructorPA1<T> that represents a partially applied reference
//! constructor `'a -> &'a T`,
//! * RefMutConstructorPA1<T> that represents a partially applied mutable reference
//! constructor `'a -> &'a mut T`,
//! * LifetimeInvariantPA1<T> that represents a partially applied lifetime invariant
//! constructor `'a -> T`,
//! * VecConstructor that represent a (statically-sized) `Vec` constructor
//! `T, N -> Vec<T>`,
//! * StackBufferConstructor that represent a (statically-sized) stack allocated
//! buffer constructor. This may be thought of as `T, N -> [T; N]` as the resukting
//! buffer shares it representation with `[T; N]`,
//!
//! One of the biggest motivation for the GAT (Generic Associated
//! Types, HKTs used as associated types) feature in
//! Rust over the past years has been the ability to write
//! streaming iterators that are a class of iterators that
//! yield their items with adaptive lifetimes dictated
//! by the actual usage of these items. This contrasts with
//! classical iterators that yield items with a lifetime
//! corresponding to the iterator's own lifetime.
//!
//! One direct consequence is that it is notoriously
//! difficult to write iterators that yield overlapping
//! mutable references in Rust. Because these references all
//! have the same lifetime and alias each other which constitutes
//! a violation of the borrowing rules.
//!
//! We here propose an implementation of
//! [`StreamingIterator`](crate::iter::StreamingIterator)s that
//! use HKTs to adapt the lifetime of yielded items.
//!
//! Note: we could have used the nightly GAT feature instead but
//! the parser does not support lifetimes in trait bounds yet:
//! `T: StreamingIterator<Item<'a> = &'a T>` does not compile while
//! our implementation allows us to do so.
//!
//! [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
//! [`this blogpost`]: https://gist.github.com/edmundsmith/855fcf0cb35dd467c29a9350481f0ecf
use std::marker::PhantomData;
use crate::stack_buffer::StackBuffer;

/// Type operator that simulates kind `'* -> *`.
///
/// See module-level documentation.
pub trait KindLifetimeType<'a> {
    /// Output type.
    type Applied;
}

/// Type operator that simulates kind `* -> * -> *`.
///
/// See module-level documentation.
pub trait KindTypeTypeType<T: ?Sized, N> {
    /// Output type.
    type Applied;
}

/// Lowered partially applied immutable reference HKT.
///
/// `Ref :: * -> '* -> *`
#[derive(Debug)]
pub struct RefConstructorPA1<T: ?Sized + 'static>(PhantomData<&'static T>);

impl<'a, T> KindLifetimeType<'a> for RefConstructorPA1<T>
where
    T: ?Sized + 'a,
{
    type Applied = &'a T;
}

/// Lowered partially applied mutable reference HKT.
///
/// `RefMut :: * -> '* -> *`
#[derive(Debug)]
pub struct RefMutConstructorPA1<T: ?Sized + 'static>(PhantomData<&'static mut T>);

impl<'a, T> KindLifetimeType<'a> for RefMutConstructorPA1<T>
where
    T: ?Sized + 'a,
{
    type Applied = &'a mut T;
}

/// Lowered partially applied lifetime invariant HKT.
///
/// `LifetimeInvariant :: * -> '* -> *`
///
/// Used to implement StreamingIterators on types that already implement
/// Iterator. To keep the same behavior in the streaming version,
/// lifetime adaptation needs not be performed at all.
#[derive(Debug)]
pub struct LifetimeInvariantPA1<T>(PhantomData<*const T>);

impl<'a, T> KindLifetimeType<'a> for LifetimeInvariantPA1<T> {
    type Applied = T;
}

/// Lowered Vec HKT.
///
/// `Vec :: * -> * -> *`
///
/// Vec is viewed as a constructor of kind `* -> const usize -> *`
/// that disregards its second argument.
#[derive(Debug)]
pub struct VecConstructor;

impl<T, N> KindTypeTypeType<T, N> for VecConstructor {
    type Applied = Vec<T>;
}

/// Lowered StackBuffer HKT.
///
/// `StackBuffer :: * -> * -> *`
#[derive(Debug)]
pub struct StackBufferConstructor;

impl<T: Copy, N> KindTypeTypeType<T, N> for StackBufferConstructor
where
    N: StackBuffer<[T; 1]>,
{
    type Applied = <N as StackBuffer<[T; 1]>>::Buffer;
}

/// Lowered partially applied twice immutable view (slice that remembers the type of
/// the referenced data) HKT.
///
/// `View :: (* -> * -> *) -> '* -> (* -> * -> *)`
#[derive(Debug)]
pub struct ViewConstructorPA2<'a, C>(PhantomData<(*const C, &'a ())>);

impl<'a, C, T: 'a, N> KindTypeTypeType<T, N> for ViewConstructorPA2<'a, C> {
    type Applied = &'a [T];
}

/// Lowered partially applied twice mutable view (slice that remembers the type of
/// the referenced data) HKT.
///
/// `View :: (* -> * -> *) -> '* -> (* -> * -> *)`
#[derive(Debug)]
pub struct ViewMutConstructorPA2<'a, C>(PhantomData<(*const C, &'a ())>);

impl<'a, C, T: 'a, N> KindTypeTypeType<T, N> for ViewMutConstructorPA2<'a, C> {
    type Applied = &'a mut [T];
}

// Pairs of `KindLifetimeType`s are `KindLifetimeType`s.
impl<'a, A, B> KindLifetimeType<'a> for (A, B)
where
    A: KindLifetimeType<'a>,
    B: KindLifetimeType<'a>,
{
    type Applied = (<A as KindLifetimeType<'a>>::Applied, <B as KindLifetimeType<'a>>::Applied);
}
