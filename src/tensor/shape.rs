//! Contains all the tools to complement [`typenum`]
//! crate and efficiently use it for shape "arithmetics".
//!
//! As in [`typenum`], this module contains two kinds of unsafe traits:
//! * type operators that act like type-level functions on type-level entities,
//! * marker traits that provide functions and constants to interact with type-level
//! entities at runtime.
//!
//! Type operators all share the `Output` associated
//! type that contains the type-level result of the operation the
//! trait represents.
//!
//! [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html

use std::ops::*;
use typenum::operator_aliases::*;
use typenum::type_operators::*;
use typenum::private::InternalMarker;
use typenum::{ATerm, Bit, TArr, UInt, Unsigned, B0, B1, U0, U1, Equal, Less, UTerm};

/// Utility function that computes the intrinsic strides of the given `shape`.
///
/// Intrinsic strides are the vector containing the consecutive products of
/// the shape's axes in reverse order, starting from 1 and excluding the
/// last product (number of elements). Formally:
///
/// # Examples
///
/// ```
/// use melange::tensor::shape::intrinsic_strides_in_place;
///
/// let a = vec![32, 3, 16, 16];
/// let b = intrinsic_strides_in_place(a);
///
/// assert_eq!(b, vec![768, 256, 16, 1]);
/// ```
pub fn intrinsic_strides_in_place(mut shape: Vec<usize>) -> Vec<usize> {
    let mut product = 1;
    for stride in shape.iter_mut().rev() {
        let tmp = product;
        product *= *stride;
        *stride = tmp;
    }

    shape
}

/// This trait "aliases" [`B1`] (type-level bit one) for use in trait bounds.
///
/// It is especially useful with type-level binary operators.
///
/// # Examples
///
/// ```no_run
/// use melange::tensor::shape::{Same, TRUE};
///
/// fn bar<S, Z>()
/// where
///     S: Same<Z>, // This bound is required by the following line
///     <S as Same<Z>>::Output: TRUE // Constrains "S Same Z" to hold (Output = B1)
/// {
///     // some code
/// }
/// ```
///
/// [`B1`]: https://docs.rs/typenum/1.12.0/typenum/bit/struct.B1.html
pub unsafe trait TRUE {}
unsafe impl TRUE for B1 {}

/// Zero-sized struct representing type-level dynamic dimension.
/// 
/// `Dyn` can be compared with type-level integers, it is strictly
/// smaller that all of them and thus only equal to itself. This
/// is a security driven design choice that prevents static level
/// coercions i.e. allowing a `Dyn` in the input to become a static
/// dimension in the output. Being smaller eases the definition of some
/// static checks such as whether some shape fits into another useful
/// for subviewsof tensors: no static dimension can fit in `Dyn`
/// (coercion prevented) but `Dyn` can bit in any type-level dimension.
/// 
/// `Dyn` also implements basic ops (at type-level). It is infectious
/// as any type-level op involving `Dyn` results in `Dyn`. This property
/// as well constitutes a security guarantee.
#[derive(Debug, PartialEq)]
pub struct Dyn;
impl Cmp<Dyn> for Dyn {
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &Dyn) -> Self::Output {
        Equal
    }
}
impl<U, B> Cmp<Dyn> for UInt<U, B> {
    type Output = Less;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &Dyn) -> Self::Output {
        Less
    }
}
impl<U, B> Cmp<UInt<U, B>> for Dyn {
    type Output = Less;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &UInt<U, B>) -> Self::Output {
        Less
    }
}

macro_rules! ops_impl_dyn {
    ($($trait:ident, $fn:ident);*) => {$(
        impl $trait<Dyn> for Dyn {
            type Output = Dyn;
        
            #[inline]
            fn $fn(self, _: Dyn) -> Self::Output {
                Dyn
            }
        }
        impl<U, B> $trait<UInt<U, B>> for Dyn {
            type Output = Dyn;
        
            #[inline]
            fn $fn(self, _: UInt<U, B>) -> Self::Output {
                Dyn
            }
        }
        impl<U, B> $trait<Dyn> for UInt<U, B> {
            type Output = Dyn;
        
            #[inline]
            fn $fn(self, _: Dyn) -> Self::Output {
                Dyn
            }
        }
    )*};
}

ops_impl_dyn! {
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div;
    Rem, rem
}

/// Marker trait that provides a runtime equality check function
/// for type-level dimensions.
///
/// It defines the `runtime_eq` function that takes a runtime unsigned
/// iteger as parameter and checks it against the type-level integer.
///
/// The implementation for [`UInt`] relies on [`typenum`]'s [`Unsigned`]
/// trait. The implementation for [`Dyn`](Dyn) always returns `true`.
///
/// # Examples
/// ```
/// use typenum::{U1, U2};
/// use melange::tensor::shape::{Dim, Dyn};
///
/// assert!(U1::runtime_eq(1));
/// assert!(Dyn::runtime_eq(1));
/// assert!(!U2::runtime_eq(1));
/// ```
///
/// [`UInt`]: https://docs.rs/typenum/1.12.0/typenum/uint/struct.UInt.html
/// [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
/// [`Unsigned`]: https://docs.rs/typenum/1.12.0/typenum/marker_traits/trait.Unsigned.html
pub unsafe trait Dim {
    /// Checks the equality of a type-level unsigned integer
    /// and a runtime unsigned integer.
    ///
    /// See trait-level documentation.
    fn runtime_eq(dim: usize) -> bool;
}

unsafe impl<U, B> Dim for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    fn runtime_eq(dim: usize) -> bool {
        Self::to_usize() == dim
    }
}

unsafe impl Dim for UTerm {
    fn runtime_eq(dim: usize) -> bool {
        dim == 0
    }
}

unsafe impl Dim for Dyn {
    fn runtime_eq(_dim: usize) -> bool {
        true
    }
}

/// Marker trait implemented on type-level unsigned integers.
///
/// Types that implement this trait are the only valid type-level
/// dimensions for a [static shape](StaticShape).
pub unsafe trait StaticDim: Dim + Unsigned {}
unsafe impl<U, B> StaticDim for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
}
unsafe impl StaticDim for UTerm {}

/// Marker trait that provides basic ways to interact with type-level shapes.
///
/// It is implemented on [`typenum`]'s [`TArr`] containing a collection
/// of type-level unsigned integers or [`Dyn`](Dyn).
///
/// [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
/// [`TArr`]: https://docs.rs/typenum/1.12.0/typenum/array/struct.TArr.html
pub unsafe trait Shape {
    /// Type-level number of axes in the shape, i.e. order of the tensor.
    ///
    /// It implements [`typenum`]'s [`Unsigned`] trait.
    ///
    /// # Examples
    /// ```
    /// use typenum::{U1, U2, Unsigned};
    /// use melange::tensor::shape::{Shape, Shape4D};
    ///
    /// assert_eq!(<Shape4D<U1, U2, U2, U2> as Shape>::Len::USIZE, 4);
    /// ```
    ///
    /// [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
    /// [`Unsigned`]: https://docs.rs/typenum/1.12.0/typenum/marker_traits/trait.Unsigned.html
    type Len: Unsigned;

    /// Checks the given slice against the implementor type-level shape for
    /// compatibility.
    ///
    /// This is the shape equivalent of [`runtime_eq`](Dim::runtime_eq) for
    /// dimensions and it uses this function under the hood on each axis.
    ///
    /// # Examples
    /// ```
    /// use typenum::{U1, U2, Unsigned};
    /// use melange::tensor::shape::{Shape, Shape4D};
    ///
    /// assert!(<Shape4D<U1, U2, U2, U2> as Shape>::runtime_compat(&[1, 2, 2, 2]));
    /// ```
    fn runtime_compat(shape: &[usize]) -> bool;
}

unsafe impl Shape for ATerm {
    type Len = U0;

    fn runtime_compat(shape: &[usize]) -> bool {
        shape.len() == 0
    }
}

unsafe impl<D, A> Shape for TArr<D, A>
where
    A: Shape,
    A::Len: Add<B1>,
    D: Dim,
    Add1<A::Len>: Unsigned,
{
    type Len = Add1<A::Len>;

    fn runtime_compat(shape: &[usize]) -> bool {
        Self::Len::USIZE == shape.len()
            && D::runtime_eq(shape[A::Len::USIZE])
            && A::runtime_compat(&shape[..A::Len::USIZE])
    }
}
/// Marker trait that provides copy of static dimensions
/// from a type-level shape to a runtime shape.
///
/// This is useful in dynamic [`reductions`](crate::tensor::reduction).
pub unsafe trait PartialCopy: Shape {
    /// Copies type-level static dimension at the same location
    /// in the given shape.
    ///
    /// Indices corresponding to a type-level `Dyn` are left unchanged.
    ///
    /// # Examples
    /// ```
    /// use typenum::{U1, U2, Unsigned};
    /// use melange::tensor::shape::{Dyn, PartialCopy, Shape4D};
    ///
    /// let mut shape = vec![6, 4, 4, 2];
    /// <Shape4D<U1, Dyn, Dyn, U2> as PartialCopy>::partial_copy(&mut shape);
    /// assert_eq!(shape, vec![1, 4, 4, 2]);
    /// ```
    fn partial_copy(shape: &mut [usize]);
}

unsafe impl PartialCopy for ATerm {
    fn partial_copy(_shape: &mut [usize]) {}
}

unsafe impl<D, A> PartialCopy for TArr<D, A>
where
    A: PartialCopy,
    A::Len: Add<B1>,
    Add1<A::Len>: Unsigned,
    D: StaticDim,
{
    fn partial_copy(shape: &mut [usize]) {
        shape[A::Len::USIZE] = D::USIZE;
        A::partial_copy(shape);
    }
}

unsafe impl<A> PartialCopy for TArr<Dyn, A>
where
    A: PartialCopy,
    A::Len: Add<B1>,
    Add1<A::Len>: Unsigned,
{
    fn partial_copy(shape: &mut [usize]) {
        A::partial_copy(shape);
    }
}

/// Marker trait providing further ways to interact with static shapes.
///
/// Implemented on shapes containing type-level unsigned integers only.
pub unsafe trait StaticShape: Shape {
    /// Type-level number of elements in the tensor, i.e. product of all dimensions
    /// of the shape.
    ///
    /// It implements [`typenum`]'s [`Unsigned`] trait.
    ///
    /// # Examples
    /// ```
    /// use typenum::{U1, U2, Unsigned};
    /// use melange::tensor::shape::{StaticShape, Shape4D};
    ///
    /// assert_eq!(<Shape4D<U1, U2, U2, U2> as StaticShape>::NumElements::USIZE, 8);
    /// ```
    ///
    /// [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html
    /// [`Unsigned`]: https://docs.rs/typenum/1.12.0/typenum/marker_traits/trait.Unsigned.html
    type NumElements: Unsigned;

    /// Outputs a [`Vec`](std::vec::Vec) containing the runtime version of the shape.
    ///
    /// # Examples
    /// ```
    /// use typenum::{U1, U2, Unsigned};
    /// use melange::tensor::shape::{StaticShape, Shape4D};
    ///
    /// assert_eq!(<Shape4D<U1, U2, U2, U2> as StaticShape>::to_vec(), vec![1, 2, 2, 2]);
    /// ```
    fn to_vec() -> Vec<usize>;

    /// Outputs a `Vec` containing the intrinsic strides of the shape.
    ///
    /// Note that these strides do not account for the real layout: see
    /// [`intrinsic_strides_in_place`](intrinsic_strides_in_place).
    /// # Examples
    /// ```
    /// use typenum::{U1, U2, Unsigned};
    /// use melange::tensor::shape::{StaticShape, Shape4D};
    ///
    /// assert_eq!(<Shape4D<U1, U2, U2, U2> as StaticShape>::strides(), vec![8, 4, 2, 1]);
    /// ```
    fn strides() -> Vec<usize>;
}

unsafe impl StaticShape for ATerm {
    type NumElements = U1;

    #[inline]
    fn to_vec() -> Vec<usize> {
        Vec::new()
    }

    #[inline]
    fn strides() -> Vec<usize> {
        Vec::new()
    }
}

unsafe impl<D, A> StaticShape for TArr<D, A>
where
    A: StaticShape,
    A::Len: Add<B1>,
    Add1<A::Len>: Unsigned,
    D: StaticDim + Mul<A::NumElements>,
    Prod<D, A::NumElements>: Unsigned,
{
    type NumElements = Prod<D, A::NumElements>;

    #[inline]
    fn to_vec() -> Vec<usize> {
        let mut vec = A::to_vec();
        vec.push(D::USIZE);

        vec
    }

    #[inline]
    fn strides() -> Vec<usize> {
        intrinsic_strides_in_place(Self::to_vec())
    }
}

/// Binary type operator that checks the compatibility of two type-level
/// shapes or two type-level dimensions.
///
/// Outputs B1 if the `Self` is compatible with `Rhs`.
/// 
/// Dimensions are compatible if they are equal or
/// at least one of them is [`Dyn`](Dyn).
/// 
/// Shapes are compatible if all their dimensions are.
/// Note that this implies compatible shapes have the
/// same length.
///
/// # Examples
/// Considering the following function and some type `Foo<T>`:
/// ```no_run
/// use melange::tensor::shape::{Same, TRUE};
/// # use std::marker::PhantomData;
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// where
///     S: Same<Z>, // This bound is required by the following line
///     <S as Same<Z>>::Output: TRUE // Constrains "S Same Z" to hold (Output = B1)
/// {
///     // some code
/// }
/// ```
///
/// This compiles:
/// ```no_run
/// use typenum::U1;
/// use melange::tensor::shape::Shape2D;
/// # use melange::tensor::shape::{Same, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// # where
/// #     S: Same<Z>,
/// #     <S as Same<Z>>::Output: TRUE
/// # {}
/// #
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// let a: Foo<Shape2D<U1, U1>> = Foo::new();
/// let b: Foo<Shape2D<U1, U1>> = Foo::new();
/// bar(a, b);
/// ```
/// While this doesn't:
/// ```compile_fail
/// use typenum::{U1, U2};
/// use melange::tensor::shape::Shape2D;
/// # use melange::tensor::shape::{Same, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// # where
/// #     S: Same<Z>,
/// #     <S as Same<Z>>::Output: TRUE
/// # {}
/// #
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// let a: Foo<Shape2D<U1, U1>> = Foo::new();
/// let b: Foo<Shape2D<U1, U2>> = Foo::new();
/// bar(a, b);
/// ```
pub unsafe trait Same<Rhs> {
    /// Output type.
    type Output;
}

unsafe impl Same<Dyn> for Dyn {
    type Output = B1;
}

unsafe impl<U, B> Same<Dyn> for UInt<U, B> {
    type Output = B1;
}

unsafe impl Same<Dyn> for UTerm {
    type Output = B1;
}

unsafe impl<U, B> Same<UInt<U, B>> for Dyn {
    type Output = B1;
}

unsafe impl Same<UTerm> for Dyn {
    type Output = B1;
}

unsafe impl<U, B, Urhs, Brhs> Same<UInt<Urhs, Brhs>> for UInt<U, B>
where
    Self: IsEqual<UInt<Urhs, Brhs>>,
{
    type Output = Eq<Self, UInt<Urhs, Brhs>>;
}

unsafe impl<U, B> Same<UTerm> for UInt<U, B>
where
    Self: IsEqual<UTerm>,
{
    type Output = Eq<Self, UTerm>;
}

unsafe impl<Urhs, Brhs> Same<UInt<Urhs, Brhs>> for UTerm
where
    Self: IsEqual<UInt<Urhs, Brhs>>,
{
    type Output = Eq<Self, UInt<Urhs, Brhs>>;
}

unsafe impl Same<UTerm> for UTerm {
    type Output = B1;
}

unsafe impl Same<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<D, A, Drhs, Arhs> Same<TArr<Drhs, Arhs>> for TArr<D, A>
where
    D: Same<Drhs>,
    A: Same<Arhs>,
    <D as Same<Drhs>>::Output: BitAnd<<A as Same<Arhs>>::Output>,
{
    type Output = And<<D as Same<Drhs>>::Output, <A as Same<Arhs>>::Output>;
}

/// Binary type operator that checks whether a block of
/// shape `Sout` can fit in a block of shape `Self` at
/// position `Offset`.
///
/// Outputs B1 if it fits.
/// 
/// Note that due to `Dyn` being smaller than all type-level
/// dimensions and infectious, an axis with a `Dyn` offset
/// or a `Dyn` output shape (`Sout`) will always fit. Conversely,
/// only an axis with a `Dyn` offset or a `Dyn` output shape (`Sout`)
/// will fit in a `Dyn` input shape (`Self`). This maintains
/// the infectious behaviour of `Dyn` in the output shape (`Sout`)
/// and prevents likely panics when a `Dyn` in the input shape
/// (`Self`) becomes a a static dim in the output shape (`Sout`).
/// 
/// # Examples
/// Considering the following function and some type `Foo<T>`:
/// ```no_run
/// use melange::tensor::shape::{Fit, TRUE};
/// # use std::marker::PhantomData;
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// fn bar<Sin, Sout, Offset>(a: Foo<Sin>, b: Foo<Sout>, c: Foo<Offset>)
/// where
///     Sin: Fit<Offset, Sout>, // This bound is required by the following line
///     <Sin as Fit<Offset, Sout>>::Output: TRUE // Constrains "Sout + Offset < Sin" to hold (Output = B1)
/// {
///     // some code
/// }
/// ```
///
/// This compiles:
/// ```no_run
/// use typenum::{U1, U2, U3};
/// use melange::tensor::shape::Shape2D;
/// # use melange::tensor::shape::{Fit, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<Sin, Sout, Offset>(a: Foo<Sin>, b: Foo<Sout>, c: Foo<Offset>)
/// # where
/// #     Sin: Fit<Offset, Sout>,
/// #     <Sin as Fit<Offset, Sout>>::Output: TRUE
/// # {}
/// #
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// let sin: Foo<Shape2D<U3, U3>> = Foo::new();
/// let sout: Foo<Shape2D<U2, U2>> = Foo::new();
/// let offset: Foo<Shape2D<U1, U1>> = Foo::new();
/// bar(sin, sout, offset);
/// ```
/// While this doesn't:
/// ```compile_fail
/// use typenum::{U1, U3, U10};
/// use melange::tensor::shape::Shape2D;
/// # use melange::tensor::shape::{Fit, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<Sin, Sout, Offset>(a: Foo<Sin>, b: Foo<Sout>, c: Foo<Offset>)
/// # where
/// #     Sin: Fit<Offset, Sout>,
/// #     <Sin as Fit<Offset, Sout>>::Output: TRUE
/// # {}
/// #
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// let sin: Foo<Shape2D<U3, U3>> = Foo::new();
/// let sout: Foo<Shape2D<U10, U10>> = Foo::new();
/// let offset: Foo<Shape2D<U1, U1>> = Foo::new();
/// bar(sin, sout, offset);
/// ```
pub unsafe trait Fit<Offset, Sout> {
    /// Output type.
    type Output;
}

unsafe impl Fit<ATerm, ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<Din, Ain, Doffset, Aoffset, Dout, Aout> Fit<TArr<Doffset, Aoffset>, TArr<Dout, Aout>> for TArr<Din, Ain>
where
    Dout: Add<Doffset>,
    Sum<Dout, Doffset>: IsLessOrEqual<Din>,
    Ain: Fit<Aoffset, Aout>,
    LeEq<Sum<Dout, Doffset>, Din>: BitAnd<<Ain as Fit<Aoffset, Aout>>::Output>,
{
    type Output = And<LeEq<Sum<Dout, Doffset>, Din>, <Ain as Fit<Aoffset, Aout>>::Output>;
}

/// Binary type operator that check whether type-level shape
/// `Self` can be broadcasted to `Sout`.
///
/// Outputs B1 if the implementor shape can be broadcasted to Rhs.
/// Broadcasting is valid if for all axes in reverse order:
/// * dimensions are equal ([`Dyn`](Dyn) is included but runtime check should be done)
/// * `Self`'s dimension is U1
/// * the axis only exist `Sout`
///
/// Note that this DOES NOT require both shapes to have the same length:
/// `Sout` might be longer than `Self`.
///
/// # Examples
/// Considering the following function and some type `Foo<T>`:
/// ```no_run
/// use melange::tensor::shape::{BroadcastShape, TRUE};
/// # use std::marker::PhantomData;
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// where
///     S: BroadcastShape<Z>, // This bound is required by the following line
///     <S as BroadcastShape<Z>>::Output: TRUE // Constrains "S BroadcastShape Z" to hold (Output = B1)
/// {
///     // some code
/// }
/// ```
///
/// This compiles:
/// ```no_run
/// use typenum::{U1, U2, U3, U4};
/// use melange::tensor::shape::{Shape3D, Shape4D};
/// # use melange::tensor::shape::{BroadcastShape, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// # where
/// #     S: BroadcastShape<Z>,
/// #     <S as BroadcastShape<Z>>::Output: TRUE
/// # {}
/// #
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// let a: Foo<Shape3D<U1, U3, U1>> = Foo::new();
/// let b: Foo<Shape4D<U4, U2, U3, U2>> = Foo::new();
/// bar(a, b);
/// ```
/// While this doesn't:
/// ```compile_fail
/// use typenum::{U2, U3};
/// use melange::tensor::shape::{Shape2D};
/// # use melange::tensor::shape::{BroadcastShape, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// # where
/// #     S: BroadcastShape<Z>,
/// #     <S as BroadcastShape<Z>>::Output: TRUE
/// # {}
/// #
/// # struct Foo<T> {
/// #     _phantoms: PhantomData<T>,
/// # }
/// #
/// # impl<T> Foo<T> {
/// #     fn new() -> Self {
/// #         Foo {
/// #             _phantoms: PhantomData,
/// #         }
/// #     }
/// # }
///
/// let a: Foo<Shape2D<U2, U3>> = Foo::new();
/// let b: Foo<Shape2D<U3, U3>> = Foo::new();
/// bar(a, b);
/// ```
pub unsafe trait BroadcastShape<Sout> {
    /// Output type.
    type Output;
}

unsafe impl BroadcastShape<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<Dout, Aout> BroadcastShape<TArr<Dout, Aout>> for ATerm {
    type Output = B1;
}

unsafe impl<Din, Ain, Dout, Aout> BroadcastShape<TArr<Dout, Aout>> for TArr<Din, Ain>
where
    Din: IsEqual<Dout> + IsEqual<U1>,
    Eq<Din, Dout>: BitOr<Eq<Din, U1>>,
    Ain: BroadcastShape<Aout>,
    Or<Eq<Din, Dout>, Eq<Din, U1>>: BitAnd<<Ain as BroadcastShape<Aout>>::Output>,
{
    type Output =
        And<Or<Eq<Din, Dout>, Eq<Din, U1>>, <Ain as BroadcastShape<Aout>>::Output>;
}

/// Type operator that strides a dimension.
///
/// Outputs the result of striding `Self` with `Stride`.
///
/// The result of striding [`Dyn`] or striding by [`Dyn`] is always [`Dyn`].
/// In accordance with [`Dyn`]'s infectious property.
///
/// # Examples
/// ```
/// use typenum::{U2, U4, Unsigned};
/// use melange::tensor::shape::{StridedDim, Dim, Dyn};
///
/// assert!(<<U4 as StridedDim<U2>>::Output as Dim>::runtime_eq(2));
///
/// // It works because Dyn can be equal to 42!
/// assert!(<<U4 as StridedDim<Dyn>>::Output as Dim>::runtime_eq(42));
/// ```
///
/// [`Dyn`]: Dyn
pub unsafe trait StridedDim<Stride> {
    /// Output type.
    type Output: Dim;
}

unsafe impl<U, B, Stride> StridedDim<Stride> for UInt<U, B>
where
    Stride: StaticDim,
    Self: Div<Stride> + Rem<Stride>,
    <Self as Rem<Stride>>::Output: IsGreater<U0>,
    <Self as Div<Stride>>::Output: Add<Gr<<Self as Rem<Stride>>::Output, U0>>,
    Sum<<Self as Div<Stride>>::Output, Gr<<Self as Rem<Stride>>::Output, U0>>: Dim,
{
    type Output = Sum<<Self as Div<Stride>>::Output, Gr<<Self as Rem<Stride>>::Output, U0>>;
}

unsafe impl<Stride> StridedDim<Stride> for UTerm {
    type Output = UTerm;
}

unsafe impl<Stride> StridedDim<Stride> for Dyn {
    type Output = Dyn;
}

unsafe impl<U, B> StridedDim<Dyn> for UInt<U, B> {
    type Output = Dyn;
}

/// Type operator that strides a static shape.
///
/// Outputs the shape corresponding to striding `Self`
/// shape with `Strides`.
///
/// This trait adds a further guarantee to `StridedShapeDyn` which is that
/// the output is guaranteed to be static. This means that the two inputs
/// must be coercible: they cannot both contain `Dyn` on the same axis.
///
/// Note that this requires both inputs to have the same length.
///
/// # Examples
/// ```
/// use typenum::{U2, U4, Unsigned};
/// use melange::tensor::shape::{Shape2D, StridedShape, StaticShape};
///
/// assert_eq!(<<Shape2D<U4, U2> as StridedShape<Shape2D<U2, U2>>>::Output as StaticShape>::to_vec(), vec![2, 1]);
/// ```
pub unsafe trait StridedShape<Strides> {
    /// Output type.
    type Output: StaticShape;
}

unsafe impl StridedShape<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A, Sstrides, Astrides> StridedShape<TArr<Sstrides, Astrides>> for TArr<S, A>
where
    S: StridedDim<Sstrides>,
    A: StridedShape<Astrides>,
    TArr<<S as StridedDim<Sstrides>>::Output, <A as StridedShape<Astrides>>::Output>: StaticShape,
{
    type Output = TArr<<S as StridedDim<Sstrides>>::Output, <A as StridedShape<Astrides>>::Output>;
}

/// Type operator that strides a shape.
///
/// Outputs the shape corresponding to striding `Self`
/// shape with `Strides`.
///
/// Note that this requires both shapes to have the same length.
/// The output is just guaranteed to be `Shape` i.e. it can still be dynamic.
///
/// # Examples
/// ```
/// use typenum::{U2, U4, Unsigned};
/// use melange::tensor::shape::{Shape2D, StridedShapeDyn, Shape, Dyn};
///
/// // It works because Dyn can be equal to 42!
/// assert!(<<Shape2D<Dyn, U4> as StridedShapeDyn<Shape2D<U2, U2>>>::Output as Shape>::runtime_compat(&[42, 2]));
/// ```
pub unsafe trait StridedShapeDyn<Strides> {
    /// Output type.
    type Output: Shape;
}

unsafe impl StridedShapeDyn<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A, Sstrides, Astrides> StridedShapeDyn<TArr<Sstrides, Astrides>> for TArr<S, A>
where
    S: StridedDim<Sstrides>,
    A: StridedShapeDyn<Astrides>,
    TArr<<S as StridedDim<Sstrides>>::Output, <A as StridedShapeDyn<Astrides>>::Output>: Shape,
{
    type Output = TArr<<S as StridedDim<Sstrides>>::Output, <A as StridedShapeDyn<Astrides>>::Output>;
}

/// Type operator that computes upsampling strides.
///
/// Outputs the strides that need to be used on `Self`
/// when upsampling `Sin` into `Self`.
/// 
/// If a dimension in `Sin` is `Dyn`, the corresponding
/// dimension in `Self` must also be `Dyn`. This constraint
/// is present because switching from a dynamic dimension in
/// the input to a statically known dimension in the output
/// will likely result in a panic at runtime.
///
/// Note that this requires both shapes to have the same length.
///
/// # Examples
/// ```
/// use typenum::{U2, U4, Unsigned};
/// use melange::tensor::shape::{Shape2D, UpsamplingStrides, Shape, Dyn};
///
/// // It works because Dyn can be equal to 42!
/// assert!(<<Shape2D<Dyn, U4> as UpsamplingStrides<Shape2D<Dyn, U2>>>::Output as Shape>::runtime_compat(&[42, 2]));
/// ```
pub unsafe trait UpsamplingStrides<Sin> {
    /// Output type.
    type Output: Shape;
}

unsafe impl UpsamplingStrides<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<Sout, Aout, Sin, Ain> UpsamplingStrides<TArr<Sin, Ain>> for TArr<Sout, Aout>
where
    Sout: IsNotEqual<Dyn> + Div<Sin>,
    Sin: IsEqual<Dyn>,
    NotEq<Sout, Dyn>: BitAnd<Eq<Sin, Dyn>>,
    And<NotEq<Sout, Dyn>, Eq<Sin, Dyn>>: IsEqual<B0>,
    Eq<And<NotEq<Sout, Dyn>, Eq<Sin, Dyn>>, B0>: TRUE,
    Aout: UpsamplingStrides<Ain>,
    TArr<Quot<Sout, Sin>, <Aout as UpsamplingStrides<Ain>>::Output>: Shape,
{
    type Output = TArr<Quot<Sout, Sin>, <Aout as UpsamplingStrides<Ain>>::Output>;
}

/// Conditionnal trait operator.
///
/// This reproduces a if/else block at the type level:
/// * outputs T if the implementor is B1,
/// * outputs Else otherwise.
pub trait If<T, Else> {
    /// Output type.
    type Output;
}

impl<T, Else> If<T, Else> for B1 {
    type Output = T;
}

impl<T, Else> If<T, Else> for B0 {
    type Output = Else;
}

/// Trait operator that inserts dimension `Dnew` before the first axis.
///
/// This is useful because dimensions are stored in reverse order in
/// the recursive `TArr` structure.
///
/// # Example
/// ```
/// use typenum::{U1, U2};
/// use melange::tensor::shape::{Shape1D, Shape2D, StaticShape, Insert};
///
/// assert_eq!(<<Shape1D<U2> as Insert<U1>>::Output as StaticShape>::to_vec(), vec![1, 2]);
/// ```
pub unsafe trait Insert<Dnew> {
    /// Output type.
    type Output;
}

unsafe impl<Dnew> Insert<Dnew> for ATerm {
    type Output = TArr<Dnew, ATerm>;
}

unsafe impl<D, A, Dnew> Insert<Dnew> for TArr<D, A>
where
    A: Insert<Dnew>,
{
    type Output = TArr<D, <A as Insert<Dnew>>::Output>;
}

/// Trait operator that removes the first axis.
///
/// This is useful because dimensions are stored in reverse order in
/// the recursive `TArr` structure.
///
/// # Example
/// ```
/// use typenum::{U1, U2};
/// use melange::tensor::shape::{Shape1D, Shape2D, StaticShape, RemoveFirst};
///
/// assert_eq!(<<Shape2D<U1, U2> as RemoveFirst>::Output as StaticShape>::to_vec(), vec![2]);
/// ```
pub unsafe trait RemoveFirst {
    /// Output type.
    type Output;
}

unsafe impl<D> RemoveFirst for TArr<D, ATerm> {
    type Output = ATerm;
}

unsafe impl<D, A> RemoveFirst for TArr<D, A>
where
    A: RemoveFirst,
{
    type Output = TArr<D, <A as RemoveFirst>::Output>;
}

/// Type operator that reverses the order of the axes in the implementor shape.
///
/// # Example
/// ```
/// use typenum::{U1, U2, U3, U4};
/// use melange::tensor::shape::{Shape4D, StaticShape, TransposeShape};
///
/// assert_eq!(<<Shape4D<U1, U2, U3, U4> as TransposeShape>::Output as StaticShape>::to_vec(), vec![4, 3, 2, 1]);
/// ```
pub unsafe trait TransposeShape {
    /// Output type.
    type Output;
}

unsafe impl TransposeShape for ATerm {
    type Output = ATerm;
}

unsafe impl<D, A> TransposeShape for TArr<D, A>
where
    A: TransposeShape,
    <A as TransposeShape>::Output: Insert<D>,
{
    type Output = <<A as TransposeShape>::Output as Insert<D>>::Output;
}

/// 1D shape alias.
pub type Shape1D<D0> = TArr<D0, ATerm>;
/// 2D shape alias.
pub type Shape2D<D0, D1> = TArr<D1, TArr<D0, ATerm>>;
/// 3D shape alias.
pub type Shape3D<D0, D1, D2> = TArr<D2, TArr<D1, TArr<D0, ATerm>>>;
/// 4D shape alias.
pub type Shape4D<D0, D1, D2, D3> = TArr<D3, TArr<D2, TArr<D1, TArr<D0, ATerm>>>>;
/// 5D shape alias.
pub type Shape5D<D0, D1, D2, D3, D4> = TArr<D4, TArr<D3, TArr<D2, TArr<D1, TArr<D0, ATerm>>>>>;
/// 6D shape alias.
pub type Shape6D<D0, D1, D2, D3, D4, D5> =
    TArr<D5, TArr<D4, TArr<D3, TArr<D2, TArr<D1, TArr<D0, ATerm>>>>>>;
