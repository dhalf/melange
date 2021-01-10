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
use typenum::private::InternalMarker;
use typenum::type_operators::*;
use typenum::{ATerm, Bit, Equal, TArr, UInt, Unsigned, B0, B1, U0, U1};

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
/// It implements type-level comparisons with type-level unsigned integers and
/// is considered equal to all of them. This involves Dyn is compatible
/// with any dimension.
///
/// # Examples
///
/// Considering the following function:
/// ```no_run
/// use typenum::{U1, IsEqual, Eq};
/// use melange::tensor::shape::TRUE;
///
/// fn bar<D>(x: D)
/// where
///     D: IsEqual<U1>, // This bound is required by the following line
///     Eq<D, U1>: TRUE // Constrains "D IsEqual U1" to hold (Output = B1)
/// {
///     // some code
/// }
/// ```
///
/// This compiles:
/// ```no_run
/// # use typenum::{U1, IsEqual, Eq};
/// # use melange::tensor::shape::{Dyn, TRUE};
/// # fn bar<D>(x: D)
/// # where
/// #     D: IsEqual<U1>, // This bound is required by the following line
/// #     Eq<D, U1>: TRUE // Constrains "D IsEqual U1" to hold (Output = B1)
/// # {
/// #     // some code
/// # }
/// #
/// let a = Dyn;
/// bar(a);
/// ```
///
/// While this doesn't:
/// ```compile_fail
/// # use typenum::{U1, U2, IsEqual, Eq};
/// # use melange::tensor::shape::TRUE;
/// # fn bar<D>(x: D)
/// # where
/// #     D: IsEqual<U1>, // This bound is required by the following line
/// #     Eq<D, U1>: TRUE // Constrains "D IsEqual U1" to hold (Output = B1)
/// # {
/// #     // some code
/// # }
/// #
/// let a = U2::new();
/// bar(a);
/// ```
#[derive(Debug, PartialEq)]
pub struct Dyn;
impl<U> Cmp<U> for Dyn
where
    U: Dim,
{
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &U) -> Self::Output {
        Equal
    }
}
impl<U, B> Cmp<Dyn> for UInt<U, B> {
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &Dyn) -> Self::Output {
        Equal
    }
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

/// Binary type operator that check the compatibility of two type-level
/// shapes.
///
/// Outputs B1 if the implementor shape is compatible with Rhs
/// i.e. all the dimensions on the respective axes are compatible.
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
/// use typenum::{U1, TArr};
/// use melange::tensor::shape::Shape2D;
/// # use melange::tensor::shape::{Same, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// # where
/// #     S: Same<Z>, // This bound is required by the following line
/// #     <S as Same<Z>>::Output: TRUE // Constrains "S Same Z" to hold (Output = B1)
/// # {
/// #     // some code
/// # }
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
/// #     S: Same<Z>, // This bound is required by the following line
/// #     <S as Same<Z>>::Output: TRUE // Constrains "S Same Z" to hold (Output = B1)
/// # {
/// #     // some code
/// # }
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

unsafe impl Same<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<S, A, SRhs, ARhs> Same<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: IsEqual<SRhs>,
    A: Same<ARhs>,
    Eq<S, SRhs>: BitAnd<<A as Same<ARhs>>::Output>,
{
    type Output = And<Eq<S, SRhs>, <A as Same<ARhs>>::Output>;
}

/// Binary type operator that check whether two type-level shapes
/// can be broadcasted.
///
/// Outputs B1 if the implementor shape can be broadcasted to Rhs.
/// Broadcasting is valid if for all axes in reverse order:
/// * dimensions are equal ([`Dyn`](Dyn) is included but runtime check should be done)
/// * one of the dimensions is U1
/// * the axis only exist in the largest shape
///
/// Note that this DOES NOT require both shapes to have the same length.
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
/// use typenum::{U1, U2, U3, U4, TArr};
/// use melange::tensor::shape::{Shape3D, Shape4D};
/// # use melange::tensor::shape::{BroadcastShape, TRUE};
/// # use std::marker::PhantomData;
/// #
/// # fn bar<S, Z>(a: Foo<S>, b: Foo<Z>)
/// # where
/// #     S: BroadcastShape<Z>, // This bound is required by the following line
/// #     <S as BroadcastShape<Z>>::Output: TRUE // Constrains "S BroadcastShape Z" to hold (Output = B1)
/// # {
/// #     // some code
/// # }
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
/// #     S: BroadcastShape<Z>, // This bound is required by the following line
/// #     <S as BroadcastShape<Z>>::Output: TRUE // Constrains "S BroadcastShape Z" to hold (Output = B1)
/// # {
/// #     // some code
/// # }
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
pub unsafe trait BroadcastShape<Rhs> {
    /// Output type.
    type Output;
}

unsafe impl BroadcastShape<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<S, A> BroadcastShape<TArr<S, A>> for ATerm {
    type Output = B1;
}

unsafe impl<S, A, SRhs, ARhs> BroadcastShape<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: IsEqual<SRhs> + IsEqual<U1>,
    SRhs: IsEqual<U1>,
    Eq<S, SRhs>: BitOr<Eq<S, U1>>,
    Or<Eq<S, SRhs>, Eq<S, U1>>: BitOr<Eq<SRhs, U1>>,
    A: BroadcastShape<ARhs>,
    Or<Or<Eq<S, SRhs>, Eq<S, U1>>, Eq<SRhs, U1>>: BitAnd<<A as BroadcastShape<ARhs>>::Output>,
{
    type Output =
        And<Or<Or<Eq<S, SRhs>, Eq<S, U1>>, Eq<SRhs, U1>>, <A as BroadcastShape<ARhs>>::Output>;
}

/// Type operator that strides a dimension.
///
/// Outputs the result of striding the implementor dimension with Rhs.
///
/// The result of striding [`Dyn`] or striding by [`Dyn`] is always [`Dyn`].
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
pub unsafe trait StridedDim<Rhs> {
    /// Output type.
    type Output: Dim;
}

unsafe impl<U, B, V> StridedDim<V> for UInt<U, B>
where
    V: StaticDim,
    Self: Div<V> + Rem<V>,
    <Self as Rem<V>>::Output: IsGreater<U0>,
    <Self as Div<V>>::Output: Add<Gr<<Self as Rem<V>>::Output, U0>>,
    Sum<<Self as Div<V>>::Output, Gr<<Self as Rem<V>>::Output, U0>>: Dim,
{
    type Output = Sum<<Self as Div<V>>::Output, Gr<<Self as Rem<V>>::Output, U0>>;
}

unsafe impl<V> StridedDim<V> for Dyn {
    type Output = Dyn;
}

unsafe impl<U, B> StridedDim<Dyn> for UInt<U, B> {
    type Output = Dyn;
}

/// Type operator that strides a static shape.
///
/// Outputs the shape corresponding to the striding of the implementor
/// shape with Rhs.
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
pub unsafe trait StridedShape<Rhs> {
    /// Output type.
    type Output: StaticShape;
}

unsafe impl StridedShape<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A, SRhs, ARhs> StridedShape<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: StridedDim<SRhs>,
    A: StridedShape<ARhs>,
    TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShape<ARhs>>::Output>: StaticShape,
{
    type Output = TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShape<ARhs>>::Output>;
}

/// Type operator that strides a shape.
///
/// Outputs the shape corresponding to the striding of the implementor
/// shape with Rhs.
///
/// Note that this requires both shapes to have the same length.
/// The is just guaranteed to be `Shape` i.e. it can still be dynamic.
///
/// # Examples
/// ```
/// use typenum::{U2, U4, Unsigned};
/// use melange::tensor::shape::{Shape2D, StridedShapeDyn, Shape, Dyn};
///
/// // It works because Dyn can be equal to 42!
/// assert!(<<Shape2D<Dyn, U4> as StridedShapeDyn<Shape2D<U2, U2>>>::Output as Shape>::runtime_compat(&[42, 2]));
/// ```
pub unsafe trait StridedShapeDyn<Rhs> {
    /// Output type.
    type Output: Shape;
}

unsafe impl StridedShapeDyn<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A, SRhs, ARhs> StridedShapeDyn<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: StridedDim<SRhs>,
    A: StridedShapeDyn<ARhs>,
    TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShapeDyn<ARhs>>::Output>: Shape,
{
    type Output = TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShapeDyn<ARhs>>::Output>;
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

/// Trait operator that replaces the dimension of the axis
/// having the (0-starting) index Ax (a type-level unsigned integer)
/// with U1.
///
/// # Example
/// ```
/// use typenum::{U0, U2};
/// use melange::tensor::shape::{Shape2D, StaticShape, Reduction};
///
/// assert_eq!(<<Shape2D<U2, U2> as Reduction<U0>>::Output as StaticShape>::to_vec(), vec![1, 2]);
/// ```
pub trait Reduction<Ax> {
    /// Output type.
    type Output;
}

impl<Ax> Reduction<Ax> for ATerm {
    type Output = ATerm;
}

impl<Ax, D, Ar> Reduction<Ax> for TArr<D, Ar>
where
    Self: Len,
    Length<Self>: Sub<B1>,
    Ax: IsEqual<Sub1<Length<Self>>>,
    Ar: Reduction<Ax>,
    Eq<Ax, Sub1<Length<Self>>>: If<TArr<U1, Ar>, TArr<D, <Ar as Reduction<Ax>>::Output>>,
{
    type Output = <Eq<Ax, Sub1<Length<Self>>> as If<
        TArr<U1, Ar>,
        TArr<D, <Ar as Reduction<Ax>>::Output>,
    >>::Output;
}

/// Trait operator that inserts dimension S before the first axis.
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
pub unsafe trait Insert<S> {
    /// Output type.
    type Output;
}

unsafe impl<S> Insert<S> for ATerm {
    type Output = TArr<S, ATerm>;
}

unsafe impl<S, A, Z> Insert<Z> for TArr<S, A>
where
    A: Insert<Z>,
{
    type Output = TArr<S, <A as Insert<Z>>::Output>;
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

unsafe impl<S, A> TransposeShape for TArr<S, A>
where
    A: TransposeShape,
    <A as TransposeShape>::Output: Insert<S>,
{
    type Output = <<A as TransposeShape>::Output as Insert<S>>::Output;
}

/// 1D shape alias.
pub type Shape1D<S0> = TArr<S0, ATerm>;
/// 2D shape alias.
pub type Shape2D<S0, S1> = TArr<S1, TArr<S0, ATerm>>;
/// 3D shape alias.
pub type Shape3D<S0, S1, S2> = TArr<S2, TArr<S1, TArr<S0, ATerm>>>;
/// 4D shape alias.
pub type Shape4D<S0, S1, S2, S3> = TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>;
/// 5D shape alias.
pub type Shape5D<S0, S1, S2, S3, S4> = TArr<S4, TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>>;
/// 6D shape alias.
pub type Shape6D<S0, S1, S2, S3, S4, S5> =
    TArr<S5, TArr<S4, TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>>>;
