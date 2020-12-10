//! Contains basic mathematical operations at the tensor level.
//! 
//! All operations are wrapped in their own trait which eases reuse.
//!
//! There are two families of operations: functionnal operations
//! and in-place operations.
//! 
//! Functionnal operations are methods that immutably borrow `self` (and
//! optionnaly borrow other tensors) and return a new tensor. They are
//! actually defined with traits that move their inputs to be compatible
//! with traits in [`std::ops`](`std::ops`). The trick is to implement
//! those traits for references.
//!
//! Conversely, in-place operations mutably borrow `self` (and optionnaly
//! immutably borrow other tensors) to directly mutate its data. The actual
//! implementation, in this case, does borrow `self` mutably but moves other
//! parameters. Once again the trick is to implement for references.
//! 
//! Functionnal operations are interesting when backpropagating because they
//! preserve operands whereas in-place operations reduce the memory footprint.
//! 
//! Both families of operations perform ad-hoc parallel computation
//! acording to how data is stored. Note that operations on more than one
//! tensor require all tensors to have compatible shapes (all dimensions
//! must be equal or `Dyn`). If this is not the case consider broadcasting.
//! 
//! Note that functionnal operations actually use in-place operations
//! under the hood.
//!
//! Note that some operations are only available for tensors based on float
//! types `f32` and `f64` or primitive integers.
//! This is inherent to how numeric types are treated in rust.
//! The doc of all traits clearly mentions which kind of tensors implement them.
//! 
//! Those operations rely on the scalar operations implemented for the underlying
//! scalar data type `T` of the tensor. Please refer to the relevant methods defined
//! on primitive types.

use std::ops::*;
use super::strided_iterator::{StridedIterator, StridedIteratorMut};
use crate::tensor::{Tensor, Static, Dynamic};
use super::layout::Layout;
use super::shape::{Shape, Same, TRUE};
use crate::gat::{RefMutGat, StreamingIterator};
use crate::algebra::Field;
use crate::tensor::alloc::AllocLike;

macro_rules! assert_shape_eq {
    ($lhs:expr, $rhs:expr) => {
        assert_eq!(
            $lhs,
            $rhs,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            $lhs,
            $rhs
        );
    };
}

/// In-place elementwise addition operator.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`AddAssign`](std::ops::AddAssign). `rhs` can either be a
/// tensor or a scalar of type `T`.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 1, 1, 1]).unwrap();
/// a.add_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 1, 1, 2]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// a.add_(1);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 1, 1, 2]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Add_<Rhs=Self> {
    /// Performs the in-place addition.
    fn add_(&mut self, rhs: Rhs);
}

/// In-place elementwise subtraction operator.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`SubAssign`](std::ops::SubAssign). `rhs` can either be a
/// tensor or a scalar of type `T`.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 1, 1, 1]).unwrap();
/// a.sub_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![0, -1, -1, 0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// a.sub_(1);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![0, -1, -1, 0]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Sub_<Rhs=Self> {
    /// Performs the in-place subtraction.
    fn sub_(&mut self, rhs: Rhs);
}

/// In-place elementwise multiplication operator.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`MulAssign`](std::ops::MulAssign). `rhs` can either be a
/// tensor or a scalar of type `T`.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 2, 2, 2]).unwrap();
/// a.mul_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 0, 0, 2]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// a.mul_(2);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 0, 0, 2]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Mul_<Rhs=Self> {
    /// Performs the in-place multiplication.
    fn mul_(&mut self, rhs: Rhs);
}

/// In-place elementwise division operator.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`DivAssign`](std::ops::DivAssign). `rhs` can either be a
/// tensor or a scalar of type `T`.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 0, 0, 2]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 2, 2, 2]).unwrap();
/// a.div_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 0, 0, 2]).unwrap();
/// a.div_(2);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Div_<Rhs=Self> {
    /// Performs the in-place division.
    fn div_(&mut self, rhs: Rhs);
}

/// In-place elementwise remainder operator.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`RemAssign`](std::ops::RemAssign). `rhs` can either be a
/// tensor or a scalar of type `T`.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![5, 0, 0, 5]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 2, 2, 2]).unwrap();
/// a.rem_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![5, 0, 0, 5]).unwrap();
/// a.rem_(2);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, 0, 1]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Rem_<Rhs=Self> {
    /// Performs the in-place remainder operation.
    fn rem_(&mut self, rhs: Rhs);
}

/// In-place elementwise four quadrant arctangent operator.
/// 
/// Computes the four quadrant arctengent of `self` (`y`)
/// and `rhs` (`x`) in radians as follows:
/// * `x = 0`, `y = 0`: `0`
/// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
/// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
/// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`atan2`](f32::atan2) function.
/// `rhs` can either be a tensor or a scalar having the same scalar
/// type.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]).unwrap();
/// a.atan2_(&b);
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_4, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// a.sub_(&c);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::FRAC_PI_2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, -2.0, 1.0]).unwrap();
/// a.atan2_(0.0);
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
///```
pub trait Atan2_<Rhs=Self> {
    /// Performs the in-place four quadrant arctangent operation.
    fn atan2_(&mut self, rhs: Rhs);
}

/// In-place elementwise copysign operator.
/// 
/// Result is composed of the magnitude of `self` and the sign of `rhs`.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`copysign`](f32::copysign) function.
/// `rhs` can either be a tensor or a scalar having the same scalar type.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, -4.0, -4.0, 2.0]).unwrap();
/// a.copysign_(&b);
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -3.0, 4.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -1.0, 5.0]).unwrap();
/// a.copysign_(-5.0);
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, -2.0, -1.0, -5.0]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Copysign_<Rhs=Self> {
    /// Performs the in-place copysign operation.
    fn copysign_(&mut self, rhs: Rhs);
}

/// In-place elementwise quotient of Euclidean division operator.
/// 
/// Computes the quotient of Euclidean division of `self` by `rhs`.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive integer (e.g. `i64`) using primitive
/// [`div_euclid`](isize::div_euclid) function.
/// `rhs` can either be a tensor or a scalar having the same scalar type.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![7, 7, -7, -7]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![4, -4, 4, -4]).unwrap();
/// a.div_euclid_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, -1, -2, 2]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![7, -7]).unwrap();
/// a.div_euclid_(4);
/// let b: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![1, -2]).unwrap();
/// assert_eq!(a, b);
///```
pub trait DivEuclid_<Rhs=Self> {
    /// Performs the in-place quotient of Euclidean division operation.
    fn div_euclid_(&mut self, rhs: Rhs);
}

/// In-place elementwise remainder of Euclidean division operator.
/// 
/// Computes the least nonnegative remainder of `self (mod rhs)`.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive integer (e.g. `i64`) using primitive
/// [`rem_euclid`](isize::rem_euclid) function.
/// `rhs` can either be a tensor or a scalar having the same scalar type.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![7, -7, 7, -7]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![4, 4, -4, -4]).unwrap();
/// a.rem_euclid_(&b);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![3, 1, 3, 1]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![7, -7]).unwrap();
/// a.rem_euclid_(4);
/// let b: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![3, 1]).unwrap();
/// assert_eq!(a, b);
///```
pub trait RemEuclid_<Rhs=Self> {
    /// Performs the in-place remainder of Euclidean division operation.
    fn rem_euclid_(&mut self, rhs: Rhs);
}

/// In-place elementwise max operator.
/// 
/// Computes the max of `self` and `rhs`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`max`](f32::max) function.
/// `rhs` can either be a tensor or a scalar having the same scalar type.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`Ord`](std::cmp::Ord).
/// `rhs` can either be a tensor or a scalar having the same scalar type.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// a.max_(&b);
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![7.0, -3.0, 6.0, 12.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// a.max_(5.0);
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, 10.0, 5.0, 6.0]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Max_<Rhs=Self> {
    /// Performs the in-place max operation.
    fn max_(&mut self, rhs: Rhs);
}

/// In-place elementwise min operator.
/// 
/// Computes the min of `self` and `rhs`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`max`](f32::max) function.
/// `rhs` can either be a tensor or a scalar having the same scalar type.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`Ord`](std::cmp::Ord).
/// `rhs` can either be a tensor or a scalar having the same scalar type.
///
/// # Panics
/// If `self` and `rhs` are dynamic tensors with differing
/// runtime shapes.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// a.min_(&b);
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, -8.0, 1.0, 4.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// a.min_(5.0);
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 5.0, 3.0, 5.0]).unwrap();
/// assert_eq!(a, b);
///```
pub trait Min_<Rhs=Self> {
    /// Performs the in-place max operation.
    fn min_(&mut self, rhs: Rhs);
}

/// In-place elementwise power operator.
/// 
/// Raise `self` to `rhs` power.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`powi`](f32::powi)
/// [`powf`](f32::powf) functions. `rhs` is respectively
/// a scalar `f32` or `f64`.
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive integer (e.g. `i64`) using primitive
/// [`pow`](isize::pow) function. `rhs` is a scalar `u32`.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// a.pow_(2);
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 4, 9, 16]).unwrap();
/// assert_eq!(a, c);
/// ```
pub trait Pow_<Rhs=Self> {
    /// Performs the in-place max operation.
    fn pow_(&mut self, rhs: Rhs);
}

/// In-place elementwise exponential function.
/// 
/// `self -> e^(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`exp`](f32::exp).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.exp_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Exp_ {
    /// Applies in-place exp function.
    fn exp_(&mut self);
}

/// In-place elementwise power of 2 function.
/// 
/// `self -> 2^(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`exp2`](f32::exp2).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// a.exp2_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Exp2_ {
    /// Applies in-place `2^x` function.
    fn exp2_(&mut self);
}

/// In-place elementwise exponential function minus 1.
///
/// This is more accurate than if the operations were
/// performed separately even if `self` is close to zero.
/// 
/// `self -> exp(x) - 1`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`exp_m1`](f32::exp_m1).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::LN_2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![LN_2, 0.0, 0.0, LN_2]).unwrap();
/// a.exp_m1_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait ExpM1_ {
    /// Applies in place `exp(x) - 1` function.
    fn exp_m1_(&mut self);
}

/// In-place elementwise natural logarithm function.
/// 
/// `self -> ln(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`ln`](f32::ln).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// a.ln_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Ln_ {
    /// Applies in-place natural logarithm function.
    fn ln_(&mut self);
}

/// In-place elementwise natural logarithm of 1 plus x function.
///
/// This is more accurate than if the operations were performed separately.
/// 
/// `self -> ln(1 + self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`ln_1p`](f32::ln_1p).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, E - 1.0, E - 1.0, 0.0]).unwrap();
/// a.ln_1p_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, 1.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Ln1p_ {
    /// Applies in-place `ln(1 + x)` function.
    fn ln_1p_(&mut self);
}

/// In-place elementwise base 2 logarithm function.
/// 
/// `self -> log2(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`log2`](f32::log2).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// a.log2_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Log2_ {
    /// Applies in-place base 2 logarithm function.
    fn log2_(&mut self);
}

/// In-place elementwise base 10 logarithm function.
/// 
/// `self -> log10(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`log10`](f32::log10).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![100.0, 1_000.0, 1_000.0, 100.0]).unwrap();
/// a.log10_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, 3.0, 3.0, 2.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Log10_ {
    /// Applies in-place base 10 logarithm function.
    fn log10_(&mut self);
}

/// In-place elementwise sine function.
/// 
/// `self` (in radians) `-> sin(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`sin`](f32::sin).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// a.sin_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Sin_ {
    /// Applies in-place sine function.
    fn sin_(&mut self);
}

/// In-place elementwise cosine function.
/// 
/// `self` (in radians) `-> cos(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`cos`](f32::cos).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.cos_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Cos_ {
    /// Applies in-place cosine function.
    fn cos_(&mut self);
}

/// In-place elementwise tangent function.
/// 
/// `self` (in radians) `-> tan(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`tan`](f32::tan).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// a.tan_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Tan_ {
    /// Applies in-place tangent function.
    fn tan_(&mut self);
}

/// In-place elementwise hyperbolic sine function.
/// 
/// `self -> sinh(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`sinh`](f32::sinh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sinh_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Sinh_ {
    /// Applies in-place hyperbolic sine function.
    fn sinh_(&mut self);
}

/// In-place elementwise hyperbolic cosine function.
/// 
/// `self -> cosh(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`cosh`](f32::cosh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.cosh_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Cosh_ {
    /// Applies in-place hyperbolic cosine function.
    fn cosh_(&mut self);
}

/// In-place elementwise hyperbolic tangent function.
/// 
/// `self -> tanh(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`tanh`](f32::tanh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.tanh_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Tanh_ {
    /// Applies in-place hyperbolic tangent function.
    fn tanh_(&mut self);
}

/// In-place elementwise arcsine function.
/// 
/// `self` in [-1, 1] `-> asin(self)` (in radians) in [-pi/2, pi/2]
///
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`asin`](f32::asin).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// a.asin_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Asin_ {
    /// Applies in-place arcsine function.
    fn asin_(&mut self);
}

/// In-place elementwise arccosine function.
/// 
/// `self` in [-1, 1] `-> acos(self)` (in radians) in [0, pi]
/// 
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`acos`](f32::acos).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// a.acos_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Acos_ {
    /// Applies in-place arccosine function.
    fn acos_(&mut self);
}

/// In-place elementwise arctangent function.
/// 
/// `self -> atan(self)` (in radians) in [-pi/2, pi/2]
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`atan`](f32::atan).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// a.atan_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Atan_ {
    /// Applies in-place arctangent function.
    fn atan_(&mut self);
}

/// In-place elementwise inverse hyperbolic sine function.
/// 
/// `self -> asinh(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`asinh`](f32::asinh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.asinh_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Asinh_ {
    /// Applies in-place inverse hyperbolic sine function.
    fn asinh_(&mut self);
}

/// In-place elementwise inverse hyperbolic cosine function.
/// 
/// `self -> acosh(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`acosh`](f32::acosh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// a.acosh_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Acosh_ {
    /// Applies in-place inverse hyperbolic cosine function.
    fn acosh_(&mut self);
}

/// In-place elementwise inverse hyperbolic tangent function.
/// 
/// `self -> atanh(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`atanh`](f32::atanh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.atanh_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Atanh_ {
    /// Applies in-place inverse hyperbolic tangent function.
    fn atanh_(&mut self);
}

/// In-place elementwise square root function.
/// 
/// `self -> sqrt(self)`
/// 
/// Note: `sqrt(self)` is NaN if `self` is negative.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`sqrt`](f32::sqrt).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 4.0, 9.0, 16.0]).unwrap();
/// a.sqrt_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Sqrt_ {
    /// Applies in-place square root function.
    fn sqrt_(&mut self);
}

/// In-place elementwise cubic root function.
/// 
/// `self -> cbrt(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`cbrt`](f32::cbrt).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 8.0, 64.0, 512.0]).unwrap();
/// a.cbrt_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 4.0, 8.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Cbrt_ {
    /// Applies in-place cubic root function.
    fn cbrt_(&mut self);
}

/// In-place elementwise absolute value function.
/// 
/// `self -> abs(self)`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`abs`](f32::abs).
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive signed integer (e.g. `i64`) using primitive
/// [`abs`](isize::abs) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -6.0, 3.0]).unwrap();
/// a.abs_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 6.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Abs_ {
    /// Applies in-place absolute value function.
    fn abs_(&mut self);
}

/// In-place elementwise sign function.
/// 
/// `self ->`
/// * `1` if `self` is positive
/// * `-1` id `self` is negative
/// 
/// Note that for signed integers, `0` -> `0`. However,
/// for floats, `+0.0` -> `1.0` and `-0.0` -> `-1.0`. 
///
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`signum`](f32::signum).
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive signed integer (e.g. `i64`) using primitive
/// [`signum`](isize::signum) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, -2.0, -6.0, 0.0]).unwrap();
/// a.signum_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -1.0, -1.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Signum_ {
    /// Applies in-place sign function.
    fn signum_(&mut self);
}

/// In-place elementwise ceiling function.
/// 
/// `self -> ceil(self)`
/// 
/// Note: `ceil(x)` is the smallest integer greater than or equal to `x`. 
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`ceil`](f32::ceil).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// a.ceil_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, -2.0, -6.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Ceil_ {
    /// Applies in-place ceiling function.
    fn ceil_(&mut self);
}

/// In-place elementwise floor function.
/// 
/// `self -> floor(self)`
/// 
/// Note: `floor(x)` is the greatest integer smaller than or equal to `x`. 
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`floor`](f32::floor).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// a.floor_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -3.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Floor_ {
    /// Applies in-place floor function.
    fn floor_(&mut self);
}

/// In-place elementwise rounding function.
/// 
/// `self -> round(self)`
/// 
/// Note: `round(x)` is the nearest integer to `x`. Half-way cases
/// are rounded away from zero.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`round`](f32::round).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// a.round_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Round_ {
    /// Applies in-place rounding function.
    fn round_(&mut self);
}

/// In-place elementwise inverse function.
/// 
/// `self -> 1 / self`
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`recip`](f32::recip).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, 0.5, 4.0]).unwrap();
/// a.recip_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -0.5, 2.0, 0.25]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Recip_ {
    /// Applies in-place inverse function.
    fn recip_(&mut self);
}

/// In-place elementwise multiplicative inverse function.
/// 
/// `self -> 1 / self`
///  
/// Implemented for all tensors whose scalar type implements
/// the [`Field`](crate::algebra::Field) trait. For primitive
/// `f32` and `f64` types, this is strictly equivalent to
/// [`Recip_`](Recip_).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, 0.5, 4.0]).unwrap();
/// a.minv_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -0.5, 2.0, 0.25]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait Minv_ {
    /// Computes in-place elementwise the multiplicative inverse.
    fn minv_(&mut self);
}

/// In-place elementwise conversion to degrees function.
/// 
/// `self` (in radians) `-> self` (in degrees)
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`to_degrees`](f32::to_degrees).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.to_degrees_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait ToDegrees_ {
    /// Converts in-place to degrees.
    fn to_degrees_(&mut self);
}

/// In-place elementwise conversion to radians function.
/// 
/// `self` (in degrees) `-> self` (in radians)
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`to_radians`](f32::to_radians).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// a.to_radians_();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.sub_(&b);
/// a.abs_();
/// assert!(a < 1e-10);
/// ```
pub trait ToRadians_ {
    /// Converts in-place to radians.
    fn to_radians_(&mut self);
}

/// Elementwise four quadrant arctangent operator.
/// 
/// Computes the four quadrant arctengent of `self` (`y`)
/// and `rhs` (`x`) in radians as follows:
/// * `x = 0`, `y = 0`: `0`
/// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
/// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
/// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`atan2`](f32::atan2) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_4, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// assert!(a.atan2(&b).sub(&c).abs() < f64::EPSILON);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::FRAC_PI_2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, -2.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2]).unwrap();
/// assert!(a.atan2(0.0).sub(&b).abs() < f64::EPSILON);
///```
pub trait Atan2<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the four quadrant arctangent operation.
    fn atan2(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise copysign operator.
/// 
/// Result is composed of the magnitude of `self` and the sign of `rhs`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`copysign`](f32::copysign) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, -4.0, -4.0, 2.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -3.0, 4.0]).unwrap();
/// assert_eq!(a.copysign(&b), c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -1.0, 5.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, -2.0, -1.0, -5.0]).unwrap();
/// assert_eq!(a.copysign(-5.0), b);
///```
pub trait Copysign<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the copysign operation.
    fn copysign(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise quotient of Euclidean division operator.
/// 
/// Computes the quotient of Euclidean division of `self` by `rhs`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive integer (e.g. `i64`) using primitive
/// [`div_euclid`](isize::div_euclid) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![7, 7, -7, -7]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![4, -4, 4, -4]).unwrap();
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, -1, -2, 2]).unwrap();
/// assert_eq!(a.div_euclid(&b), c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![7, -7]).unwrap();
/// let b: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![1, -2]).unwrap();
/// assert_eq!(a.div_euclid(4), b);
///```
pub trait DivEuclid<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the quotient of Euclidean division operation.
    fn div_euclid(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise remainder of Euclidean division operator.
/// 
/// Computes the least nonnegative remainder of `self (mod rhs)`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive integer (e.g. `i64`) using primitive
/// [`rem_euclid`](isize::rem_euclid) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![7, -7, 7, -7]).unwrap();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![4, 4, -4, -4]).unwrap();
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![3, 1, 3, 1]).unwrap();
/// assert_eq!(a.rem_euclid(&b), c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![7, -7]).unwrap();
/// let b: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![3, 1]).unwrap();
/// assert_eq!(a.rem_euclid(4), b);
///```
pub trait RemEuclid<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the remainder of Euclidean division operation.
    fn rem_euclid(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise max operator.
/// 
/// Computes the max of `self` and `rhs`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`max`](f32::max) function.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`Ord`](std::cmp::Ord).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![7.0, -3.0, 6.0, 12.0]).unwrap();
/// assert_eq!(a.max(&b), c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, 10.0, 5.0, 6.0]).unwrap();
/// assert_eq!(a.max(5.0), b);
///```
pub trait Max<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn max(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise min operator.
/// 
/// Computes the min of `self` and `rhs`.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`max`](f32::max) function.
/// 
/// Implemented for all tensors whose scalar type `T` implements
/// [`Ord`](std::cmp::Ord).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![5.0, -8.0, 1.0, 4.0]).unwrap();
/// assert_eq!(a.min(&b), c);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 5.0, 3.0, 5.0]).unwrap();
/// assert_eq!(a.min(5.0), b);
///```
pub trait Min<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn min(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise pow operator.
/// 
/// Raise `self` to `rhs` power.
/// 
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
/// If `Rhs` is a tensor, it should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`powi`](f32::powi)
/// [`powf`](f32::powf) functions.
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive integer (e.g. `i64`) using primitive
/// [`pow`](isize::pow) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 4, 9, 16]).unwrap();
/// assert_eq!(a.pow(2), c);
/// ```
pub trait Pow<Rhs=Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn pow(self, rhs: Rhs) -> Self::Output;
}

/// Elementwise exponential function.
/// 
/// Returns `e^(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`exp`](f32::exp).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// assert_eq!(a.exp(), b);
/// ```
pub trait Exp {
    /// Output type.
    type Output;
    /// Returns `e^(self)`, the (exponential function).
    fn exp(self) -> Self::Output;
}

/// Elementwise power of 2 function.
/// 
/// Returns `2^(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`exp2`](f32::exp2).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// assert_eq!(a.exp2(), b);
/// ```
pub trait Exp2 {
    /// Output type.
    type Output;
    /// Returns `2^(self)`.
    fn exp2(self) -> Self::Output;
}

/// Elementwise exponential function minus 1.
///
/// This is more accurate than if the operations were
/// performed separately even if `self` is close to zero.
/// 
/// Returns `exp(x) - 1`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`exp_m1`](f32::exp_m1).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::LN_2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![LN_2, 0.0, 0.0, LN_2]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a.exp_m1(), b);
/// ```
pub trait ExpM1 {
    /// Output type.
    type Output;
    /// Returns `e^(self) - 1` in a way that is accurate
    /// even if the number is close to zero.
    fn exp_m1(self) -> Self::Output;
}

/// Elementwise natural logarithm function.
/// 
/// Returns `ln(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`ln`](f32::ln).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a.ln(), b);
/// ```
pub trait Ln {
    /// Output type.
    type Output;
    /// Returns the natural logarithm of `self`.
    fn ln(self) -> Self::Output;
}

/// Elementwise natural logarithm of 1 plus x function.
///
/// This is more accurate than if the operations were performed separately.
/// 
/// Returns `ln(1 + self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`ln_1p`](f32::ln_1p).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, E - 1.0, E - 1.0, 0.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, 1.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a.ln_1p(), b);
/// ```
pub trait Ln1p {
    /// Output type.
    type Output;
    /// Returns `ln(1 + self)` more accurately than if the
    /// operation were performed separately.
    fn ln_1p(self) -> Self::Output;
}

/// Elementwise base 2 logarithm function.
/// 
/// Returns `log2(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`log2`](f32::log2).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// assert_eq!(a.log2(), b);
/// ```
pub trait Log2 {
    /// Output type.
    type Output;
    /// Returns the base 2 logarithm of `self`.
    fn log2(self) -> Self::Output;
}

/// Elementwise base 10 logarithm function.
/// 
/// Returns `log10(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`log10`](f32::log10).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![100.0, 1_000.0, 1_000.0, 100.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, 3.0, 3.0, 2.0]).unwrap();
/// assert_eq!(a.log10(), b);
/// ```
pub trait Log10 {
    /// Output type.
    type Output;
    /// Returns the base 10 logarithm of `self`.
    fn log10(self) -> Self::Output;
}

/// Elementwise sine function.
/// 
/// Returns `sin(self `(in radians)`)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`sin`](f32::sin).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// assert!(a.sin().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Sin {
    /// Output type.
    type Output;
    /// Returns the sine of `self`.
    fn sin(self) -> Self::Output;
}

/// Elementwise cosine function.
/// 
/// Returns `cos(self `(in radians)`)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`cos`](f32::cos).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// assert!(a.cos().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Cos {
    /// Output type.
    type Output;
    /// Returns the cosine of `self`.
    fn cos(self) -> Self::Output;
}

/// Elementwise tangent function.
/// 
/// Returns `tan(self `(in radians)`)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`tan`](f32::tan).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// assert!(a.tan().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Tan {
    /// Output type.
    type Output;
    /// Returns the tangent of `self`.
    fn tan(self) -> Self::Output;
}

/// Elementwise hyperbolic sine function.
/// 
/// Returns `sinh(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`sinh`](f32::sinh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// assert!(a.sinh().sub(&b).abs() < 1e-10);
/// ```
pub trait Sinh {
    /// Output type.
    type Output;
    /// Hyperbolic sine function
    fn sinh(self) -> Self::Output;
}

/// Elementwise hyperbolic cosine function.
/// 
/// Returns `cosh(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`cosh`](f32::cosh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// assert!(a.cosh().sub(&b).abs() < 1e-10);
/// ```
pub trait Cosh {
    /// Output type.
    type Output;
    /// Hyperbolic cosine function.
    fn cosh(self) -> Self::Output;
}

/// Elementwise hyperbolic tangent function.
/// 
/// Returns `tanh(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`tanh`](f32::tanh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// assert!(a.tanh().sub(&b).abs() < 1e-10);
/// ```
pub trait Tanh {
    /// Output type.
    type Output;
    /// Hyperbolic tangent function.
    fn tanh(self) -> Self::Output;
}

/// Elementwise arcsine function.
/// 
/// Returns `asin(self)` (in radians) in [-pi/2, pi/2],
/// if `self` is in [-1, 1].
///
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`asin`](f32::asin).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// assert!(a.asin().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Asin {
    /// Output type.
    type Output;
    /// Returns the arcsine of `self`.
    fn asin(self) -> Self::Output;
}

/// Elementwise arccosine function.
/// 
/// Returns `acos(self)` (in radians) in [0, pi],
/// if `self` in [-1, 1].
/// 
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`acos`](f32::acos).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// assert!(a.acos().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Acos {
    /// Output type.
    type Output;
    /// Returns the arccosine of `self`.
    fn acos(self) -> Self::Output;
}

/// Elementwise arctangent function.
/// 
/// Returns `atan(self)` (in radians) in [-pi/2, pi/2].
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`atan`](f32::atan).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// assert!(a.atan().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Atan {
    /// Output type.
    type Output;
    /// Returns the arctangent of `self`.
    fn atan(self) -> Self::Output;
}

/// Elementwise inverse hyperbolic sine function.
/// 
/// Returns `asinh(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`asinh`](f32::asinh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(a.asinh().sub(&b).abs() < 1e-10);
/// ```
pub trait Asinh {
    /// Output type.
    type Output;
    /// Inverse hyperbolic sine function.
    fn asinh(self) -> Self::Output;
}

/// Elementwise inverse hyperbolic cosine function.
/// 
/// Returns `acosh(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`acosh`](f32::acosh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(a.acosh().sub(&b).abs() < 1e-10);
/// ```
pub trait Acosh {
    /// Output type.
    type Output;
    /// Inverse hyperbolic cosine function.
    fn acosh(self) -> Self::Output;
}

/// Elementwise inverse hyperbolic tangent function.
/// 
/// Returns `atanh(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`atanh`](f32::atanh).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(a.atanh().sub(&b).abs() < 1e-10);
/// ```
pub trait Atanh {
    /// Output type.
    type Output;
    /// Inverse hyperbolic tangent function.
    fn atanh(self) -> Self::Output;
}

/// Elementwise square root function.
/// 
/// Returns `sqrt(self)`.
/// 
/// Note: `sqrt(self)` is NaN if `self` is negative.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`sqrt`](f32::sqrt).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 4.0, 9.0, 16.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(a.sqrt(), b);
/// ```
pub trait Sqrt {
    /// Output type.
    type Output;
    /// Returns the square root of `self`.
    fn sqrt(self) -> Self::Output;
}

/// Elementwise cubic root function.
/// 
/// Returns `cbrt(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`cbrt`](f32::cbrt).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 8.0, 64.0, 512.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 4.0, 8.0]).unwrap();
/// assert_eq!(a.cbrt(), b);
/// ```
pub trait Cbrt {
    /// Output type.
    type Output;
    /// Returns the cubic root of `self`.
    fn cbrt(self) -> Self::Output;
}

/// Elementwise absolute value function.
/// 
/// Returns `abs(self)`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`abs`](f32::abs).
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive signed integer (e.g. `i64`) using primitive
/// [`abs`](isize::abs) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -6.0, 3.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 6.0, 3.0]).unwrap();
/// assert_eq!(a.abs(), b);
/// ```
pub trait Abs {
    /// Output type.
    type Output;
    /// Returns the absolute value of `self`.
    fn abs(self) -> Self::Output;
}

/// Elementwise sign function.
/// 
/// Returns:
/// * `1` if `self` is positive,
/// * `-1` id `self` is negative.
///
/// Note that for signed integers, `0` -> `0`. However,
/// for floats, `+0.0` -> `1.0` and `-0.0` -> `-1.0`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`signum`](f32::signum).
/// 
/// Implemented for all tensors whose scalar type is
/// a primitive signed integer (e.g. `i64`) using primitive
/// [`signum`](isize::signum) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, -2.0, -6.0, 0.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -1.0, -1.0, 1.0]).unwrap();
/// assert_eq!(a.signum(), b);
/// ```
pub trait Signum {
    /// Output type.
    type Output;
    /// Returns the sign of `self`.
    fn signum(self) -> Self::Output;
}

/// Elementwise ceiling function.
/// 
/// Returns `ceil(self)`.
/// 
/// Note: `ceil(x)` is the smallest integer greater than or equal to `x`. 
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`ceil`](f32::ceil).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, -2.0, -6.0, 3.0]).unwrap();
/// assert_eq!(a.ceil(), b);
/// ```
pub trait Ceil {
    /// Output type.
    type Output;
    /// Returns the ceiling of `self`.
    fn ceil(self) -> Self::Output;
}

/// Elementwise floor function.
/// 
/// Returns `floor(self)`.
/// 
/// Note: `floor(x)` is the greatest integer smaller than or equal to `x`. 
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`floor`](f32::floor).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -3.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a.floor(), b);
/// ```
pub trait Floor {
    /// Output type.
    type Output;
    /// Returns the floor of `self`.
    fn floor(self) -> Self::Output;
}

/// Elementwise rounding function.
/// 
/// Returns `round(self)`.
/// 
/// Note: `round(x)` is the nearest integer to `x`. Half-way cases
/// are rounded away from zero.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`round`](f32::round).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a.round(), b);
/// ```
pub trait Round {
    /// Output type.
    type Output;
    /// Returns the rounding of `self`.
    fn round(self) -> Self::Output;
}

/// Elementwise inverse function.
/// 
/// Returns `1 / self`.
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`recip`](f32::recip).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, 0.5, 4.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -0.5, 2.0, 0.25]).unwrap();
/// assert!(a.recip().sub(&b).abs() < 1e-10);
/// ```
pub trait Recip {
    /// Output type.
    type Output;
    /// Returns the inverse of `self`.
    fn recip(self) -> Self::Output;
}

/// Elementwise multiplicative inverse function.
/// 
/// Returns `1 / self`.
///  
/// Implemented for all tensors whose scalar type implements
/// the [`Field`](crate::algebra::Field) trait. For primitive
/// `f32` and `f64` types, this is strictly equivalent to
/// [`Recip_`](Recip_).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -2.0, 0.5, 4.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -0.5, 2.0, 0.25]).unwrap();
/// assert!(a.minv().sub(&b).abs() < 1e-10);
/// ```
pub trait Minv {
    /// Output type.
    type Output;
    /// Returns the multiplicative inverse of `self`.
    fn minv(self) -> Self::Output;
}

/// Elementwise conversion to degrees function.
/// 
/// Returns `self` (in degrees).
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`to_degrees`](f32::to_degrees).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// assert!(a.to_degrees().sub(&b).abs() < 1e-10);
/// ```
pub trait ToDegrees {
    /// Output type.
    type Output;
    /// Converts radians to degrees.
    fn to_degrees(self) -> Self::Output;
}

/// Elementwise conversion to radians function.
/// 
/// Returns `self` (in radians).
///  
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`to_radians`](f32::to_radians).
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// assert!(a.to_radians().sub(&b).abs() < 1e-10);
/// ```
pub trait ToRadians {
    /// Output type.
    type Output;
    /// Converts degrees to radians.
    fn to_radians(self) -> Self::Output;
}

/// In-place elementwise linear transformation operator.
///
/// `self -> (self * rhs0) + rhs1`
///
/// Computations only involve one rounding error, yielding
/// a more accurate result than an unfused multiply-add.
///
/// Using mul_add can be more performant than an unfused
/// multiply-add if the target architecture has a dedicated
/// fma CPU instruction.
///
/// 
/// Note that `Rhs0` and `Rhs1` are `Self` by default, but this is not mandatory.
/// If `Rhs0` or `Rhs1` are a tensor, they should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`mul_add`](f32::mul_add) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, 2.0, 2.0, 2.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// a.mul_add_(&b, &c);
/// let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a, d);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.mul_add_(2.0, 1.0);
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a, b);
///```
pub trait MulAdd_<Rhs0=Self, Rhs1=Self> {
    /// Fused in-place multiply-add.
    fn mul_add_(&mut self, rhs0: Rhs0, rhs1: Rhs1);
}

/// Elementwise linear transformation operator.
/// 
/// Returns `(self * rhs0) + rhs1`.
///
/// Computations only involve one rounding error, yielding
/// a more accurate result than an unfused multiply-add.
///
/// Using mul_add can be more performant than an unfused
/// multiply-add if the target architecture has a dedicated
/// fma CPU instruction.
/// 
/// Note that `Rhs0` and `Rhs1` are `Self` by default, but this is not mandatory.
/// If `Rhs0` or `Rhs1` are a tensor, they should have a shape compatible with
/// the shape of `Self`.
/// 
/// Implemented for all tensors whose scalar type is
/// `f32` or `f64` using primitive [`mul_add`](f32::mul_add) function.
///
/// # Examples
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, 2.0, 2.0, 2.0]).unwrap();
/// let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a.mul_add(&b, &c), d);
/// ```
///
/// ```
/// use melange_scratch::prelude::*;
/// use typenum::U2;
///
/// let mut a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a.mul_add(2.0, 1.0), b);
///```
pub trait MulAdd<Rhs0=Self, Rhs1=Self> {
    /// Output type.
    type Output;
    /// Returns `(self * rhs0) + rhs1`.
    fn mul_add(self, rhs0: Rhs0, rhs1: Rhs1) -> Self::Output;
}

macro_rules! inplace_op_trait_impl_binary {
    (
        $trait_name:ident; $fn_name:ident; $(where $generic:ident: $($bound:path),*;)? $(for $scalar_type:ty;)?
        ($x:ident, $y:ident) => {$scalar_op:stmt}
    ) => {
        macro_rules! op_unchecked {
            (($self:ident, $rhs:ident, $x_:ident, $y_:ident) => {$scalar_op_:stmt}) => {
                let chunk_size = $self.opt_chunk_size().min($rhs.opt_chunk_size());

                let mut it = $self.strided_iter_mut(chunk_size).streaming_zip($rhs.strided_iter(chunk_size));
                while let Some((self_chunk, rhs_chunk)) = it.next() {
                    for ($x_, $y_) in self_chunk.iter_mut().zip(rhs_chunk.iter()) {
                        $scalar_op_
                    }
                }
            };
        }

        impl<Y, Z, $($generic,)? S, A, D, L, Yrhs, Zrhs, Arhs, Drhs, Lrhs> $trait_name<&Tensor<Static, Yrhs, Zrhs, $($generic,)? $($scalar_type,)? S, Arhs, Drhs, Lrhs>> for Tensor<Static, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            for<'a> &'a Tensor<Static, Yrhs, Zrhs, $($generic,)? $($scalar_type,)? S, Arhs, Drhs, Lrhs>: StridedIterator<Item=&'a [$($generic)? $($scalar_type)?]>,
            $(T: $($bound+)* 'static,)?
            S: Shape,
            L: Layout<S::Len>,
            Lrhs: Layout<S::Len>,
        {  
            fn $fn_name(&mut self, rhs: &Tensor<Static, Yrhs, Zrhs, $($generic,)? $($scalar_type,)? S, Arhs, Drhs, Lrhs>) {
                op_unchecked! { (self, rhs, $x, $y) => { $scalar_op } }
            }
        }

        impl<Y, Z, $($generic,)? S, A, D, L, Yrhs, Zrhs, Srhs, Arhs, Drhs, Lrhs> $trait_name<&Tensor<Dynamic, Yrhs, Zrhs, $($generic,)? $($scalar_type,)? Srhs, Arhs, Drhs, Lrhs>> for Tensor<Dynamic, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            for<'a> &'a Tensor<Dynamic, Yrhs, Zrhs, $($generic,)? $($scalar_type,)? Srhs, Arhs, Drhs, Lrhs>: StridedIterator<Item=&'a [$($generic)? $($scalar_type)?]>,
            $(T: $($bound+)* 'static,)?
            S: Shape + Same<Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            Srhs: Shape,
            L: Layout<S::Len>,
            Lrhs: Layout<Srhs::Len>,
        {
            fn $fn_name(&mut self, rhs: &Tensor<Dynamic, Yrhs, Zrhs, $($generic,)? $($scalar_type,)? Srhs, Arhs, Drhs, Lrhs>) {
                assert_shape_eq!(self.shape().deref(), rhs.shape().deref());
                op_unchecked! { (self, rhs, $x, $y) => { $scalar_op } }
            }
        }
    };
}

inplace_op_trait_impl_binary! { Add_; add_; where T: Copy, AddAssign; (x, y) => { *x += *y } }
inplace_op_trait_impl_binary! { Sub_; sub_; where T: Copy, SubAssign; (x, y) => { *x -= *y } }
inplace_op_trait_impl_binary! { Mul_; mul_; where T: Copy, MulAssign; (x, y) => { *x *= *y } }
inplace_op_trait_impl_binary! { Div_; div_; where T: Copy, DivAssign; (x, y) => { *x /= *y } }
inplace_op_trait_impl_binary! { Rem_; rem_; where T: Copy, RemAssign; (x, y) => { *x %= *y } }
// FIX REQUIRED
// Conflicts with Max and Min for f64 and f32
// inplace_op_trait_impl_binary! { Max_; max_; where T: Copy, Ord; (x, y) => { *x = (*x).max(*y) } }
// inplace_op_trait_impl_binary! { Min_; min_; where T: Copy, Ord; (x, y) => { *x = (*x).min(*y) } }
inplace_op_trait_impl_binary! { Atan2_; atan2_; for f64; (x, y) => { *x = x.atan2(*y) } }
inplace_op_trait_impl_binary! { Copysign_; copysign_; for f64; (x, y) => { *x = x.copysign(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for f64; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { Max_; max_; for f64; (x, y) => { *x = x.max(*y) } }
inplace_op_trait_impl_binary! { Min_; min_; for f64; (x, y) => { *x = x.min(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for f64; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { Atan2_; atan2_; for f32; (x, y) => { *x = x.atan2(*y) } }
inplace_op_trait_impl_binary! { Copysign_; copysign_; for f32; (x, y) => { *x = x.copysign(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for f32; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { Max_; max_; for f32; (x, y) => { *x = x.max(*y) } }
inplace_op_trait_impl_binary! { Min_; min_; for f32; (x, y) => { *x = x.min(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for f32; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for u128; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for u128; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for u64; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for u64; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for u32; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for u32; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for u16; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for u16; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for u8; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for u8; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for i128; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for i128; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for i64; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for i64; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for i32; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for i32; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for i16; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for i16; (x, y) => { *x = x.rem_euclid(*y) } }
inplace_op_trait_impl_binary! { DivEuclid_; div_euclid_; for i8; (x, y) => { *x = x.div_euclid(*y) } }
inplace_op_trait_impl_binary! { RemEuclid_; rem_euclid_; for i8; (x, y) => { *x = x.rem_euclid(*y) } }

macro_rules! inplace_op_trait_impl_scalar {
    (
        $trait_name:ident<$param_type:ty>; $fn_name:ident; $(where $generic:ident: $($bound:path),*;)? $(for $scalar_type:ty;)?
        ($x:ident, $rhs:ident) => {$scalar_op:stmt}
    ) => {
        impl<X, Y, Z, $($generic,)? S, A, D, L> $trait_name<$param_type> for Tensor<X, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            $(T: $($bound+)* 'static,)?
            S: Shape,
            L: Layout<S::Len>,
        {
            fn $fn_name(&mut self, $rhs: $param_type) {
                let chunk_size = self.opt_chunk_size();

                let mut it = self.strided_iter_mut(chunk_size);
                while let Some(chunk) = it.next() {
                    for $x in chunk.iter_mut() {
                        $scalar_op
                    }
                }
            }
        }
    };
}

inplace_op_trait_impl_scalar! { Add_<T>; add_; where T: Copy, AddAssign; (x, rhs) => { *x += rhs } }
inplace_op_trait_impl_scalar! { Sub_<T>; sub_; where T: Copy, SubAssign; (x, rhs) => { *x -= rhs } }
inplace_op_trait_impl_scalar! { Mul_<T>; mul_; where T: Copy, MulAssign; (x, rhs) => { *x *= rhs } }
inplace_op_trait_impl_scalar! { Div_<T>; div_; where T: Copy, DivAssign; (x, rhs) => { *x /= rhs } }
inplace_op_trait_impl_scalar! { Rem_<T>; rem_; where T: Copy, RemAssign; (x, rhs) => { *x %= rhs } }
inplace_op_trait_impl_scalar! { Atan2_<f64>; atan2_; for f64; (x, rhs) => { *x = x.atan2(rhs) } }
inplace_op_trait_impl_scalar! { Copysign_<f64>; copysign_; for f64; (x, rhs) => { *x = x.copysign(rhs) } }
// FIX REQUIRED
// Conflicts with Max and Min for f64 and f32
// inplace_op_trait_impl_scalar! { Max_<T>; max_; where T: Copy, Ord; (x, rhs) => { *x = (*x).max(rhs) } }
// inplace_op_trait_impl_scalar! { Min_<T>; min_; where T: Copy, Ord; (x, rhs) => { *x = (*x).min(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<f64>; div_euclid_; for f64; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { Max_<f64>; max_; for f64; (x, rhs) => { *x = x.max(rhs) } }
inplace_op_trait_impl_scalar! { Min_<f64>; min_; for f64; (x, rhs) => { *x = x.min(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<f64>; pow_; for f64; (x, rhs) => { *x = x.powf(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<f64>; rem_euclid_; for f64; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<i32>; pow_; for f64; (x, rhs) => { *x = x.powi(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<f32>; div_euclid_; for f32; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { Max_<f32>; max_; for f32; (x, rhs) => { *x = x.max(rhs) } }
inplace_op_trait_impl_scalar! { Min_<f32>; min_; for f32; (x, rhs) => { *x = x.min(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<f32>; pow_; for f32; (x, rhs) => { *x = x.powf(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<f32>; rem_euclid_; for f32; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<i32>; pow_; for f32; (x, rhs) => { *x = x.powi(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for u128; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for u64; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for u32; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for u16; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for u8; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for i128; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for i64; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for i32; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for i16; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { Pow_<u32>; pow_; for i8; (x, rhs) => { *x = x.pow(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<u128>; div_euclid_; for u128; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<u128>; rem_euclid_; for u128; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<u64>; div_euclid_; for u64; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<u64>; rem_euclid_; for u64; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<u32>; div_euclid_; for u32; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<u32>; rem_euclid_; for u32; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<u16>; div_euclid_; for u16; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<u16>; rem_euclid_; for u16; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<u8>; div_euclid_; for u8; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<u8>; rem_euclid_; for u8; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<i128>; div_euclid_; for i128; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<i128>; rem_euclid_; for i128; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<i64>; div_euclid_; for i64; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<i64>; rem_euclid_; for i64; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<i32>; div_euclid_; for i32; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<i32>; rem_euclid_; for i32; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<i16>; div_euclid_; for i16; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<i16>; rem_euclid_; for i16; (x, rhs) => { *x = x.rem_euclid(rhs) } }
inplace_op_trait_impl_scalar! { DivEuclid_<i8>; div_euclid_; for i8; (x, rhs) => { *x = x.div_euclid(rhs) } }
inplace_op_trait_impl_scalar! { RemEuclid_<i8>; rem_euclid_; for i8; (x, rhs) => { *x = x.rem_euclid(rhs) } }

macro_rules! inplace_fn_trait_impl {
    (
        $trait_name:ident; $fn_name:ident; $(where $generic:ident: $($bound:path),*;)? $(for $scalar_type:ty;)?
        ($x:ident) => {$scalar_op:stmt}
    ) => {
        impl<X, Y, Z, $($generic,)? S, A, D, L> $trait_name for Tensor<X, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            $(T: $($bound+)* 'static,)?
            S: Shape,
            L: Layout<S::Len>,
        {
            fn $fn_name(&mut self) {
                let chunk_size = self.opt_chunk_size();

                let mut it = self.strided_iter_mut(chunk_size);
                while let Some(chunk) = it.next() {
                    for $x in chunk.iter_mut() {
                        $scalar_op
                    }
                }
            }
        }
    };
}

inplace_fn_trait_impl! { Exp_; exp_; for f64; (x) => { *x = x.exp() } }
inplace_fn_trait_impl! { Exp2_; exp2_; for f64; (x) => { *x = x.exp2() } }
inplace_fn_trait_impl! { ExpM1_; exp_m1_; for f64; (x) => { *x = x.exp_m1() } }
inplace_fn_trait_impl! { Ln_; ln_; for f64; (x) => { *x = x.ln() } }
inplace_fn_trait_impl! { Ln1p_; ln_1p_; for f64; (x) => { *x = x.ln_1p() } }
inplace_fn_trait_impl! { Log2_; log2_; for f64; (x) => { *x = x.log2() } }
inplace_fn_trait_impl! { Log10_; log10_; for f64; (x) => { *x = x.log10() } }
inplace_fn_trait_impl! { Sin_; sin_; for f64; (x) => { *x = x.sin() } }
inplace_fn_trait_impl! { Cos_; cos_; for f64; (x) => { *x = x.cos() } }
inplace_fn_trait_impl! { Tan_; tan_; for f64; (x) => { *x = x.tan() } }
inplace_fn_trait_impl! { Sinh_; sinh_; for f64; (x) => { *x = x.sinh() } }
inplace_fn_trait_impl! { Cosh_; cosh_; for f64; (x) => { *x = x.cosh() } }
inplace_fn_trait_impl! { Tanh_; tanh_; for f64; (x) => { *x = x.tanh() } }
inplace_fn_trait_impl! { Asin_; asin_; for f64; (x) => { *x = x.asin() } }
inplace_fn_trait_impl! { Acos_; acos_; for f64; (x) => { *x = x.acos() } }
inplace_fn_trait_impl! { Atan_; atan_; for f64; (x) => { *x = x.atan() } }
inplace_fn_trait_impl! { Asinh_; asinh_; for f64; (x) => { *x = x.asinh() } }
inplace_fn_trait_impl! { Acosh_; acosh_; for f64; (x) => { *x = x.acosh() } }
inplace_fn_trait_impl! { Atanh_; atanh_; for f64; (x) => { *x = x.atanh() } }
inplace_fn_trait_impl! { Sqrt_; sqrt_; for f64; (x) => { *x = x.sqrt() } }
inplace_fn_trait_impl! { Cbrt_; cbrt_; for f64; (x) => { *x = x.cbrt() } }
inplace_fn_trait_impl! { Abs_; abs_; for f64; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for f64; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Ceil_; ceil_; for f64; (x) => { *x = x.ceil() } }
inplace_fn_trait_impl! { Floor_; floor_; for f64; (x) => { *x = x.floor() } }
inplace_fn_trait_impl! { Round_; round_; for f64; (x) => { *x = x.round() } }
inplace_fn_trait_impl! { Recip_; recip_; for f64; (x) => { *x = x.recip() } }
inplace_fn_trait_impl! { ToDegrees_; to_degrees_; for f64; (x) => { *x = x.to_degrees() } }
inplace_fn_trait_impl! { ToRadians_; to_radians_; for f64; (x) => { *x = x.to_radians() } }
inplace_fn_trait_impl! { Exp_; exp_; for f32; (x) => { *x = x.exp() } }
inplace_fn_trait_impl! { Exp2_; exp2_; for f32; (x) => { *x = x.exp2() } }
inplace_fn_trait_impl! { ExpM1_; exp_m1_; for f32; (x) => { *x = x.exp() } }
inplace_fn_trait_impl! { Ln_; ln_; for f32; (x) => { *x = x.ln() } }
inplace_fn_trait_impl! { Ln1p_; ln_1p_; for f32; (x) => { *x = x.ln_1p() } }
inplace_fn_trait_impl! { Log2_; log2_; for f32; (x) => { *x = x.log2() } }
inplace_fn_trait_impl! { Log10_; log10_; for f32; (x) => { *x = x.log10() } }
inplace_fn_trait_impl! { Sin_; sin_; for f32; (x) => { *x = x.sin() } }
inplace_fn_trait_impl! { Cos_; cos_; for f32; (x) => { *x = x.cos() } }
inplace_fn_trait_impl! { Tan_; tan_; for f32; (x) => { *x = x.tan() } }
inplace_fn_trait_impl! { Sinh_; sinh_; for f32; (x) => { *x = x.sinh() } }
inplace_fn_trait_impl! { Cosh_; cosh_; for f32; (x) => { *x = x.cosh() } }
inplace_fn_trait_impl! { Tanh_; tanh_; for f32; (x) => { *x = x.tanh() } }
inplace_fn_trait_impl! { Asin_; asin_; for f32; (x) => { *x = x.asin() } }
inplace_fn_trait_impl! { Acos_; acos_; for f32; (x) => { *x = x.acos() } }
inplace_fn_trait_impl! { Atan_; atan_; for f32; (x) => { *x = x.atan() } }
inplace_fn_trait_impl! { Asinh_; asinh_; for f32; (x) => { *x = x.asinh() } }
inplace_fn_trait_impl! { Acosh_; acosh_; for f32; (x) => { *x = x.acosh() } }
inplace_fn_trait_impl! { Atanh_; atanh_; for f32; (x) => { *x = x.atanh() } }
inplace_fn_trait_impl! { Sqrt_; sqrt_; for f32; (x) => { *x = x.sqrt() } }
inplace_fn_trait_impl! { Cbrt_; cbrt_; for f32; (x) => { *x = x.cbrt() } }
inplace_fn_trait_impl! { Abs_; abs_; for f32; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for f32; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Ceil_; ceil_; for f32; (x) => { *x = x.ceil() } }
inplace_fn_trait_impl! { Floor_; floor_; for f32; (x) => { *x = x.floor() } }
inplace_fn_trait_impl! { Round_; round_; for f32; (x) => { *x = x.round() } }
inplace_fn_trait_impl! { Recip_; recip_; for f32; (x) => { *x = x.recip() } }
inplace_fn_trait_impl! { ToDegrees_; to_degrees_; for f32; (x) => { *x = x.to_degrees() } }
inplace_fn_trait_impl! { ToRadians_; to_radians_; for f32; (x) => { *x = x.to_radians() } }
inplace_fn_trait_impl! { Abs_; abs_; for i128; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for i128; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Abs_; abs_; for i64; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for i64; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Abs_; abs_; for i32; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for i32; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Abs_; abs_; for i16; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for i16; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Abs_; abs_; for i8; (x) => { *x = x.abs() } }
inplace_fn_trait_impl! { Signum_; signum_; for i8; (x) => { *x = x.signum() } }
inplace_fn_trait_impl! { Minv_; minv_; where T: Field, Div<Output=T>, Copy; (x) => { *x = x.minv() } }

macro_rules! inplace_op_trait_impl_scalar2 {
    (
        $trait_name:ident<$param_type0:ty, $param_type1:ty>; $fn_name:ident; $(where $generic:ident: $($bound:path),*;)? $(for $scalar_type:ty;)?
        ($x:ident, $rhs0:ident, $rhs1:ident) => {$scalar_op:stmt}
    ) => {
        impl<X, Y, Z, $($generic,)? S, A, D, L> $trait_name<$param_type0, $param_type1> for Tensor<X, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            $(T: $($bound+)* 'static,)?
            S: Shape,
            L: Layout<S::Len>,
        {
            fn $fn_name(&mut self, $rhs0: $param_type0, $rhs1: $param_type1) {
                let chunk_size = self.opt_chunk_size();

                let mut it = self.strided_iter_mut(chunk_size);
                while let Some(chunk) = it.next() {
                    for $x in chunk.iter_mut() {
                        $scalar_op
                    }
                }
            }
        }
    };
}

inplace_op_trait_impl_scalar2! { MulAdd_<f64, f64>; mul_add_; for f64; (x, rhs0, rhs1) => { *x = x.mul_add(rhs0, rhs1) } }
inplace_op_trait_impl_scalar2! { MulAdd_<f32, f32>; mul_add_; for f32; (x, rhs0, rhs1) => { *x = x.mul_add(rhs0, rhs1) } }

macro_rules! inplace_op_trait_impl_ternary {
    (
        $trait_name:ident; $fn_name:ident; $(where $generic:ident: $($bound:path),*;)? $(for $scalar_type:ty;)?
        ($x:ident, $y0:ident, $y1:ident) => {$scalar_op:stmt}
    ) => {
        macro_rules! op_unchecked {
            (($self:ident, $rhs0:ident, $rhs1:ident, $x_:ident, $y0_:ident, $y1_:ident) => {$scalar_op_:stmt}) => {
                let chunk_size = $self.opt_chunk_size().min($rhs0.opt_chunk_size().min($rhs1.opt_chunk_size()));

                let mut it = $self.strided_iter_mut(chunk_size).streaming_zip($rhs0.strided_iter(chunk_size)).streaming_zip($rhs1.strided_iter(chunk_size));
                while let Some(((self_chunk, rhs0_chunk), rhs1_chunk)) = it.next() {
                    for (($x_, $y0_), $y1_) in self_chunk.iter_mut().zip(rhs0_chunk.iter()).zip(rhs1_chunk.iter()) {
                        $scalar_op_
                    }
                }
            };
        }

        impl<Y, Z, $($generic,)? S, A, D, L, Yrhs0, Zrhs0, Arhs0, Drhs0, Lrhs0, Yrhs1, Zrhs1, Arhs1, Drhs1, Lrhs1> $trait_name<&Tensor<Static, Yrhs0, Zrhs0, $($generic,)? $($scalar_type,)? S, Arhs0, Drhs0, Lrhs0>, &Tensor<Static, Yrhs1, Zrhs1, $($generic,)? $($scalar_type,)? S, Arhs1, Drhs1, Lrhs1>> for Tensor<Static, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            for<'a> &'a Tensor<Static, Yrhs0, Zrhs0, $($generic,)? $($scalar_type,)? S, Arhs0, Drhs0, Lrhs0>: StridedIterator<Item=&'a [$($generic)? $($scalar_type)?]>,
            for<'a> &'a Tensor<Static, Yrhs1, Zrhs1, $($generic,)? $($scalar_type,)? S, Arhs1, Drhs1, Lrhs1>: StridedIterator<Item=&'a [$($generic)? $($scalar_type)?]>,
            $(T: $($bound+)* 'static,)?
            S: Shape,
            L: Layout<S::Len>,
            Lrhs0: Layout<S::Len>,
            Lrhs1: Layout<S::Len>,
        {
            fn $fn_name(
                &mut self,
                rhs0: &Tensor<Static, Yrhs0, Zrhs0, $($generic,)? $($scalar_type,)? S, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Static, Yrhs1, Zrhs1, $($generic,)? $($scalar_type,)? S, Arhs1, Drhs1, Lrhs1>,
            ) {
                op_unchecked! { (self, rhs0, rhs1, $x, $y0, $y1) => { $scalar_op } }
            }
        }

        impl<Y, Z, $($generic,)? S, A, D, L, Yrhs0, Zrhs0, Srhs0, Arhs0, Drhs0, Lrhs0, Yrhs1, Zrhs1, Arhs1, Srhs1, Drhs1, Lrhs1> $trait_name<&Tensor<Dynamic, Yrhs0, Zrhs0, $($generic,)? $($scalar_type,)? Srhs0, Arhs0, Drhs0, Lrhs0>, &Tensor<Dynamic, Yrhs1, Zrhs1, $($generic,)? $($scalar_type,)? Srhs1, Arhs1, Drhs1, Lrhs1>> for Tensor<Dynamic, Y, Z, $($generic,)? $($scalar_type,)? S, A, D, L>
        where
            for<'a> &'a mut Self: StridedIteratorMut<Item=RefMutGat<[$($generic)? $($scalar_type)?]>>,
            for<'a> &'a Tensor<Dynamic, Yrhs0, Zrhs0, $($generic,)? $($scalar_type,)? Srhs0, Arhs0, Drhs0, Lrhs0>: StridedIterator<Item=&'a [$($generic)? $($scalar_type)?]>,
            for<'a> &'a Tensor<Dynamic, Yrhs1, Zrhs1, $($generic,)? $($scalar_type,)? Srhs1, Arhs1, Drhs1, Lrhs1>: StridedIterator<Item=&'a [$($generic)? $($scalar_type)?]>,
            S: Shape + Same<Srhs0> + Same<Srhs1>,
            <S as Same<Srhs0>>::Output: TRUE,
            <S as Same<Srhs1>>::Output: TRUE,
            Srhs0: Shape,
            Srhs1: Shape,
            L: Layout<S::Len>,
            Lrhs0: Layout<Srhs0::Len>,
            Lrhs1: Layout<Srhs1::Len>,
        {
            fn $fn_name(
                &mut self,
                rhs0: &Tensor<Dynamic, Yrhs0, Zrhs0, $($generic,)? $($scalar_type,)? Srhs0, Arhs0, Drhs0, Lrhs0>,
                rhs1: &Tensor<Dynamic, Yrhs1, Zrhs1, $($generic,)? $($scalar_type,)? Srhs1, Arhs1, Drhs1, Lrhs1>,
            ) {
                assert_shape_eq!(self.shape().deref(), rhs0.shape().deref());
                assert_shape_eq!(self.shape().deref(), rhs1.shape().deref());
                op_unchecked! { (self, rhs0, rhs1, $x, $y0, $y1) => { $scalar_op } }
            }
        }
    };
}

inplace_op_trait_impl_ternary! { MulAdd_; mul_add_; for f64; (x, y0, y1) => { *x = x.mul_add(*y0, *y1) } }
inplace_op_trait_impl_ternary! { MulAdd_; mul_add_; for f32; (x, y0, y1) => { *x = x.mul_add(*y0, *y1) } }

macro_rules! op_trait_impl {
    (
        $trait_name:ident; $fn_name:ident; $inplace_trait_name:ident; $inplace_fn_name:ident
    ) => {
        impl<'a, X, Y, Z, T, S, A, D, L, Rhs> $trait_name<Rhs> for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Tensor<X, Y, Z, T, S, A, D, L>: AllocLike,
            <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc: $inplace_trait_name<Rhs>,
        {
            type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc;
            fn $fn_name(
                self,
                rhs: Rhs,
            ) -> Self::Output {
                let mut out = self.to_contiguous();
                out.$inplace_fn_name(rhs);

                out
            }
        }
    };
}

op_trait_impl! { Add; add; Add_; add_ }
op_trait_impl! { Sub; sub; Sub_; sub_ }
op_trait_impl! { Mul; mul; Mul_; mul_ }
op_trait_impl! { Div; div; Div_; div_ }
op_trait_impl! { Rem; rem; Rem_; rem_ }
op_trait_impl! { Atan2; atan2; Atan2_; atan2_ }
op_trait_impl! { Copysign; copysign; Copysign_; copysign_ }
op_trait_impl! { DivEuclid; div_euclid; DivEuclid_; div_euclid_ }
op_trait_impl! { Max; max; Max_; max_ }
op_trait_impl! { Min; min; Min_; min_ }
op_trait_impl! { RemEuclid; rem_euclid; RemEuclid_; rem_euclid_ }
op_trait_impl! { Pow; pow; Pow_; pow_ }

macro_rules! fn_trait_impl {
    (
        $trait_name:ident; $fn_name:ident; $inplace_trait_name:ident; $inplace_fn_name:ident
    ) => {
        impl<'a, X, Y, Z, T, S, A, D, L> $trait_name for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Tensor<X, Y, Z, T, S, A, D, L>: AllocLike,
            <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc: $inplace_trait_name,
        {
            type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc;
            fn $fn_name(self) -> Self::Output {
                let mut out = self.to_contiguous();
                out.$inplace_fn_name();

                out
            }
        }
    };
}

fn_trait_impl! { Exp; exp; Exp_; exp_ }
fn_trait_impl! { Exp2; exp2; Exp2_; exp2_ }
fn_trait_impl! { ExpM1; exp_m1; ExpM1_; exp_m1_ }
fn_trait_impl! { Ln; ln; Ln_; ln_ }
fn_trait_impl! { Ln1p; ln_1p; Ln1p_; ln_1p_ }
fn_trait_impl! { Log2; log2; Log2_; log2_ }
fn_trait_impl! { Log10; log10; Log10_; log10_ }
fn_trait_impl! { Sin; sin; Sin_; sin_ }
fn_trait_impl! { Cos; cos; Cos_; cos_ }
fn_trait_impl! { Tan; tan; Tan_; tan_ }
fn_trait_impl! { Sinh; sinh; Sinh_; sinh_ }
fn_trait_impl! { Cosh; cosh; Cosh_; cosh_ }
fn_trait_impl! { Tanh; tanh; Tanh_; tanh_ }
fn_trait_impl! { Asin; asin; Asin_; asin_ }
fn_trait_impl! { Acos; acos; Acos_; acos_ }
fn_trait_impl! { Atan; atan; Atan_; atan_ }
fn_trait_impl! { Asinh; asinh; Asinh_; asinh_ }
fn_trait_impl! { Acosh; acosh; Acosh_; acosh_ }
fn_trait_impl! { Atanh; atanh; Atanh_; atanh_ }
fn_trait_impl! { Sqrt; sqrt; Sqrt_; sqrt_ }
fn_trait_impl! { Cbrt; cbrt; Cbrt_; cbrt_ }
fn_trait_impl! { Abs; abs; Abs_; abs_ }
fn_trait_impl! { Signum; signum; Signum_; signum_ }
fn_trait_impl! { Ceil; ceil; Ceil_; ceil_ }
fn_trait_impl! { Floor; floor; Floor_; floor_ }
fn_trait_impl! { Round; round; Round_; round_ }
fn_trait_impl! { Recip; recip; Recip_; recip_ }
fn_trait_impl! { ToDegrees; to_degrees; ToDegrees_; to_degrees_ }
fn_trait_impl! { ToRadians; to_radians; ToRadians_; to_radians_ }
fn_trait_impl! { Minv; minv; Minv_; minv_ }

macro_rules! op2_trait_impl {
    (
        $trait_name:ident; $fn_name:ident; $inplace_trait_name:ident; $inplace_fn_name:ident
    ) => {
        impl<'a, X, Y, Z, T, S, A, D, L, Rhs0, Rhs1> $trait_name<Rhs0, Rhs1> for &'a Tensor<X, Y, Z, T, S, A, D, L>
        where
            Tensor<X, Y, Z, T, S, A, D, L>: AllocLike,
            <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc: $inplace_trait_name<Rhs0, Rhs1>,
    {
        type Output = <Tensor<X, Y, Z, T, S, A, D, L> as AllocLike>::Alloc;
            fn $fn_name(
                self,
                rhs0: Rhs0,
                rhs1: Rhs1,
            ) -> Self::Output {
                let mut out = self.to_contiguous();
                out.$inplace_fn_name(rhs0, rhs1);

                out
            }
        }
    };
}

op2_trait_impl! { MulAdd; mul_add; MulAdd_; mul_add_ }
