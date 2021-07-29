//! Contains traits that wrap
//! common mathematical operations.
//!
//! It extends `[std::ops]`(std::ops).
//!
//! These traits make it possible to
//! write bounds on the capabilities
//! of types involved in specific
//! maths operations.
//!
//! They are available in two flavor:
//! * functional style ops such as `Add`
//! that allow backpropagating,
//! * in-place ops such as `AddAssign`
//! that are more efficient in terms of
//! memory footprint.
//!
//! They are implemented for primitive
//! numeric types, num_complex's
//! [`Complex64`] and [`Complex32`]
//! types and on all tensors whose
//! scalar type implements them.
//!
//! [`Complex64`]: https://docs.rs/num-complex/0.2.4/num_complex/type.Complex64.html
//! [`Complex32`]: https://docs.rs/num-complex/0.2.4/num_complex/type.Complex32.html

use num_complex::{Complex, Complex32, Complex64};

/// Four quadrant arctangent operator.
///
/// Computes the four quadrant arctengent of `self` (`y`)
/// and `rhs` (`x`) in radians as follows:
/// * `x = 0`, `y = 0`: `0`
/// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
/// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
/// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_4, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// assert!(a.atan2(&b).sub(&c).abs() < f64::EPSILON);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::FRAC_PI_2;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, -2.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2]).unwrap();
/// assert!(a.atan2(0.0).sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Atan2<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the four quadrant arctangent operation.
    fn atan2(self, rhs: Rhs) -> Self::Output;
}

/// Hypotenuse operator.
///
/// Calculates the length of the hypotenuse of a right-angle triangle
/// given legs of length `self` and `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::SQRT_2;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![SQRT_2, 1.0, 1.0, SQRT_2]).unwrap();
/// assert!(a.hypot(&b).sub(&c).abs() < f64::EPSILON);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::SQRT_2;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![SQRT_2, 1.0, 1.0, SQRT_2]).unwrap();
/// assert!(a.hypot(1.0).sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Hypot<Rhs = Self> {
    /// Output type.
    type Output;
    /// Calculates the length of the hypotenuse of a right-angle triangle
    /// given legs of length `self` and `rhs`.
    fn hypot(self, rhs: Rhs) -> Self::Output;
}

/// Copysign operator.
///
/// Result is composed of the magnitude of `self` and the sign of `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, -4.0, -4.0, 2.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -3.0, 4.0]).unwrap();
/// assert_eq!(a.copysign(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -1.0, 5.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, -2.0, -1.0, -5.0]).unwrap();
/// assert_eq!(a.copysign(-5.0), b);
/// ```
pub trait Copysign<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the copysign operation.
    fn copysign(self, rhs: Rhs) -> Self::Output;
}

/// Quotient of Euclidean division operator.
///
/// Computes the quotient of Euclidean division of `self` by `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![7, 7, -7, -7]).unwrap();
/// let b: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![4, -4, 4, -4]).unwrap();
/// let c: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, -1, -2, 2]).unwrap();
/// assert_eq!(a.div_euclid(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor1D<i32, 2> = Tensor::try_from(vec![7, -7]).unwrap();
/// let b: VecTensor1D<i32, 2> = Tensor::try_from(vec![1, -2]).unwrap();
/// assert_eq!(a.div_euclid(4), b);
/// ```
pub trait DivEuclid<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the quotient of Euclidean division operation.
    fn div_euclid(self, rhs: Rhs) -> Self::Output;
}

/// Remainder of Euclidean division operator.
///
/// Computes the least nonnegative remainder of `self (mod rhs)`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![7, -7, 7, -7]).unwrap();
/// let b: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![4, 4, -4, -4]).unwrap();
/// let c: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![3, 1, 3, 1]).unwrap();
/// assert_eq!(a.rem_euclid(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor1D<i32, 2> = Tensor::try_from(vec![7, -7]).unwrap();
/// let b: VecTensor1D<i32, 2> = Tensor::try_from(vec![3, 1]).unwrap();
/// assert_eq!(a.rem_euclid(4), b);
/// ```
pub trait RemEuclid<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the remainder of Euclidean division operation.
    fn rem_euclid(self, rhs: Rhs) -> Self::Output;
}

/// Max operator.
///
/// Computes the max of `self` and `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 6.0, 12.0]).unwrap();
/// assert_eq!(a.max(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, 10.0, 5.0, 6.0]).unwrap();
/// assert_eq!(a.max(5.0), b);
/// ```
pub trait Max<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn max(self, rhs: Rhs) -> Self::Output;
}

/// Min operator.
///
/// Computes the min of `self` and `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 1.0, 4.0]).unwrap();
/// assert_eq!(a.min(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 5.0, 3.0, 5.0]).unwrap();
/// assert_eq!(a.min(5.0), b);
/// ```
pub trait Min<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn min(self, rhs: Rhs) -> Self::Output;
}

/// Mask of max operator.
///
/// Computes the mask of max on `self`:
/// * `1` if `self` is the max,
/// * `0` otherwise.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a.max_mask(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 1.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a.max_mask(5.0), b);
/// ```
pub trait MaxMask<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn max_mask(self, rhs: Rhs) -> Self::Output;
}

/// Mask of min operator.
///
/// Computes the mask of max on `self`:
/// * `1` if `self` is the min,
/// * `0` otherwise.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 0.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a.min_mask(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a.min_mask(5.0), b);
/// ```
pub trait MinMask<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn min_mask(self, rhs: Rhs) -> Self::Output;
}

/// Power operator.
///
/// Raises `self` to `rhs` power.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let c: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, 4, 9, 16]).unwrap();
/// assert_eq!(a.pow(2), c);
/// ```
pub trait Pow<Rhs = Self> {
    /// Output type.
    type Output;
    /// Performs the max operation.
    fn pow(self, rhs: Rhs) -> Self::Output;
}

/// Arbitrary base logarithm function.
///
/// Returns the logarithm of `self` with respect to an arbitrary base.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, 9.0, 9.0, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 2.0, 1.0]).unwrap();
/// assert!(a.log(3.0).sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Log<Rhs = Self> {
    /// Output type.
    type Output;
    /// Returns the base `rhs` logarithm of `self`.
    fn log(self, rhs: Rhs) -> Self::Output;
}

/// Exponential function.
///
/// Returns `e^(self)`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// assert_eq!(a.exp(), b);
/// ```
pub trait Exp {
    /// Output type.
    type Output;
    /// Returns `e^(self)`, the (exponential function).
    fn exp(self) -> Self::Output;
}

/// Power of 2 function.
///
/// Returns `2^(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// assert_eq!(a.exp2(), b);
/// ```
pub trait Exp2 {
    /// Output type.
    type Output;
    /// Returns `2^(self)`.
    fn exp2(self) -> Self::Output;
}

/// Exponential function minus 1.
///
/// Returns `exp(x) - 1`.
///  
/// This is more accurate than if the operations were
/// performed separately even if `self` is close to zero.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::LN_2;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![LN_2, 0.0, 0.0, LN_2]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a.exp_m1(), b);
/// ```
pub trait ExpM1 {
    /// Output type.
    type Output;
    /// Returns `e^(self) - 1` in a way that is accurate
    /// even if the number is close to zero.
    fn exp_m1(self) -> Self::Output;
}

/// Natural logarithm function.
///
/// Returns `ln(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a.ln(), b);
/// ```
pub trait Ln {
    /// Output type.
    type Output;
    /// Returns the natural logarithm of `self`.
    fn ln(self) -> Self::Output;
}

/// Natural logarithm of 1 plus x function.
///
/// Returns `ln(1 + self)`.
///  
/// This is more accurate than if the operations were performed separately.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, E - 1.0, E - 1.0, 0.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 1.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a.ln_1p(), b);
/// ```
pub trait Ln1p {
    /// Output type.
    type Output;
    /// Returns `ln(1 + self)` more accurately than if the
    /// operation were performed separately.
    fn ln_1p(self) -> Self::Output;
}

/// Base 2 logarithm function.
///
/// Returns `log2(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// assert_eq!(a.log2(), b);
/// ```
pub trait Log2 {
    /// Output type.
    type Output;
    /// Returns the base 2 logarithm of `self`.
    fn log2(self) -> Self::Output;
}

/// Base 10 logarithm function.
///
/// Returns `log10(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![100.0, 1_000.0, 1_000.0, 100.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, 3.0, 3.0, 2.0]).unwrap();
/// assert_eq!(a.log10(), b);
/// ```
pub trait Log10 {
    /// Output type.
    type Output;
    /// Returns the base 10 logarithm of `self`.
    fn log10(self) -> Self::Output;
}

/// Sine function.
///
/// Returns `sin(self `(in radians)`)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// assert!(a.sin().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Sin {
    /// Output type.
    type Output;
    /// Returns the sine of `self`.
    fn sin(self) -> Self::Output;
}

/// Cosine function.
///
/// Returns `cos(self `(in radians)`)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// assert!(a.cos().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Cos {
    /// Output type.
    type Output;
    /// Returns the cosine of `self`.
    fn cos(self) -> Self::Output;
}

/// Tangent function.
///
/// Returns `tan(self `(in radians)`)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// assert!(a.tan().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Tan {
    /// Output type.
    type Output;
    /// Returns the tangent of `self`.
    fn tan(self) -> Self::Output;
}

/// Hyperbolic sine function.
///
/// Returns `sinh(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// assert!(a.sinh().sub(&b).abs() < 1e-10);
/// ```
pub trait Sinh {
    /// Output type.
    type Output;
    /// Hyperbolic sine function
    fn sinh(self) -> Self::Output;
}

/// Hyperbolic cosine function.
///
/// Returns `cosh(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// assert!(a.cosh().sub(&b).abs() < 1e-10);
/// ```
pub trait Cosh {
    /// Output type.
    type Output;
    /// Hyperbolic cosine function.
    fn cosh(self) -> Self::Output;
}

/// Hyperbolic tangent function.
///
/// Returns `tanh(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// assert!(a.tanh().sub(&b).abs() < 1e-10);
/// ```
pub trait Tanh {
    /// Output type.
    type Output;
    /// Hyperbolic tangent function.
    fn tanh(self) -> Self::Output;
}

/// Arcsine function.
///
/// Returns `asin(self)` (in radians) in [-pi/2, pi/2],
/// if `self` is in [-1, 1].
///
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// assert!(a.asin().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Asin {
    /// Output type.
    type Output;
    /// Returns the arcsine of `self`.
    fn asin(self) -> Self::Output;
}

/// Arccosine function.
///
/// Returns `acos(self)` (in radians) in [0, pi],
/// if `self` in [-1, 1].
///
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// assert!(a.acos().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Acos {
    /// Output type.
    type Output;
    /// Returns the arccosine of `self`.
    fn acos(self) -> Self::Output;
}

/// Arctangent function.
///
/// Returns `atan(self)` (in radians) in [-pi/2, pi/2].
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// assert!(a.atan().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Atan {
    /// Output type.
    type Output;
    /// Returns the arctangent of `self`.
    fn atan(self) -> Self::Output;
}

/// Inverse hyperbolic sine function.
///
/// Returns `asinh(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(a.asinh().sub(&b).abs() < 1e-10);
/// ```
pub trait Asinh {
    /// Output type.
    type Output;
    /// Inverse hyperbolic sine function.
    fn asinh(self) -> Self::Output;
}

/// Inverse hyperbolic cosine function.
///
/// Returns `acosh(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(a.acosh().sub(&b).abs() < 1e-10);
/// ```
pub trait Acosh {
    /// Output type.
    type Output;
    /// Inverse hyperbolic cosine function.
    fn acosh(self) -> Self::Output;
}

/// Inverse hyperbolic tangent function.
///
/// Returns `atanh(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(a.atanh().sub(&b).abs() < 1e-10);
/// ```
pub trait Atanh {
    /// Output type.
    type Output;
    /// Inverse hyperbolic tangent function.
    fn atanh(self) -> Self::Output;
}

/// Square root function.
///
/// Returns `sqrt(self)`.
///
/// Note: `sqrt(self)` is NaN if `self` is negative.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 4.0, 9.0, 16.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(a.sqrt(), b);
/// ```
pub trait Sqrt {
    /// Output type.
    type Output;
    /// Returns the square root of `self`.
    fn sqrt(self) -> Self::Output;
}

/// Cubic root function.
///
/// Returns `cbrt(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 8.0, 64.0, 512.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 4.0, 8.0]).unwrap();
/// assert_eq!(a.cbrt(), b);
/// ```
pub trait Cbrt {
    /// Output type.
    type Output;
    /// Returns the cubic root of `self`.
    fn cbrt(self) -> Self::Output;
}

/// Absolute value function.
///
/// Returns `abs(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -6.0, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 6.0, 3.0]).unwrap();
/// assert_eq!(a.abs(), b);
/// ```
pub trait Abs {
    /// Output type.
    type Output;
    /// Returns the absolute value of `self`.
    fn abs(self) -> Self::Output;
}

/// Sign function.
///
/// Returns:
/// * `1` if `self` is positive,
/// * `-1` id `self` is negative.
///
/// Note that for signed integers, `0` -> `0`. However,
/// for floats, `+0.0` -> `1.0` and `-0.0` -> `-1.0`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, -2.0, -6.0, 0.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -1.0, -1.0, 1.0]).unwrap();
/// assert_eq!(a.signum(), b);
/// ```
pub trait Signum {
    /// Output type.
    type Output;
    /// Returns the sign of `self`.
    fn signum(self) -> Self::Output;
}

/// Ceiling function.
///
/// Returns `ceil(self)`.
///
/// Note: `ceil(x)` is the smallest integer greater than or equal to `x`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, -2.0, -6.0, 3.0]).unwrap();
/// assert_eq!(a.ceil(), b);
/// ```
pub trait Ceil {
    /// Output type.
    type Output;
    /// Returns the ceiling of `self`.
    fn ceil(self) -> Self::Output;
}

/// Floor function.
///
/// Returns `floor(self)`.
///
/// Note: `floor(x)` is the greatest integer smaller than or equal to `x`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -3.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a.floor(), b);
/// ```
pub trait Floor {
    /// Output type.
    type Output;
    /// Returns the floor of `self`.
    fn floor(self) -> Self::Output;
}

/// Rounding function.
///
/// Returns `round(self)`.
///
/// Note: `round(x)` is the nearest integer to `x`. Half-way cases
/// are rounded away from zero.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a.round(), b);
/// ```
pub trait Round {
    /// Output type.
    type Output;
    /// Returns the rounding of `self`.
    fn round(self) -> Self::Output;
}

/// Trucation function.
///
/// Returns the integer part of `self`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -6.0, 3.0]).unwrap();
/// assert_eq!(a.trunc(), b);
/// ```
pub trait Trunc {
    /// Output type.
    type Output;
    /// Returns the integer part of `self`.
    fn trunc(self) -> Self::Output;
}

/// Fractional part function.
///
/// Returns the fractional part of `self`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.01, -0.37, -0.5, 0.0]).unwrap();
/// assert!(a.fract().sub(&b).abs() < f64::EPSILON);
/// ```
pub trait Fract {
    /// Output type.
    type Output;
    /// Returns the fractional part of `self`.
    fn fract(self) -> Self::Output;
}

/// Inverse function.
///
/// Returns `1 / self`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, 0.5, 4.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -0.5, 2.0, 0.25]).unwrap();
/// assert!(a.recip().sub(&b).abs() < 1e-10);
/// ```
pub trait Recip {
    /// Output type.
    type Output;
    /// Returns the inverse of `self`.
    fn recip(self) -> Self::Output;
}

/// Conversion to degrees function.
///
/// Returns `self` (in degrees).
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// assert!(a.to_degrees().sub(&b).abs() < 1e-10);
/// ```
pub trait ToDegrees {
    /// Output type.
    type Output;
    /// Converts radians to degrees.
    fn to_degrees(self) -> Self::Output;
}

/// Conversion to radians function.
///
/// Returns `self` (in radians).
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// assert!(a.to_radians().sub(&b).abs() < 1e-10);
/// ```
pub trait ToRadians {
    /// Output type.
    type Output;
    /// Converts degrees to radians.
    fn to_radians(self) -> Self::Output;
}

/// Real portion of a complex number.
///
/// Returns `Re(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![
///      1.0, 0.0,
///     -1.0, 0.0
/// ]).unwrap();
/// assert_eq!(a.re(), b);
/// ```
pub trait Re {
    /// Output type.
    type Output;
    /// Returns the real portion.
    fn re(self) -> Self::Output;
}

/// Imaginary portion of a complex number.
///
/// Returns `Im(self)`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![
///     0.0,  1.0,
///     0.0, -1.0
/// ]).unwrap();
/// assert_eq!(a.im(), b);
/// ```
pub trait Im {
    /// Output type.
    type Output;
    /// Returns the real portion.
    fn im(self) -> Self::Output;
}

/// L2-norm, modulus of a complex number.
///
/// Returns `||self||`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![
///     1.0, 1.0,
///     1.0, 1.0
/// ]).unwrap();
/// assert_eq!(a.norm(), b);
/// ```
pub trait Norm {
    /// Output type.
    type Output;
    /// Returns the L2-norm.
    fn norm(self) -> Self::Output;
}

/// Squared L2-norm, squared modulus of a complex number.
///
/// Returns `||self||^2`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(2.0, 0.0), Complex::new(0.0, 2.0),
///     Complex::new(-2.0, 0.0), Complex::new(0.0, -2.0)
/// ]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![
///     4.0, 4.0,
///     4.0, 4.0
/// ]).unwrap();
/// assert_eq!(a.norm_sqr(), b);
/// ```
pub trait NormSqr {
    /// Output type.
    type Output;
    /// Returns the squared L2-norm.
    fn norm_sqr(self) -> Self::Output;
}

/// Argument of a complex number.
///
/// Returns `arg(self)` (in radians).
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
/// use std::f64::consts::{FRAC_PI_2, PI};
///
/// let a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![
///     0.0, FRAC_PI_2,
///     PI, -FRAC_PI_2
/// ]).unwrap();
/// assert_eq!(a.arg(), b);
/// ```
pub trait Arg {
    /// Output type.
    type Output;
    /// Returns the principal argument.
    fn arg(self) -> Self::Output;
}

/// Conjugate of a complex number.
///
/// Returns `a - jb` where `self = a + jb`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// let b: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, -1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, 1.0)
/// ]).unwrap();
/// assert_eq!(a.conj(), b);
/// ```
pub trait Conj {
    /// Output type.
    type Output;
    /// Returns the complex conjugate.
    fn conj(self) -> Self::Output;
}

/// Linear transformation operator.
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
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, 2.0, 2.0, 2.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let d: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a.mul_add(&b, &c), d);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a.mul_add(2.0, 1.0), b);
/// ```
pub trait MulAdd<Rhs0 = Self, Rhs1 = Self> {
    /// Output type.
    type Output;
    /// Returns `(self * rhs0) + rhs1`.
    fn mul_add(self, rhs0: Rhs0, rhs1: Rhs1) -> Self::Output;
}

/// Cartesian complex operator.
///
/// Returns the complex `self + j * rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, -1.0, 0.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 1.0, 0.0, -1.0]).unwrap();
/// let c: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// assert_eq!(a.add_j(&b), c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, -1.0, 0.0]).unwrap();
/// let b: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, 0.0)
/// ]).unwrap();
/// assert_eq!(a.add_j(0.0), b);
/// ```
pub trait AddJ<Rhs = Self> {
    /// Output type.
    type Output;
    /// Outputs the complex `self + j * rhs`.
    fn add_j(self, rhs: Rhs) -> Self::Output;
}

/// Multiply by j operator.
///
/// Returns the complex `j * self`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, -1.0, 5.0]).unwrap();
/// let b: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(0.0, 1.0), Complex::new(0.0, 2.0),
///     Complex::new(0.0, -1.0), Complex::new(0.0, 5.0)
/// ]).unwrap();
/// assert_eq!(a.j(), b);
/// ```
pub trait J {
    /// Output type.
    type Output;
    /// Outputs the complex `j * self`.
    fn j(self) -> Self::Output;
}

/// Polar complex operator.
///
/// Returns the complex `self * e^(j * rhs)`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
/// use std::f64::consts::{FRAC_PI_2, PI};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_2, PI, -FRAC_PI_2]).unwrap();
/// let c: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// assert!(a.mul_e_pow_j(&b).sub(&c).re() < f64::EPSILON);
/// assert!(a.mul_e_pow_j(&b).sub(&c).im() < f64::EPSILON);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, -1.0, 0.0]).unwrap();
/// let b: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, 0.0)
/// ]).unwrap();
/// assert!(a.mul_e_pow_j(0.0).sub(&b).re() < f64::EPSILON);
/// assert!(a.mul_e_pow_j(0.0).sub(&b).im() < f64::EPSILON);
/// ```
pub trait MulEPowJ<Rhs = Self> {
    /// Output type.
    type Output;
    /// Outputs the complex `self * e^(j * rhs)`.
    fn mul_e_pow_j(self, rhs: Rhs) -> Self::Output;
}

/// Complex phase operator.
///
/// Returns the complex `e^(j * self)`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
/// use std::f64::consts::{FRAC_PI_2, PI};
///
/// let a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_2, PI, -FRAC_PI_2]).unwrap();
/// let b: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// assert!(a.e_pow_j().sub(&b).re() < f64::EPSILON);
/// assert!(a.e_pow_j().sub(&b).im() < f64::EPSILON);
/// ```
pub trait EPowJ {
    /// Output type.
    type Output;
    /// Outputs the complex `e^(j * self)`.
    fn e_pow_j(self) -> Self::Output;
}

macro_rules! binary_op_impl {
    (
        $trait:ident$(<$param_type:ty$(, Output = $output_type:ty)?>)? for $t:ty;
        $trait_fn:ident;
        $inner_fn:path
        $(;$ref:tt self $(;$ref_rhs:tt rhs)?)?
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { $t };
        }
        impl $trait$(<$param_type>)? for $t {
            type Output = isset_or_default!($($($output_type)?)?);
            #[inline]
            fn $trait_fn(self, rhs: isset_or_default!($($param_type)?)) -> Self::Output {
                $inner_fn($($ref)? self, $($($ref_rhs)?)? rhs)
            }
        }
    };
}

macro_rules! ternary_op_impl {
    (
        $trait:ident$(<$param_type0:ty, $param_type1:ty$(, Output = $output_type:ty)?>)? for $t:ty;
        $trait_fn:ident;
        $inner_fn:path
        $(;$ref:tt self $(;$ref_rhs:tt rhs)?)?
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { $t };
        }
        impl $trait$(<$param_type0, $param_type1>)? for $t {
            type Output = isset_or_default!($($output_type)?);
            #[inline]
            fn $trait_fn(self, rhs0: isset_or_default!($($param_type0)?), rhs1: isset_or_default!($($param_type1)?)) -> Self::Output {
                $inner_fn($($ref)? self, $($($ref_rhs)?)? rhs0, $($($ref_rhs)?)? rhs1)
            }
        }
    };
}

macro_rules! fn_impl {
    (
        $trait:ident$(<Output = $output_type:ty>)? for $t:ty;
        $trait_fn:ident;
        $inner_fn:path
        $(;$ref:tt self)?
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { $t };
        }
        impl $trait for $t {
            type Output = isset_or_default!($($output_type)?);
            #[inline]
            fn $trait_fn(self) -> Self::Output {
                $inner_fn($($ref)? self)
            }
        }
    };
}

macro_rules! ops_impl_integer {
    ($($t:ty)*) => {$(
        binary_op_impl! { DivEuclid for $t; div_euclid; Self::div_euclid }
        binary_op_impl! { RemEuclid for $t; rem_euclid; Self::rem_euclid }
        binary_op_impl! { Max for $t; max; std::cmp::Ord::max }
        binary_op_impl! { Min for $t; min; std::cmp::Ord::min }
        binary_op_impl! { Pow<u32> for $t; pow; Self::pow }

        impl MaxMask for $t {
            type Output = $t;
            #[inline]
            fn max_mask(self, rhs: $t) -> Self::Output {
                if self > rhs { 1 } else { 0 }
            }
        }

        impl MinMask for $t {
            type Output = $t;
            #[inline]
            fn min_mask(self, rhs: $t) -> Self::Output {
                if self < rhs { 1 } else { 0 }
            }
        }
    )*};
}

ops_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

macro_rules! ops_impl_signed_integer {
    ($($t:ty)*) => {$(
        fn_impl! { Abs for $t; abs; Self::abs }
        fn_impl! { Signum for $t; signum; Self::signum }
    )*};
}

ops_impl_signed_integer! { i128 i64 i32 i16 i8 }

macro_rules! ops_impl_float {
    ($($t:ty)*) => {$(
        binary_op_impl! { Atan2 for $t; atan2; Self::atan2 }
        binary_op_impl! { Hypot for $t; hypot; Self::hypot }
        binary_op_impl! { Copysign for $t; copysign; Self::copysign }
        binary_op_impl! { DivEuclid for $t; div_euclid; Self::div_euclid }
        binary_op_impl! { RemEuclid for $t; rem_euclid; Self::rem_euclid }
        binary_op_impl! { Max for $t; max; Self::max }
        binary_op_impl! { Min for $t; min; Self::min }
        binary_op_impl! { Pow for $t; pow; Self::powf }
        binary_op_impl! { Pow<i32> for $t; pow; Self::powi }
        binary_op_impl! { Log for $t; log; Self::log }
        binary_op_impl! { AddJ<$t, Output = Complex<$t>> for $t; add_j; Complex::new }
        binary_op_impl! { MulEPowJ<$t, Output = Complex<$t>> for $t; mul_e_pow_j; Complex::from_polar; &self; &rhs }
        fn_impl! { Exp for $t; exp; Self::exp }
        fn_impl! { Exp2 for $t; exp2; Self::exp2 }
        fn_impl! { ExpM1 for $t; exp_m1; Self::exp_m1 }
        fn_impl! { Ln for $t; ln; Self::ln }
        fn_impl! { Ln1p for $t; ln_1p; Self::ln_1p }
        fn_impl! { Log2 for $t; log2; Self::log2 }
        fn_impl! { Log10 for $t; log10; Self::log10 }
        fn_impl! { Sin for $t; sin; Self::sin }
        fn_impl! { Cos for $t; cos; Self::cos }
        fn_impl! { Tan for $t; tan; Self::tan }
        fn_impl! { Asin for $t; asin; Self::asin }
        fn_impl! { Acos for $t; acos; Self::acos }
        fn_impl! { Atan for $t; atan; Self::atan }
        fn_impl! { Sinh for $t; sinh; Self::sinh }
        fn_impl! { Cosh for $t; cosh; Self::cosh }
        fn_impl! { Tanh for $t; tanh; Self::tanh }
        fn_impl! { Asinh for $t; asinh; Self::asinh }
        fn_impl! { Acosh for $t; acosh; Self::acosh }
        fn_impl! { Atanh for $t; atanh; Self::atanh }
        fn_impl! { Sqrt for $t; sqrt; Self::sqrt }
        fn_impl! { Cbrt for $t; cbrt; Self::cbrt }
        fn_impl! { Abs for $t; abs; Self::abs }
        fn_impl! { Signum for $t; signum; Self::signum }
        fn_impl! { Ceil for $t; ceil; Self::ceil }
        fn_impl! { Floor for $t; floor; Self::floor }
        fn_impl! { Round for $t; round; Self::round }
        fn_impl! { Trunc for $t; trunc; Self::trunc }
        fn_impl! { Fract for $t; fract; Self::fract }
        fn_impl! { Recip for $t; recip; Self::recip }
        fn_impl! { ToDegrees for $t; to_degrees; Self::to_degrees }
        fn_impl! { ToRadians for $t; to_radians; Self::to_radians }
        ternary_op_impl! { MulAdd for $t; mul_add; Self::mul_add }

        impl MaxMask for $t {
            type Output = $t;
            #[inline]
            fn max_mask(self, rhs: $t) -> Self::Output {
                if self > rhs { 1.0 } else { 0.0 }
            }
        }

        impl MinMask for $t {
            type Output = $t;
            #[inline]
            fn min_mask(self, rhs: $t) -> Self::Output {
                if self < rhs { 1.0 } else { 0.0 }
            }
        }

        impl Conj for $t {
            type Output = $t;
            #[inline]
            fn conj(self) -> Self::Output {
                self
            }
        }

        impl J for $t {
            type Output = Complex<$t>;
            #[inline]
            fn j(self) -> Self::Output {
                Complex::new(0.0, self)
            }
        }

        impl EPowJ for $t {
            type Output = Complex<$t>;
            #[inline]
            fn e_pow_j(self) -> Self::Output {
                Complex::from_polar(&1.0, &self)
            }
        }
    )*};
}

ops_impl_float! { f64 f32 }

macro_rules! ops_impl_complex_float {
    ($($t:ty)*) => {$(
        binary_op_impl! { Pow for Complex<$t>; pow; Self::powc; &self }
        binary_op_impl! { Pow<i32> for Complex<$t>; pow; Self::powi; &self }
        binary_op_impl! { Pow<u32> for Complex<$t>; pow; Self::powu; &self }
        binary_op_impl! { Pow<$t> for Complex<$t>; pow; Self::powf; &self }
        fn_impl! { Exp for Complex<$t>; exp; Self::exp; &self }
        fn_impl! { Ln for Complex<$t>; ln; Self::ln; &self }
        fn_impl! { Sin for Complex<$t>; sin; Self::sin; &self }
        fn_impl! { Cos for Complex<$t>; cos; Self::cos; &self }
        fn_impl! { Tan for Complex<$t>; tan; Self::tan; &self }
        fn_impl! { Asin for Complex<$t>; asin; Self::asin; &self }
        fn_impl! { Acos for Complex<$t>; acos; Self::acos; &self }
        fn_impl! { Atan for Complex<$t>; atan; Self::atan; &self }
        fn_impl! { Sinh for Complex<$t>; sinh; Self::sinh; &self }
        fn_impl! { Cosh for Complex<$t>; cosh; Self::cosh; &self }
        fn_impl! { Tanh for Complex<$t>; tanh; Self::tanh; &self }
        fn_impl! { Asinh for Complex<$t>; asinh; Self::asinh; &self }
        fn_impl! { Acosh for Complex<$t>; acosh; Self::acosh; &self }
        fn_impl! { Atanh for Complex<$t>; atanh; Self::atanh; &self }
        fn_impl! { Sqrt for Complex<$t>; sqrt; Self::sqrt; &self }
        fn_impl! { Cbrt for Complex<$t>; cbrt; Self::cbrt; &self }
        fn_impl! { Recip for Complex<$t>; recip; Self::inv; &self }
        fn_impl! { Conj for Complex<$t>; conj; Self::conj; &self }
        fn_impl! { Norm<Output=$t> for Complex<$t>; norm; Self::norm; &self }
        fn_impl! { NormSqr<Output=$t> for Complex<$t>; norm_sqr; Self::norm_sqr; &self }
        fn_impl! { Arg<Output=$t> for Complex<$t>; arg; Self::arg; &self }

        impl Re for Complex<$t> {
            type Output = $t;
            #[inline]
            fn re(self) -> Self::Output {
                self.re
            }
        }

        impl Im for Complex<$t> {
            type Output = $t;
            #[inline]
            fn im(self) -> Self::Output {
                self.im
            }
        }
    )*};
}

ops_impl_complex_float! { f64 f32 }

macro_rules! op_impl_ref {
    ($($trait:ident, $trait_fn:ident);*) => {$(
        impl<T, U> $trait<&U> for &T
        where
            T: $trait<U> + Copy,
            U: Copy,
        {
            type Output = T::Output;
            fn $trait_fn(self, rhs: &U) -> Self::Output {
                (*self).$trait_fn(*rhs)
            }
        }
    )*};
}

op_impl_ref! {
    Atan2, atan2;
    Hypot, hypot;
    Copysign, copysign;
    DivEuclid, div_euclid;
    Max, max;
    Min, min;
    MaxMask, max_mask;
    MinMask, min_mask;
    RemEuclid, rem_euclid;
    Pow, pow;
    Log, log;
    AddJ, add_j;
    MulEPowJ, mul_e_pow_j
}

macro_rules! fn_impl_ref {
    ($($trait:ident, $trait_fn:ident);*) => {$(
        impl<T> $trait for &T
        where
            T: $trait + Copy,
        {
            type Output = T::Output;
            fn $trait_fn(self) -> Self::Output {
                (*self).$trait_fn()
            }
        }
    )*};
}

fn_impl_ref! {
    Exp, exp;
    Exp2, exp2;
    ExpM1, exp_m1;
    Ln, ln;
    Ln1p, ln_1p;
    Log2, log2;
    Log10, log10;
    Sin, sin;
    Cos, cos;
    Tan, tan;
    Sinh, sinh;
    Cosh, cosh;
    Tanh, tanh;
    Asin, asin;
    Acos, acos;
    Atan, atan;
    Asinh, asinh;
    Acosh, acosh;
    Atanh, atanh;
    Sqrt, sqrt;
    Cbrt, cbrt;
    Abs, abs;
    Signum, signum;
    Ceil, ceil;
    Floor, floor;
    Round, round;
    Trunc, trunc;
    Fract, fract;
    Recip, recip;
    ToDegrees, to_degrees;
    ToRadians, to_radians;
    Conj, conj;
    J, j;
    EPowJ, e_pow_j
}

/// In-place linear transformation operator.
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
/// Note that `Rhs0` and `Rhs1` are `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, 2.0, 2.0, 2.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// a.mul_add_assign(&b, &c);
/// let d: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a, d);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.mul_add_assign(2.0, 1.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, 1.0, 1.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait MulAddAssign<Rhs0 = Self, Rhs1 = Self> {
    /// Fused in-place multiply-add.
    fn mul_add_assign(&mut self, rhs0: Rhs0, rhs1: Rhs1);
}

/// In-place four quadrant arctangent operator.
///
/// Computes the four quadrant arctengent of `self` (`y`)
/// and `rhs` (`x`) in radians as follows:
/// * `x = 0`, `y = 0`: `0`
/// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
/// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
/// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]).unwrap();
/// a.atan2_assign(&b);
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_4, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// a.sub_assign(&c);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::FRAC_PI_2;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, -2.0, 1.0]).unwrap();
/// a.atan2_assign(0.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_2, FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait Atan2Assign<Rhs = Self> {
    /// Performs the in-place four quadrant arctangent operation.
    fn atan2_assign(&mut self, rhs: Rhs);
}

/// In-place hypotenuse operator.
///
/// Calculates the length of the hypotenuse of a right-angle triangle
/// given legs of length `self` and `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::SQRT_2;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![SQRT_2, 1.0, 1.0, SQRT_2]).unwrap();
/// a.hypot_assign(&b);
/// a.sub_assign(&c);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::SQRT_2;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![SQRT_2, 1.0, 1.0, SQRT_2]).unwrap();
/// a.hypot_assign(1.0);
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait HypotAssign<Rhs = Self> {
    /// Calculates the length of the hypotenuse of a right-angle triangle
    /// given legs of length `self` and `rhs`.
    fn hypot_assign(&mut self, rhs: Rhs);
}

/// In-place copysign operator.
///
/// Result is composed of the magnitude of `self` and the sign of `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, -4.0, -4.0, 2.0]).unwrap();
/// a.copysign_assign(&b);
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -3.0, 4.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -1.0, 5.0]).unwrap();
/// a.copysign_assign(-5.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, -2.0, -1.0, -5.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait CopysignAssign<Rhs = Self> {
    /// Performs the in-place copysign operation.
    fn copysign_assign(&mut self, rhs: Rhs);
}

/// In-place quotient of Euclidean division operator.
///
/// Computes the quotient of Euclidean division of `self` by `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![7, 7, -7, -7]).unwrap();
/// let b: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![4, -4, 4, -4]).unwrap();
/// a.div_euclid_assign(&b);
/// let c: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, -1, -2, 2]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor1D<i32, 2> = Tensor::try_from(vec![7, -7]).unwrap();
/// a.div_euclid_assign(4);
/// let b: VecTensor1D<i32, 2> = Tensor::try_from(vec![1, -2]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait DivEuclidAssign<Rhs = Self> {
    /// Performs the in-place quotient of Euclidean division operation.
    fn div_euclid_assign(&mut self, rhs: Rhs);
}

/// In-place remainder of Euclidean division operator.
///
/// Computes the least nonnegative remainder of `self (mod rhs)`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![7, -7, 7, -7]).unwrap();
/// let b: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![4, 4, -4, -4]).unwrap();
/// a.rem_euclid_assign(&b);
/// let c: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![3, 1, 3, 1]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor1D<i32, 2> = Tensor::try_from(vec![7, -7]).unwrap();
/// a.rem_euclid_assign(4);
/// let b: VecTensor1D<i32, 2> = Tensor::try_from(vec![3, 1]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait RemEuclidAssign<Rhs = Self> {
    /// Performs the in-place remainder of Euclidean division operation.
    fn rem_euclid_assign(&mut self, rhs: Rhs);
}

/// In-place max operator.
///
/// Computes the max of `self` and `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// a.max_assign(&b);
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 6.0, 12.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// a.max_assign(5.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, 10.0, 5.0, 6.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait MaxAssign<Rhs = Self> {
    /// Performs the in-place max operation.
    fn max_assign(&mut self, rhs: Rhs);
}

/// In-place min operator.
///
/// Computes the min of `self` and `rhs`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// a.min_assign(&b);
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 1.0, 4.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// a.min_assign(5.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 5.0, 3.0, 5.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait MinAssign<Rhs = Self> {
    /// Performs the in-place max operation.
    fn min_assign(&mut self, rhs: Rhs);
}

/// In-place mask of max operator.
///
/// Computes the mask of max on `self`:
/// * `1` if `self` is the max,
/// * `0` otherwise.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// a.max_mask_assign(&b);
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 1.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// a.max_mask_assign(5.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 1.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait MaxMaskAssign<Rhs = Self> {
    /// Performs the in-place argmax operation.
    fn max_mask_assign(&mut self, rhs: Rhs);
}

/// In-place mask of min operator.
///
/// Computes the mask of min on `self`:
/// * `1` if `self` is the min,
/// * `0` otherwise.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![7.0, -3.0, 1.0, 12.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![5.0, -8.0, 6.0, 4.0]).unwrap();
/// a.min_mask_assign(&b);
/// let c: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 0.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a, c);
/// ```
///
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 10.0, 3.0, 6.0]).unwrap();
/// a.min_mask_assign(5.0);
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait MinMaskAssign<Rhs = Self> {
    /// Performs the in-place max operation.
    fn min_mask_assign(&mut self, rhs: Rhs);
}

/// In-place power operator.
///
/// Raise `self` to `rhs` power.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// a.pow_assign(2);
/// let c: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, 4, 9, 16]).unwrap();
/// assert_eq!(a, c);
/// ```
pub trait PowAssign<Rhs = Self> {
    /// Performs the in-place max operation.
    fn pow_assign(&mut self, rhs: Rhs);
}

/// In-place arbitrary base logarithm function.
///
/// Computes the logarithm of `self` with respect to an arbitrary base.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, 9.0, 9.0, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 2.0, 1.0]).unwrap();
/// a.log_assign(3.0);
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait LogAssign<Rhs = Self> {
    /// Returns the base `rhs` logarithm of `self`.
    fn log_assign(&mut self, rhs: Rhs);
}

/// In-place exponential function.
///
/// `self -> e^(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.exp_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait ExpAssign {
    /// Applies in-place exp function.
    fn exp_assign(&mut self);
}

/// In-place power of 2 function.
///
/// `self -> 2^(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// a.exp2_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Exp2Assign {
    /// Applies in-place `2^x` function.
    fn exp2_assign(&mut self);
}

/// In-place exponential function minus 1.
///
/// `self -> exp(x) - 1`
///
/// This is more accurate than if the operations were
/// performed separately even if `self` is close to zero.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::LN_2;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![LN_2, 0.0, 0.0, LN_2]).unwrap();
/// a.exp_m1_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait ExpM1Assign {
    /// Applies in place `exp(x) - 1` function.
    fn exp_m1_assign(&mut self);
}

/// In-place natural logarithm function.
///
/// `self -> ln(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![E, 1.0, 1.0, E]).unwrap();
/// a.ln_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait LnAssign {
    /// Applies in-place natural logarithm function.
    fn ln_assign(&mut self);
}

/// In-place natural logarithm of 1 plus x function.
///
/// `self -> ln(1 + self)`
///
/// This is more accurate than if the operations were performed separately.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, E - 1.0, E - 1.0, 0.0]).unwrap();
/// a.ln_1p_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 1.0, 1.0, 0.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Ln1pAssign {
    /// Applies in-place `ln(1 + x)` function.
    fn ln_1p_assign(&mut self);
}

/// In-place base 2 logarithm function.
///
/// `self -> log2(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![256.0, 1_024.0, 1_024.0, 256.0]).unwrap();
/// a.log2_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![8.0, 10.0, 10.0, 8.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Log2Assign {
    /// Applies in-place base 2 logarithm function.
    fn log2_assign(&mut self);
}

/// In-place base 10 logarithm function.
///
/// `self -> log10(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::f64::consts::E;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![100.0, 1_000.0, 1_000.0, 100.0]).unwrap();
/// a.log10_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, 3.0, 3.0, 2.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Log10Assign {
    /// Applies in-place base 10 logarithm function.
    fn log10_assign(&mut self);
}

/// In-place sine function.
///
/// `self` (in radians) `-> sin(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// a.sin_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait SinAssign {
    /// Applies in-place sine function.
    fn sin_assign(&mut self);
}

/// In-place cosine function.
///
/// `self` (in radians) `-> cos(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.cos_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait CosAssign {
    /// Applies in-place cosine function.
    fn cos_assign(&mut self);
}

/// In-place tangent function.
///
/// `self` (in radians) `-> tan(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// a.tan_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait TanAssign {
    /// Applies in-place tangent function.
    fn tan_assign(&mut self);
}

/// In-place hyperbolic sine function.
///
/// `self -> sinh(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sinh_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait SinhAssign {
    /// Applies in-place hyperbolic sine function.
    fn sinh_assign(&mut self);
}

/// In-place hyperbolic cosine function.
///
/// `self -> cosh(self)`
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.cosh_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait CoshAssign {
    /// Applies in-place hyperbolic cosine function.
    fn cosh_assign(&mut self);
}

/// In-place hyperbolic tangent function.
///
/// `self -> tanh(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.tanh_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait TanhAssign {
    /// Applies in-place hyperbolic tangent function.
    fn tanh_assign(&mut self);
}

/// In-place arcsine function.
///
/// `self` in [-1, 1] `-> asin(self)` (in radians) in [-pi/2, pi/2]
///
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, 0.5, FRAC_1_SQRT_2, 1.0]).unwrap();
/// a.asin_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait AsinAssign {
    /// Applies in-place arcsine function.
    fn asin_assign(&mut self);
}

/// In-place arccosine function.
///
/// `self` in [-1, 1] `-> acos(self)` (in radians) in [0, pi]
///
/// Note: `asin(self)` is NaN if `self` is outside [-1, 1].
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, FRAC_1_SQRT_2};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, FRAC_1_SQRT_2, 0.5, 0.0]).unwrap();
/// a.acos_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.0, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait AcosAssign {
    /// Applies in-place arccosine function.
    fn acos_assign(&mut self);
}

/// In-place arctangent function.
///
/// `self -> atan(self)` (in radians) in [-pi/2, pi/2]
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_4, FRAC_PI_3};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-1.0, 0.0, 1.0, 0.0]).unwrap();
/// a.atan_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![-FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait AtanAssign {
    /// Applies in-place arctangent function.
    fn atan_assign(&mut self);
}

/// In-place inverse hyperbolic sine function.
///
/// `self -> asinh(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::E;
///
/// let g = ((E * E) - 1.0) / (2.0 * E);
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.asinh_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait AsinhAssign {
    /// Applies in-place inverse hyperbolic sine function.
    fn asinh_assign(&mut self);
}

/// In-place inverse hyperbolic cosine function.
///
/// `self -> acosh(self)`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::E;
///
/// let g = ((E * E) + 1.0) / (2.0 * E);
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 1.0, 1.0, g]).unwrap();
/// a.acosh_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait AcoshAssign {
    /// Applies in-place inverse hyperbolic cosine function.
    fn acosh_assign(&mut self);
}

/// In-place inverse hyperbolic tangent function.
///
/// `self -> atanh(self)`
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::E;
///
/// let g = (1.0 - E.powi(-2)) / (1.0 + E.powi(-2));
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![g, 0.0, 0.0, g]).unwrap();
/// a.atanh_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait AtanhAssign {
    /// Applies in-place inverse hyperbolic tangent function.
    fn atanh_assign(&mut self);
}

/// In-place square root function.
///
/// `self -> sqrt(self)`
///
/// Note: `sqrt(self)` is NaN if `self` is negative.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 4.0, 9.0, 16.0]).unwrap();
/// a.sqrt_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait SqrtAssign {
    /// Applies in-place square root function.
    fn sqrt_assign(&mut self);
}

/// In-place cubic root function.
///
/// `self -> cbrt(self)`
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 8.0, 64.0, 512.0]).unwrap();
/// a.cbrt_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 4.0, 8.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait CbrtAssign {
    /// Applies in-place cubic root function.
    fn cbrt_assign(&mut self);
}

/// In-place absolute value function.
///
/// `self -> abs(self)`
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -6.0, 3.0]).unwrap();
/// a.abs_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, 2.0, 6.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait AbsAssign {
    /// Applies in-place absolute value function.
    fn abs_assign(&mut self);
}

/// In-place sign function.
///
/// `self ->`
/// * `1` if `self` is positive
/// * `-1` id `self` is negative
///
/// Note that for signed integers, `0` -> `0`. However,
/// for floats, `+0.0` -> `1.0` and `-0.0` -> `-1.0`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![3.0, -2.0, -6.0, 0.0]).unwrap();
/// a.signum_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -1.0, -1.0, 1.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait SignumAssign {
    /// Applies in-place sign function.
    fn signum_assign(&mut self);
}

/// In-place ceiling function.
///
/// `self -> ceil(self)`
///
/// Note: `ceil(x)` is the smallest integer greater than or equal to `x`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// a.ceil_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![2.0, -2.0, -6.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait CeilAssign {
    /// Applies in-place ceiling function.
    fn ceil_assign(&mut self);
}

/// In-place floor function.
///
/// `self -> floor(self)`
///
/// Note: `floor(x)` is the greatest integer smaller than or equal to `x`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.81, 3.0]).unwrap();
/// a.floor_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -3.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait FloorAssign {
    /// Applies in-place floor function.
    fn floor_assign(&mut self);
}

/// In-place rounding function.
///
/// `self -> round(self)`
///
/// Note: `round(x)` is the nearest integer to `x`. Half-way cases
/// are rounded away from zero.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// a.round_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -7.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait RoundAssign {
    /// Applies in-place rounding function.
    fn round_assign(&mut self);
}

/// In-place trucation function.
///
/// Retains the integer part of `self`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// a.trunc_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, -6.0, 3.0]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait TruncAssign {
    /// Retains the integer part of `self`.
    fn trunc_assign(&mut self);
}

/// In-place fractional part function.
///
/// Retains the fractional part of `self`.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.01, -2.37, -6.5, 3.0]).unwrap();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![0.01, -0.37, -0.5, 0.0]).unwrap();
/// a.fract_assign();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < f64::EPSILON);
/// ```
pub trait FractAssign {
    /// Retains the fractional part of `self`.
    fn fract_assign(&mut self);
}

/// In-place inverse function.
///
/// `self -> 1 / self`
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -2.0, 0.5, 4.0]).unwrap();
/// a.recip_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![1.0, -0.5, 2.0, 0.25]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait RecipAssign {
    /// Applies in-place inverse function.
    fn recip_assign(&mut self);
}

/// In-place conversion to degrees function.
///
/// `self` (in radians) `-> self` (in degrees)
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.to_degrees_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait ToDegreesAssign {
    /// Converts in-place to degrees.
    fn to_degrees_assign(&mut self);
}

/// In-place conversion to radians function.
///
/// `self` (in degrees) `-> self` (in radians)
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use std::ops::SubAssign;
/// use std::f64::consts::{FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2};
///
/// let mut a: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![30.0, 45.0, 60.0, 90.0]).unwrap();
/// a.to_radians_assign();
/// let b: VecTensor2D<f64, 2, 2> = Tensor::try_from(vec![FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2]).unwrap();
/// a.sub_assign(&b);
/// a.abs_assign();
/// assert!(a < 1e-10);
/// ```
pub trait ToRadiansAssign {
    /// Converts in-place to radians.
    fn to_radians_assign(&mut self);
}

/// Conjugate of a complex number.
///
/// `self = a + ib` -> `a - ib`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
/// use num_complex::{Complex, Complex64};
///
/// let mut a: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)
/// ]).unwrap();
/// let b: VecTensor2D<Complex64, 2, 2> = Tensor::try_from(vec![
///     Complex::new(1.0, 0.0), Complex::new(0.0, -1.0),
///     Complex::new(-1.0, 0.0), Complex::new(0.0, 1.0)
/// ]).unwrap();
/// a.conj_assign();
/// assert_eq!(a, b);
/// ```
pub trait ConjAssign {
    /// Computes the complex conjugate.
    fn conj_assign(&mut self);
}

/// Set to zero.
///
/// `self -> 0`.
///  
/// # Examples
/// ```
/// use melange::prelude::*;
/// use std::ops::*;
/// use std::convert::TryFrom;
///
/// let mut a: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![1, 1, 1, 1]).unwrap();
/// let b: VecTensor2D<i32, 2, 2> = Tensor::try_from(vec![0, 0, 0, 0]).unwrap();
/// a.zero_out();
/// assert_eq!(a, b);
/// ```
pub trait ZeroOut {
    /// Set to zero.
    fn zero_out(&mut self);
}

macro_rules! in_place_binary_op_impl {
    (
        $trait:ident$(<$param_type:ty>)? for $t:ty;
        $trait_fn:ident;
        $inner_fn:path
        $(;$ref:tt self)?
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { $t };
        }

        impl $trait$(<$param_type>)? for $t {
            #[inline]
            fn $trait_fn(&mut self, rhs: isset_or_default!($($param_type)?)) {
                *self = $inner_fn($($ref)? *self, rhs);
            }
        }
    };
}

macro_rules! in_place_ternary_op_impl {
    (
        $trait:ident$(<$param_type0:ty, $param_type1:ty>)? for $t:ty;
        $trait_fn:ident;
        $inner_fn:path
        $(;$ref:tt self)?
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => { $var };
            () => { $t };
        }
        impl $trait$(<$param_type0, $param_type1>)? for $t {
            #[inline]
            fn $trait_fn(&mut self, rhs0: isset_or_default!($($param_type0)?), rhs1: isset_or_default!($($param_type1)?)) {
                *self = $inner_fn($($ref)? *self, rhs0, rhs1);
            }
        }
    };
}

macro_rules! in_place_fn_impl {
    (
        $trait:ident for $t:ty;
        $trait_fn:ident;
        $inner_fn:path
        $(;$ref:tt self)?
    ) => {
        impl $trait for $t {
            #[inline]
            fn $trait_fn(&mut self) {
                *self = $inner_fn($($ref)? *self);
            }
        }
    };
}

macro_rules! in_place_ops_impl_integer {
    ($($t:ty)*) => {$(
        in_place_binary_op_impl! { DivEuclidAssign for $t; div_euclid_assign; Self::div_euclid }
        in_place_binary_op_impl! { RemEuclidAssign for $t; rem_euclid_assign; Self::rem_euclid }
        in_place_binary_op_impl! { MaxAssign for $t; max_assign; std::cmp::Ord::max }
        in_place_binary_op_impl! { MinAssign for $t; min_assign; std::cmp::Ord::min }
        in_place_binary_op_impl! { PowAssign<u32> for $t; pow_assign; Self::pow }

        impl MaxMaskAssign for $t {
            #[inline]
            fn max_mask_assign(&mut self, rhs: $t) {
                *self = if *self > rhs { 1 } else { 0 };
            }
        }

        impl MinMaskAssign for $t {
            #[inline]
            fn min_mask_assign(&mut self, rhs: $t) {
                *self = if *self < rhs { 1 } else { 0 };
            }
        }

        impl ZeroOut for $t {
            #[inline]
            fn zero_out(&mut self) {
                *self = 0;
            }
        }
    )*};
}

in_place_ops_impl_integer! { u128 u64 u32 u16 u8 i128 i64 i32 i16 i8 }

macro_rules! in_place_ops_impl_signed_integer {
    ($($t:ty)*) => {$(
        in_place_fn_impl! { AbsAssign for $t; abs_assign; Self::abs }
        in_place_fn_impl! { SignumAssign for $t; signum_assign; Self::signum }
    )*};
}

in_place_ops_impl_signed_integer! { i128 i64 i32 i16 i8 }

macro_rules! in_place_ops_impl_float {
    ($($t:ty)*) => {$(
        in_place_binary_op_impl! { Atan2Assign for $t; atan2_assign; Self::atan2 }
        in_place_binary_op_impl! { HypotAssign for $t; hypot_assign; Self::hypot }
        in_place_binary_op_impl! { CopysignAssign for $t; copysign_assign; Self::copysign }
        in_place_binary_op_impl! { DivEuclidAssign for $t; div_euclid_assign; Self::div_euclid }
        in_place_binary_op_impl! { RemEuclidAssign for $t; rem_euclid_assign; Self::rem_euclid }
        in_place_binary_op_impl! { MaxAssign for $t; max_assign; Self::max }
        in_place_binary_op_impl! { MinAssign for $t; min_assign; Self::min }
        in_place_binary_op_impl! { PowAssign for $t; pow_assign; Self::powf }
        in_place_binary_op_impl! { PowAssign<i32> for $t; pow_assign; Self::powi }
        in_place_binary_op_impl! { LogAssign for $t; log_assign; Self::log }
        in_place_fn_impl! { ExpAssign for $t; exp_assign; Self::exp }
        in_place_fn_impl! { Exp2Assign for $t; exp2_assign; Self::exp2 }
        in_place_fn_impl! { ExpM1Assign for $t; exp_m1_assign; Self::exp_m1 }
        in_place_fn_impl! { LnAssign for $t; ln_assign; Self::ln }
        in_place_fn_impl! { Ln1pAssign for $t; ln_1p_assign; Self::ln_1p }
        in_place_fn_impl! { Log2Assign for $t; log2_assign; Self::log2 }
        in_place_fn_impl! { Log10Assign for $t; log10_assign; Self::log10 }
        in_place_fn_impl! { SinAssign for $t; sin_assign; Self::sin }
        in_place_fn_impl! { CosAssign for $t; cos_assign; Self::cos }
        in_place_fn_impl! { TanAssign for $t; tan_assign; Self::tan }
        in_place_fn_impl! { AsinAssign for $t; asin_assign; Self::asin }
        in_place_fn_impl! { AcosAssign for $t; acos_assign; Self::acos }
        in_place_fn_impl! { AtanAssign for $t; atan_assign; Self::atan }
        in_place_fn_impl! { SinhAssign for $t; sinh_assign; Self::sinh }
        in_place_fn_impl! { CoshAssign for $t; cosh_assign; Self::cosh }
        in_place_fn_impl! { TanhAssign for $t; tanh_assign; Self::tanh }
        in_place_fn_impl! { AsinhAssign for $t; asinh_assign; Self::asinh }
        in_place_fn_impl! { AcoshAssign for $t; acosh_assign; Self::acosh }
        in_place_fn_impl! { AtanhAssign for $t; atanh_assign; Self::atanh }
        in_place_fn_impl! { SqrtAssign for $t; sqrt_assign; Self::sqrt }
        in_place_fn_impl! { CbrtAssign for $t; cbrt_assign; Self::cbrt }
        in_place_fn_impl! { AbsAssign for $t; abs_assign; Self::abs }
        in_place_fn_impl! { SignumAssign for $t; signum_assign; Self::signum }
        in_place_fn_impl! { CeilAssign for $t; ceil_assign; Self::ceil }
        in_place_fn_impl! { FloorAssign for $t; floor_assign; Self::floor }
        in_place_fn_impl! { RoundAssign for $t; round_assign; Self::round }
        in_place_fn_impl! { TruncAssign for $t; trunc_assign; Self::trunc }
        in_place_fn_impl! { FractAssign for $t; fract_assign; Self::fract }
        in_place_fn_impl! { RecipAssign for $t; recip_assign; Self::recip }
        in_place_fn_impl! { ToDegreesAssign for $t; to_degrees_assign; Self::to_degrees }
        in_place_fn_impl! { ToRadiansAssign for $t; to_radians_assign; Self::to_radians }
        in_place_ternary_op_impl! { MulAddAssign for $t; mul_add_assign; Self::mul_add }

        impl MaxMaskAssign for $t {
            #[inline]
            fn max_mask_assign(&mut self, rhs: $t) {
                *self = if *self > rhs { 1.0 } else { 0.0 };
            }
        }

        impl MinMaskAssign for $t {
            #[inline]
            fn min_mask_assign(&mut self, rhs: $t) {
                *self = if *self < rhs { 1.0 } else { 0.0 };
            }
        }

        impl ConjAssign for $t {
            #[inline]
            fn conj_assign(&mut self) {}
        }

        impl ZeroOut for $t {
            #[inline]
            fn zero_out(&mut self) {
                *self = 0.0;
            }
        }
    )*};
}

in_place_ops_impl_float! { f64 f32 }

macro_rules! in_place_ops_impl_complex_float {
    ($($t:ty)*) => {$(
        in_place_binary_op_impl! { PowAssign for $t; pow_assign; Self::powc; &self }
        in_place_binary_op_impl! { PowAssign<i32> for $t; pow_assign; Self::powi; &self }
        in_place_binary_op_impl! { PowAssign<u32> for $t; pow_assign; Self::powu; &self }
        in_place_fn_impl! { ExpAssign for $t; exp_assign; Self::exp; &self }
        in_place_fn_impl! { LnAssign for $t; ln_assign; Self::ln; &self }
        in_place_fn_impl! { SinAssign for $t; sin_assign; Self::sin; &self }
        in_place_fn_impl! { CosAssign for $t; cos_assign; Self::cos; &self }
        in_place_fn_impl! { TanAssign for $t; tan_assign; Self::tan; &self }
        in_place_fn_impl! { AsinAssign for $t; asin_assign; Self::asin; &self }
        in_place_fn_impl! { AcosAssign for $t; acos_assign; Self::acos; &self }
        in_place_fn_impl! { AtanAssign for $t; atan_assign; Self::atan; &self }
        in_place_fn_impl! { SinhAssign for $t; sinh_assign; Self::sinh; &self }
        in_place_fn_impl! { CoshAssign for $t; cosh_assign; Self::cosh; &self }
        in_place_fn_impl! { TanhAssign for $t; tanh_assign; Self::tanh; &self }
        in_place_fn_impl! { AsinhAssign for $t; asinh_assign; Self::asinh; &self }
        in_place_fn_impl! { AcoshAssign for $t; acosh_assign; Self::acosh; &self }
        in_place_fn_impl! { AtanhAssign for $t; atanh_assign; Self::atanh; &self }
        in_place_fn_impl! { SqrtAssign for $t; sqrt_assign; Self::sqrt; &self }
        in_place_fn_impl! { CbrtAssign for $t; cbrt_assign; Self::cbrt; &self }
        in_place_fn_impl! { RecipAssign for $t; recip_assign; Self::inv; &self }
        in_place_fn_impl! { ConjAssign for $t; conj_assign; Self::conj; &self }

        impl ZeroOut for $t {
            #[inline]
            fn zero_out(&mut self) {
                *self = Complex::new(0.0, 0.0);
            }
        }
    )*};
}

in_place_ops_impl_complex_float! { Complex64 Complex32 }
