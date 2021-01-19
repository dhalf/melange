//! Implements the basic operations defined
//! by the traits of the [`ops`](crate::ops)
//! module for `Variables`.

use super::variable::{BackwardCompatible, InternalVariable, Variable};
use crate::ops::*;
use crate::scalar_traits::*;
use crate::tensor::prelude::*;
use crate::tensor::tensor_traits::BinaryOp;
use num_complex::Complex;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::*;
use std::rc::Rc;

// NOTE: due to macro_rules limitations such as no look ahead
// or very strict and future proofed "follow sets", we cannot
// parse the usual where clause syntax. We thus had to change
// it in two respects:
// - HRTBs parsing requires lookahead, we write the generic
//   lifetimes after the type instead.
//   `for<'a> T: Trait<'a>,` becomes `T | 'a: Trait<'a>,`
// - `+` is not in the follow set of path syntax nodes used
//   to parse traits in the bounds. We use `|` instead of `+`.
//   `T: Debug + Copy,` becomes `T: Debug | Copy,`
macro_rules! two_diff_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $backward_closure:expr
    ) => {
        impl<'var, T, V, G, B, Vrhs, Grhs> $trait_name<Variable<'var, T, Vrhs, Grhs, B::Alloc>> for Variable<'var, T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<&'a Vrhs, Output = <V as BinaryOp<Vrhs>>::Output>,
            V: BinaryOp<Vrhs>,

            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward gradient "copy" in output backward closure.
            B: AllocLike<Scalar = T>,

            // Backward calls on inputs in output backward closure.
            G: BackwardCompatible<B>,
            Grhs: BackwardCompatible<B::Alloc>,

            // Lifetime bounds required by output backward closure.
            T: 'var,
            V: 'var,
            G: 'var,
            B: 'var,
            Vrhs: 'var,
            Grhs: 'var,
            B::Alloc: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<'var, T, <V as BinaryOp<Vrhs>>::Output, V::Alloc, B>;
            fn $fn_name($self, $rhs: Variable<'var, T, Vrhs, Grhs, B::Alloc>) -> Self::Output {
                let value = $self.value.$fn_name(&$rhs.value);

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(None),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

two_diff_op_impl! {
    Add; add; "add_back";
    (self, rhs) => move |grad| {
        rhs.backward(grad.to_contiguous());
        self.backward(grad);
    }
}

two_diff_op_impl! {
    Sub; sub; "sub_back";
    where
        &'a B | 'a: Neg<Output = B::Alloc>;
    (self, rhs) => move |grad| {
        rhs.backward(-&grad);
        self.backward(grad);
    }
}

two_diff_op_impl! {
    Mul; mul; "mul_back";
    where
        Vrhs: AllocLike<Scalar = T>,
        &'a V | 'a: Conj<Output = V::Alloc>,
        &'a Vrhs | 'a: Conj<Output = Vrhs::Alloc>,
        &'a B | 'a: Mul<&'a V::Alloc, Output = B::Alloc>,
        B: Mul<Vrhs::Alloc, Output = B>;
    (self, rhs) => move |grad| {
        rhs.backward(&grad * &self.value.conj());
        self.backward(grad * rhs.value.conj());
    }
}

two_diff_op_impl! {
    Div; div; "div_back";
    where
        Vrhs: AllocLike<Scalar = T>,
        &'a V | 'a: Neg<Output = V::Alloc>,
        V::Alloc | 'a: Div<&'a Vrhs, Output = V::Output>,
        V::Output | 'a: Div<&'a Vrhs, Output = V::Output>,
        V::Output: Conj<Output = V::Alloc>,
        &'a B | 'a: Mul<&'a V::Alloc, Output = B::Alloc>,
        &'a Vrhs | 'a: Recip<Output = Vrhs::Alloc>,
        Vrhs::Alloc: Conj<Output = Vrhs::Alloc>,
        B: Mul<Vrhs::Alloc, Output = B>;
    (self, rhs) => move |grad| {
        rhs.backward(&grad * &((-&self.value / &rhs.value) / &rhs.value).conj());
        self.backward(grad * rhs.value.recip().conj());
    }
}

// NOTE: float only.
two_diff_op_impl! {
    Atan2; atan2; "atan2_back";
    where
        Vrhs: AllocLike<Scalar = T>,
        &'a V | 'a: Neg<Output = V::Alloc>,
        &'a V | 'a: Mul<Output = V::Alloc>,
        &'a Vrhs | 'a: Mul<Output = Vrhs::Alloc>,
        V::Alloc: Add<Vrhs::Alloc, Output = V::Output>,
        &'a B | 'a: Mul<&'a V::Alloc, Output = B::Alloc>,
        B::Alloc | 'a: Div<&'a V::Output, Output = B::Alloc>,
        B | 'a: Mul<&'a Vrhs, Output = B>,
        B: Div<V::Output, Output = B>;
    (self, rhs) => move |grad| {
        let norm_sqr = &self.value * &self.value + &rhs.value * &rhs.value;
        rhs.backward(&grad * &(-&self.value) / &norm_sqr);
        self.backward(grad * &rhs.value / norm_sqr);
    }
}

// NOTE: float only.
two_diff_op_impl! {
    Hypot; hypot; "hypot_back";
    where
        &'a B | 'a: Mul<&'a Vrhs, Output = B::Alloc>,
        B::Alloc | 'a: Div<&'a V::Output, Output = B::Alloc>,
        B | 'a: Mul<&'a V, Output = B>,
        B: Div<V::Output, Output = B>;
    (self, rhs) => move |grad| {
        let result = self.value.hypot(&rhs.value);
        rhs.backward(&grad * &rhs.value / &result);
        self.backward(grad * &self.value / result);
    }
}

// NOTE: float only.
two_diff_op_impl! {
    Copysign; copysign; "copysign_back";
    where
        Vrhs: AllocLike<Scalar = T>,
        &'a V | 'a: Signum<Output = V::Alloc>,
        &'a Vrhs | 'a: Signum<Output = Vrhs::Alloc>,
        B: Mul<V::Alloc, Output = B>,
        B: Mul<Vrhs::Alloc, Output = B>;
    (self, rhs) => move |grad| {
        rhs.backward(grad.fill_like(T::ZERO));
        self.backward(grad * self.value.signum() * rhs.value.signum());
    }
}

// NOTE: float only.
two_diff_op_impl! {
    Max; max; "max_back";
    where
        &'a V | 'a: MaxMask<&'a Vrhs, Output = V::Output>,
        T: Neg<Output = T> | One,
        &'a V::Output | 'a: MulAdd<T, T, Output = V::Output>,
        &'a B | 'a: Mul<&'a V::Output, Output = B::Alloc>,
        B: Mul<V::Output, Output = B>;
    (self, rhs) => move |grad| {
        let mask = self.value.max_mask(&rhs.value);
        rhs.backward(&grad * &mask.mul_add(-T::ONE, T::ONE));
        self.backward(grad * mask);
    }
}

// NOTE: float only.
two_diff_op_impl! {
    Min; min; "min_back";
    where
        &'a V | 'a: MinMask<&'a Vrhs, Output = V::Output>,
        T: Neg<Output = T> | One,
        &'a V::Output | 'a: MulAdd<T, T, Output = V::Output>,
        &'a B | 'a: Mul<&'a V::Output, Output = B::Alloc>,
        B: Mul<V::Output, Output = B>;
    (self, rhs) => move |grad| {
        let mask = self.value.min_mask(&rhs.value);
        rhs.backward(&grad * &mask.mul_add(-T::ONE, T::ONE));
        self.backward(grad * mask);
    }
}

macro_rules! three_diff_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs0:ident, $rhs1:ident) => $backward_closure:expr
    ) => {
        impl<'var, T, V, G, B, Vrhs0, Grhs0, Vrhs1, Grhs1> $trait_name<Variable<'var, T, Vrhs0, Grhs0, B::Alloc>, Variable<'var, T, Vrhs1, Grhs1, B::Alloc>> for Variable<'var, T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<&'a Vrhs0, &'a Vrhs1, Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward gradient "copy" in output backward closure.
            B: AllocLike<Scalar = T>,

            // Backward calls on inputs in output backward closure.
            G: BackwardCompatible<B>,
            Grhs0: BackwardCompatible<B::Alloc>,
            Grhs1: BackwardCompatible<B::Alloc>,

            // Lifetime bounds required by output backward closure.
            T: 'var,
            V: 'var,
            G: 'var,
            B: 'var,
            Vrhs0: 'var,
            Grhs0: 'var,
            Vrhs1: 'var,
            Grhs1: 'var,
            B::Alloc: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<'var, T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self, $rhs0: Variable<'var, T, Vrhs0, Grhs0, B::Alloc>, $rhs1: Variable<'var, T, Vrhs1, Grhs1, B::Alloc>) -> Self::Output {
                let value = $self.value.$fn_name(&$rhs0.value, &$rhs1.value);

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(None),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

three_diff_op_impl! {
    MulAdd; mul_add; "mul_add_back";
    where
        Vrhs0: AllocLike<Scalar = T>,
        &'a V | 'a: Conj<Output = V::Alloc>,
        &'a B | 'a: Mul<V::Alloc, Output = B::Alloc>,
        &'a Vrhs0 | 'a: Conj<Output = Vrhs0::Alloc>,
        B: Mul<Vrhs0::Alloc, Output = B>;
    (self, rhs0, rhs1) => move |grad| {
        rhs1.backward(grad.to_contiguous());
        rhs0.backward(&grad * self.value.conj());
        self.backward(grad * rhs0.value.conj());
    }
}

macro_rules! diff_nondiff_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $backward_closure:expr
    ) => {
        impl<'var, T, V, G, B, Rhs> $trait_name<Rhs> for Variable<'var, T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<Rhs, Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: BackwardCompatible<B>,
            T: Copy,
            Rhs: Copy,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,
            Rhs: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<'var, T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self, $rhs: Rhs) -> Self::Output {
                let value = $self.value.$fn_name($rhs);

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(None),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

diff_nondiff_op_impl! {
    Pow; pow; "pow_back";
    where
        B: ZeroOut,
        Rhs: Zero | One | PartialEq | Sub<Output = Rhs>,
        V::Alloc: Mul<Rhs, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self, rhs) => move |mut grad| {
        self.backward(if rhs == Rhs::ZERO {
            grad.zero_out();
            grad
        } else if rhs != Rhs::ONE {
            grad * (self.value.pow(rhs - Rhs::ONE) * rhs).conj()
        } else {
            grad
        });
    }
}

diff_nondiff_op_impl! {
    Log; log; "log_back";
    where
        Rhs: Ln<Output = Rhs>,
        &'a V | 'a: Mul<Rhs, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self, rhs) => move |grad| {
        self.backward(grad / (&self.value * rhs.ln()).conj());
    }
}

macro_rules! fn_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident) => $backward_closure:expr
    ) => {
        impl<'var, T, V, G, B> $trait_name for Variable<'var, T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: BackwardCompatible<B>,

            // 'static bounds required by output backward closure.
            T: 'var,
            V: 'var,
            G: 'var,
            B: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<'var, T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self) -> Self::Output {
                let value = $self.value.$fn_name();
                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(None),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

fn_impl! {
    Exp; exp; "exp_back";
    where
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.exp().conj());
    }
}

fn_impl! {
    Exp2; exp2; "exp2_back";
    where
        T: Ln2,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>,
        B: Mul<T, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.exp2().conj() * T::LN_2);
    }
}

fn_impl! {
    ExpM1; exp_m1; "exp_m1_back";
    where
        &'a V | 'a: Exp<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.exp().conj());
    }
}

fn_impl! {
    Ln; ln; "ln_back";
    where
        &'a V | 'a: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / self.value.conj());
    }
}

fn_impl! {
    Ln1p; ln_1p; "ln_1p_back";
    where
        T: One,
        T | 'a: Add<&'a V, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (T::ONE + &self.value).conj());
    }
}

fn_impl! {
    Log2; log2; "log2_back";
    where
        T: Ln2,
        &'a V | 'a: Conj<Output = V::Alloc>,
        V::Alloc: Mul<T, Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (self.value.conj() * T::LN_2));
    }
}

fn_impl! {
    Log10; log10; "log2_back";
    where
        T: Ln10,
        &'a V | 'a: Conj<Output = V::Alloc>,
        V::Alloc: Mul<T, Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (self.value.conj() * T::LN_10));
    }
}

fn_impl! {
    Sin; sin; "sin_back";
    where
        &'a V | 'a: Cos<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.cos().conj());
    }
}

fn_impl! {
    Cos; cos; "cos_back";
    where
        &'a V | 'a: Sin<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        V::Alloc: Neg<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * -self.value.sin().conj());
    }
}

fn_impl! {
    Tan; tan; "tan_back";
    where
        T: One,
        &'a V::Alloc | 'a: Mul<Output = V::Alloc>,
        T: Add<V::Alloc, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        let result = self.value.tan();
        self.backward(grad * (T::ONE + &result * &result).conj());
    }
}

fn_impl! {
    Sinh; sinh; "sinh_back";
    where
        &'a V | 'a: Cosh<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.cosh().conj());
    }
}

fn_impl! {
    Cosh; cosh; "cosh_back";
    where
        &'a V | 'a: Sinh<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.sinh().conj());
    }
}

fn_impl! {
    Tanh; tanh; "tanh_back";
    where
        T: One,
        &'a V::Alloc | 'a: Mul<Output = V::Alloc>,
        T: Sub<V::Alloc, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        let result = self.value.tanh();
        self.backward(grad * (T::ONE - &result * &result).conj());
    }
}

fn_impl! {
    Asin; asin; "asin_back";
    where
        T: One,
        &'a V | 'a: Neg<Output = V::Alloc>,
        V::Alloc | 'a: Mul<&'a V, Output = V::Alloc>,
        V::Alloc: Add<T, Output = V::Alloc>,
        V::Alloc: Sqrt<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (-&self.value * &self.value + T::ONE).sqrt().conj())
    }
}

fn_impl! {
    Acos; acos; "acos_back";
    where
        T: One,
        &'a V | 'a: Neg<Output = V::Alloc>,
        V::Alloc | 'a: Mul<&'a V, Output = V::Alloc>,
        V::Alloc: Add<T, Output = V::Alloc>,
        V::Alloc: Sqrt<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        V::Alloc: Neg<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / -(-&self.value * &self.value + T::ONE).sqrt().conj())
    }
}

fn_impl! {
    Atan; atan; "atan_back";
    where
        T: One,
        &'a V | 'a: Mul<Output = V::Alloc>,
        V::Alloc: Add<T, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (&self.value * &self.value + T::ONE).conj())
    }
}

fn_impl! {
    Asinh; asinh; "asinh_back";
    where
        T: One,
        &'a V | 'a: Mul<Output = V::Alloc>,
        V::Alloc: Add<T, Output = V::Alloc>,
        V::Alloc: Sqrt<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (&self.value * &self.value + T::ONE).sqrt().conj())
    }
}

fn_impl! {
    Acosh; acosh; "acosh_back";
    where
        T: One,
        &'a V | 'a: Mul<Output = V::Alloc>,
        V::Alloc: Sub<T, Output = V::Alloc>,
        V::Alloc: Sqrt<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (&self.value * &self.value - T::ONE).sqrt().conj())
    }
}

fn_impl! {
    Atanh; atanh; "atanh_back";
    where
        T: One,
        &'a V | 'a: Mul<Output = V::Alloc>,
        T: Sub<V::Alloc, Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (T::ONE - &self.value * &self.value).conj())
    }
}

fn_impl! {
    Sqrt; sqrt; "sqrt_back";
    where
        i32: Cast<T>,
        V::Alloc: Conj<Output = V::Alloc>,
        T: Mul<V::Alloc, Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (2.as_() * self.value.sqrt().conj()))
    }
}

fn_impl! {
    Cbrt; cbrt; "cbrt_back";
    where
        i32: Cast<T>,
        &'a V | 'a: Mul<Output = V::Alloc>,
        V::Alloc: Cbrt<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        T: Mul<V::Alloc, Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / (3.as_() * (&self.value * &self.value).cbrt().conj()))
    }
}

fn_impl! {
    Abs; abs; "abs_back";
    where
        &'a V | 'a: Signum<Output = V::Alloc>,
        B: Mul<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad * self.value.signum());
    }
}

macro_rules! zero_grad_ops_impl {
    ($($trait:ident, $trait_fn:ident, $op_name:expr);*) => {$(
        fn_impl! {
            $trait; $trait_fn; $op_name;
            where
                B | 'a: ZeroOut;
            (self) => move |mut grad| {
                grad.zero_out();
                self.backward(grad);
            }
        }
    )*};
}

zero_grad_ops_impl! {
    Signum, signum, "signum_back";
    Ceil, ceil, "ceil_back";
    Floor, floor, "floor_back";
    Round, round, "round_back";
    Trunc, trunc, "trunc_assign";
    Fract, fract, "fract_assign";
    Conj, conj, "conj_back"
}

fn_impl! {
    Recip; recip; "recip_back";
    where
        &'a V | 'a: Mul<Output = V::Alloc>,
        V::Alloc: Conj<Output = V::Alloc>,
        V::Alloc: Neg<Output = V::Alloc>,
        B: Div<V::Alloc, Output = B>;
    (self) => move |grad| {
        self.backward(grad / -(&self.value * &self.value).conj());
    }
}

fn_impl! {
    ToDegrees; to_degrees; "abs_back";
    where
        T: RadDeg,
        B: Mul<T, Output = B>;
    (self) => move |grad| {
        self.backward(grad * T::RAD_TO_DEG);
    }
}

fn_impl! {
    ToRadians; to_radians; "abs_back";
    where
        T: RadDeg,
        B: Mul<T, Output = B>;
    (self) => move |grad| {
        self.backward(grad * T::DEG_TO_RAD);
    }
}

macro_rules! cross_domain_fn_impl {
    (
        $wrapper_trait_name:ident<Output = $output_type:ty> for $t:ty;
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where trait $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        $(where fn $($generic_fn:ty $(| $($lgen_fn:lifetime),+)?: $($bound_fn:path)|*),*;)?
        ($self:ident) => $backward_closure:expr
    ) => {
        /// Wrapper trait that allows an adaptive `B` parameter in the output `Variable`.
        pub trait $wrapper_trait_name<'var> {
            /// Scalar type `T` of the output `Variable`.
            type OutputScalar;
            /// `V` parameter of the output `Variable`.
            type OutputValue;
            /// `G` parameter of the output `Variable`.
            type OutputGrad;
            /// `B` parameter of the input `Variable`.
            type Back;
            /// Trait function generic over `Bnext`, the adaptive `B` parameter of the output `Variable`.
            fn $fn_name<Bnext>(self) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        impl<'var, T, V, G, B> $wrapper_trait_name<'var> for Variable<'var, $t, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocSameShape<$output_type>,
            $output_type: Zero,

            // Backward call on input in output backward closure.
            G: BackwardCompatible<B>,

            // Lifetime bounds required by output backward closure.
            T: 'var,
            V: 'var,
            G: 'var,
            B: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = $output_type;
            type OutputValue = V::Alloc;
            type OutputGrad = V::Alloc;
            type Back = B;
            fn $fn_name<Bnext>($self) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = $self.value.$fn_name();

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(None),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

cross_domain_fn_impl! {
    VariableRe<Output = T> for Complex<T>;
    Re; re; "re_back";
    (self) => move |grad| {
        self.backward(grad.as_())
    }
}

cross_domain_fn_impl! {
    VariableIm<Output = T> for Complex<T>;
    Im; im; "im_back";
    where fn
        &'a Bnext | 'a: J<Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.j())
    }
}

cross_domain_fn_impl! {
    VariableNorm<Output = T> for Complex<T>;
    Norm; norm; "norm_back";
    where trait
        B | 'a: Mul<&'a V, Output = B>;
    where fn
        Bnext: Div<Self::OutputValue, Output = Bnext>;
    (self) => move |grad| {
        self.backward((grad / self.value.norm()).as_() * &self.value )
    }
}

cross_domain_fn_impl! {
    VariableNormSqr<Output = T> for Complex<T>;
    NormSqr; norm_sqr; "norm_sqr_back";
    where trait
        B | 'a: Mul<&'a V, Output = B>,
        i32: Cast<T>;
    where fn
        Bnext: Mul<Self::OutputScalar, Output = Bnext>;
    (self) => move |grad| {
        self.backward((grad * 2.as_()).as_() * &self.value);
    }
}

cross_domain_fn_impl! {
    VariableArg<Output = T> for Complex<T>;
    Arg; arg; "arg_back";
    where trait
        &'a V | 'a: NormSqr<Output = V::Alloc>,
        B | 'a: Mul<&'a V, Output = B>;
    where fn
        Bnext: Div<Self::OutputValue, Output = Bnext>,
        Bnext: J<Output = Self::Back>;
    (self) => move |grad| {
        self.backward((grad / self.value.norm_sqr()).as_() * &self.value);
    }
}

cross_domain_fn_impl! {
    VariableJ<Output = Complex<T>> for T;
    J; j; "j_back";
    where trait
        T: Zero | One;
    where fn
        Bnext: Conj<Output = Bnext>,
        Bnext: Mul<Self::OutputScalar, Output = Bnext>;
    (self) => move |grad| {
        self.backward((grad.conj() * Complex::new(T::ZERO, T::ONE)).as_());
    }
}

cross_domain_fn_impl! {
    VariableEPowJ<Output = Complex<T>> for T;
    EPowJ; e_pow_j; "e_pow_j_back";
    where trait
        T: Zero | One;
    where fn
        Bnext: Conj<Output = Bnext>,
        Bnext | 'a: Mul<Self::OutputValue, Output = Bnext>,
        Bnext: Mul<Self::OutputScalar, Output = Bnext>;
    (self) => move |grad| {
        self.backward((grad.conj() * self.value.e_pow_j() * Complex::new(T::ZERO, T::ONE)).as_())
    }
}

macro_rules! cross_domain_two_diff_op_impl {
    (
        $wrapper_trait_name:ident<Output = $output_type:ty> for $t:ty;
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where trait $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        $(where fn $($generic_fn:ty $(| $($lgen_fn:lifetime),+)?: $($bound_fn:path)|*),*;)?
        ($self:ident, $rhs:ident) => $backward_closure:expr
    ) => {
        /// Wrapper trait that allows an adaptive `B` parameter in the output `Variable`.
        pub trait $wrapper_trait_name<'var, Rhs = Self> {
            /// Scalar type `T` of the output `Variable`.
            type OutputScalar;
            /// `V` parameter of the output `Variable`.
            type OutputValue;
            /// `G` parameter of the output `Variable`.
            type OutputGrad;
            /// `B` parameter of the input `Variable`.
            type Back;
            /// `Vrhs::Alloc` where `Vrhs` is the `V` parameter of `Rhs`.
            ///
            /// This is useful to write bounds on `Bnext` that are parametrized by `Vrhs::Alloc`.
            /// Such bounds must indeed be attached to the trait via the function. The problem
            /// is that the trait has no knowledge of `Vrhs` that is defined in the impl.
            type VrhsAlloc;
            /// Trait function generic over `Bnext`, the adaptive `B` parameter of the output `Variable`.
            fn $fn_name<Bnext>(self, rhs: Rhs) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        impl<'var, T, V, G, B, Vrhs, Grhs> $wrapper_trait_name<'var, Variable<'var, $t, Vrhs, Grhs, B>> for Variable<'var, $t, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<&'a Vrhs, Output = V::Output>,
            V: BinaryOp<Vrhs>,
            // Output variable gradient allocation.
            V: AllocSameShape<$output_type>,
            Vrhs: AllocSameShape<$output_type>,
            $output_type: Zero,

            // Backward call on inputs in output backward closure.
            G: BackwardCompatible<B>,
            Grhs: BackwardCompatible<B>,

            // Lifetime bounds required by output backward closure.
            T: 'var,
            V: 'var,
            G: 'var,
            B: 'var,
            Vrhs: 'var,
            Grhs: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = $output_type;
            type OutputValue = V::Output;
            type OutputGrad = V::Alloc;
            type Back = B;
            type VrhsAlloc = Vrhs::Alloc;
            fn $fn_name<Bnext>($self, $rhs: Variable<'var, $t, Vrhs, Grhs, B>) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = $self.value.$fn_name(&$rhs.value);

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(None),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

cross_domain_two_diff_op_impl! {
    VariableAddJ<Output = Complex<T>> for T;
    AddJ; add_j; "add_j_back";
    where trait
        T: Zero | One;
    where fn
        Bnext: ConjAssign,
        Bnext: Mul<Self::OutputScalar, Output = Bnext>;
    (self, rhs) => move |mut grad| {
        grad.conj_assign();
        self.backward(grad.as_());
        rhs.backward((grad * Complex::new(T::ZERO, T::ONE)).as_());
    }
}

cross_domain_two_diff_op_impl! {
    VariableMulEPowJ<Output = Complex<T>> for T;
    MulEPowJ; mul_e_pow_j; "mul_e_pow_j_back";
    where trait
        T: Zero | One,
        Vrhs: AllocSameShape<Complex<T>>,
        &'a Vrhs | 'a: EPowJ<Output = Vrhs::Alloc>,
        &'a V | 'a: J<Output = V::Alloc>;
    where fn
        Bnext: Conj<Output = Bnext>,
        Bnext: Mul<Self::VrhsAlloc, Output = Bnext>,
        Bnext: Mul<Self::OutputGrad, Output = Bnext>;
    (self, rhs) => move |grad| {
        let grad = grad.conj() * rhs.value.e_pow_j();
        self.backward(grad.as_());
        rhs.backward((grad * self.value.j()).as_());
    }
}
