//! Implements the basic operations defined
//! by the traits of the [`ops`](crate::ops)
//! module for `Variables`.

use super::variable::{InternalVariable, Variable};
use crate::ops::*;
use crate::scalar_traits::*;
use crate::tensor::prelude::*;
use num_complex::{Complex, Complex32, Complex64};
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
macro_rules! binary_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $backward_closure:expr
    ) => {
        impl<T, V, G, B, Vrhs, Grhs> $trait_name<Variable<T, Vrhs, Grhs, B::Alloc>> for Variable<T, V, G, B>
        where
            // Forward operation.
            for<'a, 'b> &'a V: $trait_name<&'b Vrhs, Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward gradient "copy" in output backward closure.
            B: AllocLike<Scalar = T>,

            // Backward calls on inputs in output backward closure.
            G: for<'a> AddAssign<&'a B>,
            Grhs: for<'a> AddAssign<&'a B::Alloc>,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,
            Vrhs: 'static,
            Grhs: 'static,
            B::Alloc: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self, $rhs: Variable<T, Vrhs, Grhs, B::Alloc>) -> Self::Output {
                let value = $self.value.$fn_name(&$rhs.value);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    let rhs_grad = $rhs.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else if let Some(_) = *rhs_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

binary_op_impl! {
    Add; add; "add_back";
    (self, rhs) => move |grad| {
        rhs.backward(grad.to_contiguous());
        self.backward(grad);
    }
}

binary_op_impl! {
    Sub; sub; "sub_back";
    where
        &'a B | 'a: Mul<T, Output = B::Alloc>,
        T: Neg<Output=T> | One;
    (self, rhs) => move |grad| {
        rhs.backward(grad.mul(-T::ONE));
        self.backward(grad);
    }
}

binary_op_impl! {
    Mul; mul; "mul_back";
    where
        Vrhs: AllocLike<Scalar = T>,
        &'a V | 'a: Conj<Output = V::Alloc>,
        &'a Vrhs | 'a: Conj<Output = Vrhs::Alloc>,
        &'a B | 'a, 'b: Mul<&'b V::Alloc, Output = B::Alloc>,
        B | 'a: MulAssign<&'a Vrhs::Alloc>;
    (self, rhs) => move |mut grad| {
        rhs.backward(grad.mul(&self.value.conj()));
        grad.mul_assign(&rhs.value.conj());
        self.backward(grad);
    }
}

binary_op_impl! {
    Div; div; "div_back";
    where
        T: Neg<Output = T> | One,
        Vrhs: AllocLike<Scalar = T>,
        &'a Vrhs | 'a: Pow<i32, Output = Vrhs::Alloc>,
        Vrhs::Alloc: RecipAssign,
        Vrhs::Alloc | 'a: MulAssign<&'a V>,
        Vrhs::Alloc: MulAssign<T>,
        Vrhs::Alloc: ConjAssign,
        &'a B | 'a, 'b: Mul<&'b Vrhs::Alloc, Output = B::Alloc>,
        &'a Vrhs | 'a: Conj<Output = Vrhs::Alloc>,
        B | 'a: DivAssign<&'a Vrhs::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut rhs_part_grad = rhs.value.pow(2);
        rhs_part_grad.recip_assign();
        rhs_part_grad.mul_assign(&self.value);
        rhs_part_grad.mul_assign(-T::ONE);
        rhs_part_grad.conj_assign();
        rhs.backward(grad.mul(&rhs_part_grad));
        
        grad.div_assign(&rhs.value.conj());
        self.backward(grad);
    }
}

binary_op_impl! {
    Atan2; atan2; "atan2_back";
    where
        Vrhs: AllocLike,
        T: Neg<Output = T> | One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        &'a Vrhs | 'a: Pow<i32, Output = Vrhs::Alloc>,
        V::Alloc | 'a: AddAssign<&'a Vrhs::Alloc>,
        B::Alloc | 'a: MulAssign<&'a Vrhs>,
        B::Alloc | 'a: MulAssign<&'a V>,
        B::Alloc | 'a: DivAssign<&'a V::Alloc>,
        B::Alloc: MulAssign<T>,
        B | 'a: MulAssign<&'a Vrhs>,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut div = self.value.pow(2);
        div.add_assign(&rhs.value.pow(2));

        let mut rhs_grad = grad.to_contiguous();
        rhs_grad.mul_assign(&rhs.value);
        rhs_grad.mul_assign(&self.value);
        rhs_grad.div_assign(&div);
        rhs_grad.mul_assign(-T::ONE);
        rhs.backward(rhs_grad);

        grad.mul_assign(&rhs.value);
        grad.div_assign(&div);
        self.backward(grad);
    }
}

binary_op_impl! {
    Copysign; copysign; "copysign_back";
    where
        Vrhs: AllocLike,
        B::Alloc | 'a: MulAssign<&'a V>,
        &'a V | 'a: Signum<Output = V::Alloc>,
        &'a Vrhs | 'a: Signum<Output = Vrhs::Alloc>,
        &'a V::Alloc | 'a, 'b: Mul<&'b Vrhs::Alloc, Output = V::Alloc>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        rhs.backward(grad.fill_like(T::ZERO));
        grad.mul_assign(&self.value.signum().mul(&rhs.value.signum()));
        self.backward(grad);
    }
}

binary_op_impl! {
    Max; max; "max_back";
    where
        &'a V | 'a, 'b: Argmax<&'b Vrhs, Output = V::Alloc>,
        B::Alloc | 'a: MulAssign<&'a V::Alloc>,
        T: Neg<Output = T> | One,
        V::Alloc: MulAddAssign<T, T>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmax(&rhs.value);
        let mut rhs_grad = grad.to_contiguous();
        rhs_grad.mul_assign(&mask);
        rhs.backward(rhs_grad);

        mask.mul_add_assign(-T::ONE, T::ONE);
        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

binary_op_impl! {
    Min; min; "min_back";
    where
        &'a V | 'a, 'b: Argmin<&'b Vrhs, Output = V::Alloc>,
        B::Alloc | 'a: MulAssign<&'a V::Alloc>,
        T: Neg<Output = T> | One,
        V::Alloc: MulAddAssign<T, T>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmin(&rhs.value);
        let mut rhs_grad = grad.to_contiguous();
        rhs_grad.mul_assign(&mask);
        rhs.backward(rhs_grad);

        mask.mul_add_assign(-T::ONE, T::ONE);
        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

macro_rules! ternary_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs0:ident, $rhs1:ident) => $backward_closure:expr
    ) => {
        impl<T, V, G, B, Vrhs0, Grhs0, Vrhs1, Grhs1> $trait_name<Variable<T, Vrhs0, Grhs0, B::Alloc>, Variable<T, Vrhs1, Grhs1, B::Alloc>> for Variable<T, V, G, B>
        where
            // Forward operation.
            for<'a, 'b, 'c> &'a V: $trait_name<&'b Vrhs0, &'c Vrhs1, Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward gradient "copy" in output backward closure.
            B: AllocLike<Scalar = T>,

            // Backward calls on inputs in output backward closure.
            G: for<'a> AddAssign<&'a B>,
            Grhs0: for<'a> AddAssign<&'a B::Alloc>,
            Grhs1: for<'a> AddAssign<&'a B::Alloc>,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,
            Vrhs0: 'static,
            Grhs0: 'static,
            Vrhs1: 'static,
            Grhs1: 'static,
            B::Alloc: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self, $rhs0: Variable<T, Vrhs0, Grhs0, B::Alloc>, $rhs1: Variable<T, Vrhs1, Grhs1, B::Alloc>) -> Self::Output {
                let value = $self.value.$fn_name(&$rhs0.value, &$rhs1.value);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    let rhs0_grad = $rhs0.grad.borrow();
                    let rhs1_grad = $rhs1.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else if let Some(_) = *rhs0_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else if let Some(_) = *rhs1_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

ternary_op_impl! {
    MulAdd; mul_add; "mul_add_back";
    where
        Vrhs0: AllocLike<Scalar = T>,
        &'a V | 'a: Conj<Output = V::Alloc>,
        &'a Vrhs0 | 'a: Conj<Output = Vrhs0::Alloc>,
        &'a B | 'a, 'b: Mul<&'b V::Alloc, Output = B::Alloc>,
        B | 'a: MulAssign<&'a Vrhs0::Alloc>;
    (self, rhs0, rhs1) => move |mut grad| {
        rhs1.backward(grad.to_contiguous());
        rhs0.backward(grad.mul(&self.value.conj()));
        grad.mul_assign(&rhs0.value.conj());
        self.backward(grad);
    }
}

macro_rules! one_param_op_impl {
    (
        $trait_name:ident$(<$param:ident>)?;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $backward_closure:expr
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => {
                $var
            };
            () => {
                T
            };
        }

        impl<T, V, G, B $(,$param)?> $trait_name<isset_or_default!($($param)?)> for Variable<T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<isset_or_default!($($param)?), Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: for<'a> AddAssign<&'a B>,
            T: Copy,
            $($param: Copy,)?

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,
            $($param: 'static,)?

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self, $rhs: isset_or_default!($($param)?)) -> Self::Output {
                let value = $self.value.$fn_name($rhs);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

one_param_op_impl! {
    Add; add; "add_scalar_back";
    (self, rhs) => move |grad| {
        self.backward(grad);
    }
}

one_param_op_impl! {
    Sub; sub; "sub_scalar_back";
    (self, rhs) => move |grad| {
        self.backward(grad);
    }
}

one_param_op_impl! {
    Mul; mul; "mul_scalar_back";
    where
        T: Conj<Output = T>,
        B: MulAssign<T>;
    (self, rhs) => move |mut grad| {
        grad.mul_assign(rhs.conj());
        self.backward(grad);
    }
}

one_param_op_impl! {
    Div; div; "div_scalar_back";
    where
        T: Conj<Output = T>,
        B: DivAssign<T>;
    (self, rhs) => move |mut grad| {
        grad.div_assign(rhs.conj());
        self.backward(grad);
    }
}

one_param_op_impl! {
    Max; max; "max_scalar_back";
    where
        &'a V | 'a: Argmax<T, Output = V::Alloc>,
        T: Neg<Output = T> | One,
        V::Alloc: MulAddAssign<T, T>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmax(rhs);
        mask.mul_add_assign(-T::ONE, T::ONE);
        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

one_param_op_impl! {
    Min; min; "min_scalar_back";
    where
        &'a V | 'a: Argmin<T, Output = V::Alloc>,
        T: Neg<Output = T> | One,
        V::Alloc: MulAddAssign<T, T>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmin(rhs);
        mask.mul_add_assign(-T::ONE, T::ONE);
        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

one_param_op_impl! {
    Pow<Rhs>; pow; "pow_back";
    where
        Rhs: Zero | One | PartialEq | Sub<Output = Rhs>,
        V::Alloc: MulAssign<Rhs>,
        V::Alloc: ConjAssign,
        B: MulAssign<Rhs>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        if rhs == Rhs::ZERO {
            grad.mul_assign(Rhs::ZERO);
        } else if rhs != Rhs::ONE {
            let mut part_grad = self.value.pow(rhs - Rhs::ONE);
            part_grad.mul_assign(rhs);
            part_grad.conj_assign();
            grad.mul_assign(&part_grad);
        }
        self.backward(grad);
    }
}

// NOTE: the scalar type cannot be made generic
// as it would violate the orphan rule.
macro_rules! reversed_one_param_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $forward:block => $backward_closure:expr
    ) => {
        macro_rules! inner_impl {
            ($t:ty) => {
                impl<V, G, B> $trait_name<Variable<$t, V, G, B>> for $t
                where
                    // Forward operation.
                    for<'a> &'a V: $trait_name<$t, Output = V::Alloc>,
                    // Output variable gradient allocation.
                    V: AllocLike<Scalar = $t>,
                    $t: Zero,
                    // Backward gradient "copy" in output backward closure.
                    B: AllocLike<Scalar = $t>,
                    // Backward calls on inputs in output backward closure.
                    G: for<'a> AddAssign<&'a B>,
                    // 'static bounds required by output backward closure.
                    V: 'static,
                    G: 'static,
                    B: 'static,
                    // Additionnal bounds needed by either forward or
                    // backward passes
                    $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
                {
                    type Output = Variable<$t, V::Alloc, V::Alloc, B>;
                    fn $fn_name($self, $rhs: Variable<$t, V, G, B>) -> Self::Output {
                        let value = $forward;
                        let grad = {
                            let rhs_grad = $rhs.grad.borrow();
                            if let Some(_) = *rhs_grad {
                                Some($rhs.value.fill_like(<$t>::ZERO))
                            } else {
                                None
                            }
                        };
                        Variable(Rc::new(InternalVariable {
                            value,
                            grad: RefCell::new(grad),
                            backward_op_name: $op_name,
                            backward_closure: Box::new($backward_closure),
                        }), PhantomData)
                    }
                }
            };
        }

        inner_impl! { f64 }
        inner_impl! { f32 }
        inner_impl! { Complex64 }
        inner_impl! { Complex32 }
    };
}

reversed_one_param_op_impl! {
    Add; add; "scalar_add_back";
    (self, rhs) => {
        rhs.value.add(self)
    }
    => move |grad| {
        rhs.backward(grad);
    }
}

reversed_one_param_op_impl! {
    Sub; sub; "scalar_sub_back";
    where
        Self: One,
        &'a V | 'a: MulAdd<Self, Self, Output = V::Alloc>;
    (self, rhs) => {
        rhs.value.mul_add(-Self::ONE, Self::ONE)
    }
    => move |grad| {
        rhs.backward(grad);
    }
}

reversed_one_param_op_impl! {
    Mul; mul; "scalar_mul_back";
    where
        Self: Copy | Conj,
        B: MulAssign<Self>;
    (self, rhs) => {
        rhs.value.mul(self)
    }
    => move |mut grad| {
        grad.mul_assign(self.conj());
        rhs.backward(grad);
    }
}

reversed_one_param_op_impl! {
    Div; div; "scalar_div_back";
    where
        &'a V | 'a: Recip<Output = V::Alloc>,
        V::Alloc: MulAssign<Self>,
        Self: Neg<Output = Self> | One,
        V: AllocLike<Scalar = Self>,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: RecipAssign,
        V::Alloc: MulAssign<Self>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => {
        let mut value = rhs.value.recip();
        value.mul_assign(self);
        value
    }
    => move |mut grad| {
        let mut part_grad = rhs.value.pow(2);
        part_grad.recip_assign();
        part_grad.mul_assign(-self);
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        rhs.backward(grad);
    }
}

macro_rules! two_param_op_impl {
    (
        $trait_name:ident$(<$param0:ident, $param1:ident>)?;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs0:ident, $rhs1:ident) => $backward_closure:expr
    ) => {
        macro_rules! isset_or_default {
            ($var:ty) => {
                $var
            };
            () => {
                T
            };
        }

        impl<T, V, G, B $(,$param0, $param1)?> $trait_name<isset_or_default!($($param0)?), isset_or_default!($($param1)?)> for Variable<T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<isset_or_default!($($param0)?), isset_or_default!($($param1)?), Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: for<'a> AddAssign<&'a B>,
            T: Copy,
            $($param0: Copy,)?
            $($param1: Copy,)?

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,
            $($param0: 'static,)?
            $($param1: 'static,)?

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self, $rhs0: isset_or_default!($($param0)?), $rhs1: isset_or_default!($($param1)?)) -> Self::Output {
                let value = $self.value.$fn_name($rhs0, $rhs1);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

two_param_op_impl! {
    MulAdd; mul_add; "mul_add_scalar_back";
    where
        T: Conj<Output = T>,
        B | 'a: MulAssign<T>;
    (self, rhs0, rhs1) => move |mut grad| {
        grad.mul_assign(rhs0.conj());
        self.backward(grad);
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
        impl<T, V, G, B> $trait_name for Variable<T, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<Output = V::Alloc>,
            // Output variable gradient allocation.
            V: AllocLike<Scalar = T>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: for<'a> AddAssign<&'a B>,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = Variable<T, V::Alloc, V::Alloc, B>;
            fn $fn_name($self) -> Self::Output {
                let value = $self.value.$fn_name();
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_like(T::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
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
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.exp();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Exp2; exp2; "exp2_back";
    where
        T: Ln2,
        V::Alloc: MulAssign<V::Scalar>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.exp2();
        part_grad.mul_assign(T::LN_2);
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    ExpM1; exp_m1; "exp_m1_back";
    where
        &'a V | 'a: Exp<Output = V::Alloc>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.exp();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Ln; ln; "ln_back";
    where
        &'a V | 'a: Recip<Output = V::Alloc>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.recip();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Ln1p; ln_1p; "ln_1p_back";
    where
        T: One,
        &'a V | 'a: Add<T, Output = V::Alloc>,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.add(T::ONE);
        part_grad.recip_assign();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Log2; log2; "log2_back";
    where
        T: Ln2,
        &'a V | 'a: Mul<T, Output = V::Alloc>,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.mul(T::LN_2);
        part_grad.recip_assign();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Log10; log10; "log10_back";
    where
        T: Ln10,
        &'a V | 'a: Mul<T, Output = V::Alloc>,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.mul(T::LN_10);
        part_grad.recip_assign();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Sin; sin; "sin_back";
    where
        &'a V | 'a: Cos<Output = V::Alloc>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.cos();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Cos; cos; "cos_back";
    where
        T: Neg<Output = T> | One,
        &'a V | 'a: Sin<Output = V::Alloc>,
        V::Alloc: MulAssign<T>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.sin();
        part_grad.mul_assign(-T::ONE);
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Tan; tan; "tan_back";
    where
        T: One,
        V::Alloc: PowAssign<i32>,
        V::Alloc: AddAssign<T>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.tan();
        part_grad.pow_assign(2);
        part_grad.add_assign(T::ONE);
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Sinh; sinh; "sinh_back";
    where
        &'a V | 'a: Cosh<Output = V::Alloc>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.cosh();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Cosh; cosh; "cosh_back";
    where
        &'a V | 'a: Sinh<Output = V::Alloc>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.sinh();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Tanh; tanh; "tanh_back";
    where
        T: Neg<Output = T> | One,
        V::Alloc: PowAssign<i32>,
        V::Alloc: MulAssign<T>,
        V::Alloc: AddAssign<T>,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.tanh();
        part_grad.pow_assign(2);
        part_grad.mul_assign(-T::ONE);
        part_grad.add_assign(T::ONE);
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Asin; asin; "asin_back";
    where
        T: Neg<Output = T> | One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: MulAssign<T>,
        V::Alloc: AddAssign<T>,
        V::Alloc: SqrtAssign,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.mul_assign(-T::ONE);
        part_grad.add_assign(T::ONE);
        part_grad.sqrt_assign();
        part_grad.recip_assign();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Acos; acos; "acos_back";
    where
        T: Neg<Output = T> | One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: MulAssign<T>,
        V::Alloc: AddAssign<T>,
        V::Alloc: SqrtAssign,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.mul_assign(-T::ONE);
        part_grad.add_assign(T::ONE);
        part_grad.sqrt_assign();
        part_grad.recip_assign();
        part_grad.mul_assign(-T::ONE);
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Atan; atan; "atan_back";
    where
        T: One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: AddAssign<T>,
        V::Alloc: ConjAssign,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.add_assign(T::ONE);
        part_grad.conj_assign();
        grad.div_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Asinh; asinh; "asinh_back";
    where
        T: One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: AddAssign<T>,
        V::Alloc: SqrtAssign,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.add_assign(T::ONE);
        part_grad.sqrt_assign();
        part_grad.recip_assign();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Acosh; acosh; "acosh_back";
    where
        T: One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: SubAssign<T>,
        V::Alloc: SqrtAssign,
        V::Alloc: RecipAssign,
        V::Alloc: ConjAssign,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.sub_assign(T::ONE);
        part_grad.sqrt_assign();
        part_grad.recip_assign();
        part_grad.conj_assign();
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Atanh; atanh; "atanh_back";
    where
        T: Neg<Output = T> | One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: MulAssign<T>,
        V::Alloc: AddAssign<T>,
        V::Alloc: ConjAssign,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.mul_assign(-T::ONE);
        part_grad.add_assign(T::ONE);
        part_grad.conj_assign();
        grad.div_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Sqrt; sqrt; "sqrt_back";
    where
        i32: Cast<T>,
        V::Alloc: ConjAssign,
        V::Alloc: MulAssign<T>,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.sqrt();
        part_grad.conj_assign();
        part_grad.mul_assign(2.as_());
        grad.div_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Cbrt; cbrt; "cbrt_back";
    where
        i32: Cast<T>,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: CbrtAssign,
        V::Alloc: ConjAssign,
        V::Alloc: MulAssign<T>,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.cbrt_assign();
        part_grad.conj_assign();
        part_grad.mul_assign(3.as_());
        grad.div_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    Abs; abs; "abs_back";
    where
        &'a V | 'a: Signum<Output = V::Alloc>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        grad.mul_assign(&self.value.signum());
        self.backward(grad);
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
    Round, round, "round_back"
}

fn_impl! {
    Recip; recip; "recip_back";
    where
        T: Neg<Output = T> | One,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        V::Alloc: ConjAssign,
        V::Alloc: MulAssign<T>,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.pow(2);
        part_grad.conj_assign();
        part_grad.mul_assign(-T::ONE);
        grad.div_assign(&part_grad);
        self.backward(grad);
    }
}

fn_impl! {
    ToDegrees; to_degrees; "abs_back";
    where
        T: RadDeg,
        B | 'a: MulAssign<T>;
    (self) => move |mut grad| {
        grad.mul_assign(T::RAD_TO_DEG);
        self.backward(grad);
    }
}

fn_impl! {
    ToRadians; to_radians; "abs_back";
    where
        T: RadDeg,
        B | 'a: MulAssign<T>;
    (self) => move |mut grad| {
        grad.mul_assign(T::DEG_TO_RAD);
        self.backward(grad);
    }
}

fn_impl! {
    Conj; conj; "conj_back";
    (self) => move |grad| {
        self.backward(grad);
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
        pub trait $wrapper_trait_name {
            /// Scalar type `T` of the output `Variable`.
            type OutputScalar;
            /// `V` parameter of the output `Variable`.
            type OutputValue;
            /// `G` parameter of the output `Variable`.
            type OutputGrad;
            /// `B` parameter of the input `Variable`.
            type Back;
            /// Trait function generic over `Bnext`, the adaptive `B` parameter of the output `Variable`.
            fn $fn_name<Bnext>(self) -> Variable<Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        
        impl<T, V, G, B> $wrapper_trait_name for Variable<$t, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<Output = V::Alloc>,
            
            // Output variable gradient allocation.
            V: AllocSameShape<$output_type>,
            $output_type: Zero,

            // Backward call on input in output backward closure.
            G: for<'a> AddAssign<&'a B>,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = $output_type;
            type OutputValue = V::Alloc;
            type OutputGrad = V::Alloc;
            type Back = B;
            fn $fn_name<Bnext>($self) -> Variable<Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = $self.value.$fn_name();
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_same_shape(<$output_type>::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
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
        B | 'a: MulAssign<&'a V>;
    where fn
        Bnext | 'a: DivAssign<&'a Self::OutputValue>;
    (self) => move |mut grad| {
        grad.div_assign(&self.value.norm());
        let mut grad = grad.as_();
        grad.mul_assign(&self.value);
        self.backward(grad);
    }
}

cross_domain_fn_impl! {
    VariableNormSqr<Output = T> for Complex<T>;
    NormSqr; norm_sqr; "norm_sqr_back";
    where trait
        B | 'a: MulAssign<&'a V>;
    where fn
        Bnext | 'a: DivAssign<&'a Self::OutputValue>;
    (self) => move |grad| {
        let mut grad = grad.as_();
        grad.mul_assign(&self.value);
        self.backward(grad);
    }
}

cross_domain_fn_impl! {
    VariableArg<Output = T> for Complex<T>;
    Arg; arg; "arg_back";
    where trait
        &'a V | 'a: NormSqr<Output = V::Alloc>,
        B | 'a: MulAssign<&'a V>;
    where fn
        Bnext | 'a: DivAssign<&'a Self::OutputValue>,
        &'a Bnext| 'a: J<Output = Self::Back>;
    (self) => move |mut grad| {
        grad.div_assign(&self.value.norm_sqr());
        let mut grad = grad.j();
        grad.mul_assign(&self.value);
        self.backward(grad);
    }
}

cross_domain_fn_impl! {
    VariableJ<Output = Complex<T>> for T;
    J; j; "j_back";
    where trait
        T: Zero | One;
    where fn
        Bnext: ConjAssign | MulAssign<Self::OutputScalar>;
    (self) => move |mut grad| {
        grad.conj_assign();
        grad.mul_assign(Complex::new(T::ZERO, T::ONE));
        self.backward(grad.as_());
    }
}

cross_domain_fn_impl! {
    VariableEPowJ<Output = Complex<T>> for T;
    EPowJ; e_pow_j; "e_pow_j_back";
    where trait
        T: Zero | One,
        V::Alloc: MulAssign<Complex<T>>;
    where fn
        Bnext: ConjAssign,
        Bnext | 'a: MulAssign<&'a Self::OutputValue>;
    (self) => move |mut grad| {
        grad.conj_assign();
        let mut part_grad = self.value.e_pow_j();
        part_grad.mul_assign(Complex::new(T::ZERO, T::ONE));
        grad.mul_assign(&part_grad);
        self.backward(grad.as_());
    }
}

macro_rules! cross_domain_binary_op_impl {
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
        pub trait $wrapper_trait_name<Rhs = Self> {
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
            fn $fn_name<Bnext>(self, rhs: Rhs) -> Variable<Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        
        impl<T, V, G, B, Vrhs, Grhs> $wrapper_trait_name<Variable<$t, Vrhs, Grhs, B>> for Variable<$t, V, G, B>
        where
            // Forward operation.
            for<'a, 'b> &'a V: $trait_name<&'b Vrhs, Output = V::Alloc>,
            
            // Output variable gradient allocation.
            V: AllocSameShape<$output_type>,
            Vrhs: AllocSameShape<$output_type>,
            $output_type: Zero,

            // Backward call on inputs in output backward closure.
            G: for<'a> AddAssign<&'a B>,
            Grhs: for<'a> AddAssign<&'a B>,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,
            Vrhs: 'static,
            Grhs: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = $output_type;
            type OutputValue = V::Alloc;
            type OutputGrad = V::Alloc;
            type Back = B;
            type VrhsAlloc = Vrhs::Alloc;
            fn $fn_name<Bnext>($self, $rhs: Variable<$t, Vrhs, Grhs, B>) -> Variable<Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = $self.value.$fn_name(&$rhs.value);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    let rhs_grad = $rhs.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_same_shape(<$output_type>::ZERO))
                    } else if let Some(_) = *rhs_grad {
                        Some($self.value.fill_same_shape(<$output_type>::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

cross_domain_binary_op_impl! {
    VariableAddJ<Output = Complex<T>> for T;
    AddJ; add_j; "add_j_back";
    where trait
        T: Zero | One;
    where fn
        Bnext: ConjAssign | MulAssign<Self::OutputScalar>;
    (self, rhs) => move |mut grad| {
        grad.conj_assign();
        self.backward(grad.as_());
        grad.mul_assign(Complex::new(T::ZERO, T::ONE));
        rhs.backward(grad.as_());
    }
}

cross_domain_binary_op_impl! {
    VariableMulEPowJ<Output = Complex<T>> for T;
    MulEPowJ; mul_e_pow_j; "mul_e_pow_j_back";
    where trait
        T: Zero | One,
        Vrhs: AllocSameShape<Complex<T>>,
        &'a Vrhs | 'a: EPowJ<Output = Vrhs::Alloc>,
        &'a V | 'a: J<Output = V::Alloc>;
    where fn
        Bnext: ConjAssign,
        Bnext | 'a: MulAssign<&'a Self::VrhsAlloc>,
        Bnext | 'a: MulAssign<&'a Self::OutputValue>;
    (self, rhs) => move |mut grad| {
        grad.conj_assign();
        grad.mul_assign(&rhs.value.e_pow_j());
        self.backward(grad.as_());
        grad.mul_assign(&self.value.j());
        rhs.backward(grad.as_());
    }
}

macro_rules! cross_domain_one_param_op_impl {
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
        pub trait $wrapper_trait_name<Rhs = Self> {
            /// Scalar type `T` of the output `Variable`.
            type OutputScalar;
            /// `V` parameter of the output `Variable`.
            type OutputValue;
            /// `G` parameter of the output `Variable`.
            type OutputGrad;
            /// `B` parameter of the input `Variable`.
            type Back;
            /// Trait function generic over `Bnext`, the adaptive `B` parameter of the output `Variable`.
            fn $fn_name<Bnext>(self, rhs: Rhs) -> Variable<Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        
        impl<T, V, G, B> $wrapper_trait_name<$t> for Variable<$t, V, G, B>
        where
            // Forward operation.
            for<'a> &'a V: $trait_name<$t, Output = V::Alloc>,
            
            // Output variable gradient allocation.
            V: AllocSameShape<$output_type>,
            $output_type: Zero,

            // Backward call on inputs in output backward closure.
            G: for<'a> AddAssign<&'a B>,

            // 'static bounds required by output backward closure.
            T: 'static,
            V: 'static,
            G: 'static,
            B: 'static,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = $output_type;
            type OutputValue = V::Alloc;
            type OutputGrad = V::Alloc;
            type Back = B;
            fn $fn_name<Bnext>($self, $rhs: $t) -> Variable<Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                Bnext: AllocSameShape<Complex<Self::OutputScalar>, Alloc = Self::Back>,
                for<'a> &'a Bnext: TensorCast<Complex<Self::OutputScalar>, Output = Self::Back>,
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = $self.value.$fn_name($rhs);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some($self.value.fill_same_shape(<$output_type>::ZERO))
                    } else {
                        None
                    }
                };

                Variable(Rc::new(InternalVariable {
                    value,
                    grad: RefCell::new(grad),
                    backward_op_name: $op_name,
                    backward_closure: Box::new($backward_closure),
                }), PhantomData)
            }
        }
    };
}

cross_domain_one_param_op_impl! {
    VariableAddJScalar<Output = Complex<T>> for T;
    AddJ; add_j; "add_j_scalar_back";
    where trait
        T: Zero | One;
    where fn
        Bnext: ConjAssign | MulAssign<Self::OutputScalar>;
    (self, rhs) => move |mut grad| {
        grad.conj_assign();
        self.backward(grad.as_());
    }
}

cross_domain_one_param_op_impl! {
    VariableMulEPowJScalar<Output = Complex<T>> for T;
    MulEPowJ; mul_e_pow_j; "mul_e_pow_j_scalar_back";
    where trait
        T: Zero | One | EPowJ<Output = Complex<T>> | Copy,
        &'a V | 'a: J<Output = V::Alloc>;
    where fn
        Bnext: ConjAssign,
        Bnext | 'a: MulAssign<Self::OutputScalar>;
    (self, rhs) => move |mut grad| {
        grad.conj_assign();
        grad.mul_assign(rhs.e_pow_j());
        self.backward(grad.as_());
    }
}
