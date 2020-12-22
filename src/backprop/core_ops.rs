use crate::tensor::prelude::*;
use super::variable::{Variable, InternalVariable};
use std::rc::Rc;
use std::cell::RefCell;
use crate::algebra::*;
use std::ops::*;
use crate::ops::*;
use std::marker::PhantomData;

macro_rules! ln_2 {
    () => { 0.693147180559945309417232121458176568 };
}

macro_rules! ln_10 {
    () => { 2.30258509299404568401799145468436421 };
}

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
        macro_rules! inner_impl_float {
            ($t:ty) => {
                impl<V, G, B, Vrhs, Grhs> $trait_name<Variable<$t, Vrhs, Grhs, B::Alloc>> for Variable<$t, V, G, B>
                where
                    // Forward operation.
                    for<'a, 'b> &'a V: $trait_name<&'b Vrhs, Output = V::Alloc>,
                    
                    // Output variable gradient allocation.
                    V: AllocLike<Scalar = $t>,

                    // Backward gradient "copy" in output backward closure.
                    B: AllocLike<Scalar = V::Scalar>,

                    // Backward calls on inputs in output backward closure.
                    G: for<'a> AddAssign<&'a B>,
                    Grhs: for<'a> AddAssign<&'a B::Alloc>,

                    // 'static bounds required by output backward closure.
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
                    type Output = Variable<$t, V::Alloc, V::Alloc, B>;
                    fn $fn_name($self, $rhs: Variable<$t, Vrhs, Grhs, B::Alloc>) -> Self::Output {
                        let value = $self.value.$fn_name(&$rhs.value);
                        
                        let grad = {
                            let self_grad = $self.grad.borrow();
                            let rhs_grad = $rhs.grad.borrow();
                            if let Some(_) = *self_grad {
                                Some($self.value.fill_like(0.0))
                            } else if let Some(_) = *rhs_grad {
                                Some($self.value.fill_like(0.0))
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
        
        // NOTE: repetitions in nested macros are not supported yet.
        inner_impl_float! { f64 }
        inner_impl_float! { f32 }
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
        &'a B | 'a: Mul<V::Scalar, Output = B::Alloc>,
        V::Scalar: Neg<Output=V::Scalar>;
    (self, rhs) => move |grad| {
        rhs.backward(grad.mul(-V::Scalar::ONE));
        self.backward(grad);
    }
}

binary_op_impl! {
    Mul; mul; "mul_back";
    where
        &'a B | 'a, 'b: Mul<&'b V, Output = B::Alloc>,
        B | 'a: MulAssign<&'a Vrhs>;
    (self, rhs) => move |mut grad| {
        rhs.backward(grad.mul(&self.value));
        grad.mul_assign(&rhs.value);
        self.backward(grad);
    }
}

binary_op_impl! {
    Div; div; "div_back";
    where
        &'a B | 'a, 'b: Mul<&'b V, Output = B::Alloc>,
        B | 'a: DivAssign<&'a Vrhs>;
    (self, rhs) => move |mut grad| {
        rhs.backward(grad.mul(&self.value));

        grad.div_assign(&rhs.value);
        self.backward(grad);
    }
}

binary_op_impl! {
    Atan2; atan2; "atan2_back";
    where
        Vrhs: AllocLike,
        V::Scalar: Neg<Output = V::Scalar>,
        &'a V | 'a: Pow<i32, Output = V::Alloc>,
        &'a Vrhs | 'a: Pow<i32, Output = Vrhs::Alloc>,
        V::Alloc | 'a: AddAssign<&'a Vrhs::Alloc>,
        B::Alloc | 'a: MulAssign<&'a Vrhs>,
        B::Alloc | 'a: MulAssign<&'a V>,
        B::Alloc | 'a: DivAssign<&'a V::Alloc>,
        B::Alloc: MulAssign<V::Scalar>,
        B | 'a: MulAssign<&'a Vrhs>,
        B | 'a: DivAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut div = self.value.pow(2);
        div.add_assign(&rhs.value.pow(2));

        let mut rhs_grad = grad.to_contiguous();
        rhs_grad.mul_assign(&rhs.value);
        rhs_grad.mul_assign(&self.value);
        rhs_grad.div_assign(&div);
        rhs_grad.mul_assign(-V::Scalar::ONE);
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
        rhs.backward(grad.fill_like(V::Scalar::ZERO));

        grad.mul_assign(&self.value.signum().mul(&rhs.value.signum()));
        self.backward(grad);
    }
}

binary_op_impl! {
    Max; max; "max_back";
    where
        &'a V | 'a, 'b: Argmax<&'b Vrhs, Output = V::Alloc>,
        B::Alloc | 'a: MulAssign<&'a V::Alloc>,
        V::Scalar: Neg<Output = V::Scalar>,
        V::Alloc: MulAddAssign<V::Scalar, V::Scalar>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmax(&rhs.value);
        
        let mut rhs_grad = grad.to_contiguous();
        rhs_grad.mul_assign(&mask);
        rhs.backward(rhs_grad);

        mask.mul_add_assign(-V::Scalar::ONE, V::Scalar::ONE);

        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

binary_op_impl! {
    Min; min; "min_back";
    where
        &'a V | 'a, 'b: Argmin<&'b Vrhs, Output = V::Alloc>,
        B::Alloc | 'a: MulAssign<&'a V::Alloc>,
        V::Scalar: Neg<Output = V::Scalar>,
        V::Alloc: MulAddAssign<V::Scalar, V::Scalar>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmin(&rhs.value);
        
        let mut rhs_grad = grad.to_contiguous();
        rhs_grad.mul_assign(&mask);
        rhs.backward(rhs_grad);

        mask.mul_add_assign(-V::Scalar::ONE, V::Scalar::ONE);

        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

macro_rules! scalar_op_impl {
    (
        $trait_name:ident$(<$param_type:ty>)?;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $backward_closure:expr
    ) => {        
        macro_rules! inner_impl_float {
            ($t:ty) => {
                macro_rules! isset_or_default {
                    ($var:ty) => {
                        $var
                    };
                    () => {
                        $t
                    };
                }

                impl<V, G, B> $trait_name<isset_or_default!($($param_type)?)> for Variable<$t, V, G, B>
                where
                    // Forward operation.
                    for<'a> &'a V: $trait_name<isset_or_default!($($param_type)?), Output = V::Alloc>,
                    
                    // Output variable gradient allocation.
                    V: AllocLike<Scalar = $t>,

                    // Backward call on input in output backward closure.
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
                    fn $fn_name($self, $rhs: isset_or_default!($($param_type)?)) -> Self::Output {
                        let value = $self.value.$fn_name($rhs);
                        
                        let grad = {
                            let self_grad = $self.grad.borrow();
                            if let Some(_) = *self_grad {
                                Some($self.value.fill_like(0.0))
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
        
        // NOTE: repetitions in nested macros are not supported yet.
        inner_impl_float! { f64 }
        inner_impl_float! { f32 }
    };
}

scalar_op_impl! {
    Add; add; "add_scalar_back";
    (self, rhs) => move |grad| {
        self.backward(grad);
    }
}

scalar_op_impl! {
    Sub; sub; "sub_scalar_back";
    (self, rhs) => move |grad| {
        self.backward(grad);
    }
}

scalar_op_impl! {
    Div; div; "div_scalar_back";
    where
        B: DivAssign<V::Scalar>;
    (self, rhs) => move |mut grad| {
        grad.div_assign(rhs);
        self.backward(grad);
    }
}

scalar_op_impl! {
    Max; max; "max_scalar_back";
    where
        &'a V | 'a: Argmax<V::Scalar, Output = V::Alloc>,
        V::Alloc: MulAddAssign<V::Scalar, V::Scalar>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmax(rhs);
        mask.mul_add_assign(-1.0, 1.0);
        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

scalar_op_impl! {
    Min; min; "min_scalar_back";
    where
        &'a V | 'a: Argmin<V::Scalar, Output = V::Alloc>,
        V::Alloc: MulAddAssign<V::Scalar, V::Scalar>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        let mut mask = self.value.argmin(rhs);
        mask.mul_add_assign(-1.0, 1.0);
        grad.mul_assign(&mask);
        self.backward(grad);
    }
}

scalar_op_impl! {
    Pow; pow; "pow_back";
    where
        V::Alloc: MulAssign<V::Scalar>,
        B: MulAssign<V::Scalar>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        if rhs == 0.0 {
            grad.mul_assign(0.0);
        } else if rhs != 1.0 {
            let mut part_grad = self.value.pow(rhs - 1.0);
            part_grad.mul_assign(rhs);
            grad.mul_assign(&part_grad);
        }
        self.backward(grad);
    }
}

scalar_op_impl! {
    Pow<i32>; pow; "pow_back";
    where
        V::Alloc: MulAssign<V::Scalar>,
        B: MulAssign<V::Scalar>,
        B | 'a: MulAssign<&'a V::Alloc>;
    (self, rhs) => move |mut grad| {
        if rhs == 0 {
            grad.mul_assign(0.0);
        } else if rhs != 1 {
            let mut part_grad = self.value.pow(rhs - 1);
            part_grad.mul_assign(rhs as V::Scalar);
            grad.mul_assign(&part_grad);
        }
        self.backward(grad);
    }
}

macro_rules! reversed_scalar_op_impl {
    (
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        ($self:ident, $rhs:ident) => $forward:block => $backward_closure:expr
    ) => {
        macro_rules! inner_impl_float {
            ($t:ty) => {
                impl<V, G, B> $trait_name<Variable<$t, V, G, B>> for $t
                where
                    // Forward operation.
                    for<'a> &'a V: $trait_name<$t, Output = V::Alloc>,
                    
                    // Output variable gradient allocation.
                    V: AllocLike<Scalar = $t>,
        
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
                                Some($rhs.value.fill_like(0.0))
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

        inner_impl_float! { f64 }
        inner_impl_float! { f32 }
    };
}

reversed_scalar_op_impl! {
    Add; add; "scalar_add_back";
    (self, rhs) => {
        rhs.value.add(self)
    }
    => move |grad| {
        rhs.backward(grad);
    }
}

reversed_scalar_op_impl! {
    Sub; sub; "scalar_sub_back";
    where
        &'a V | 'a: MulAdd<Self, Self, Output = V::Alloc>;
    (self, rhs) => {
        rhs.value.mul_add(-1.0, 1.0)
    }
    => move |grad| {
        rhs.backward(grad);
    }
}

reversed_scalar_op_impl! {
    Mul; mul; "scalar_mul_back";
    where
        Self: Copy,
        B: MulAssign<Self>;
    (self, rhs) => {
        rhs.value.mul(self)
    }
    => move |mut grad| {
        grad.mul_assign(self);
        rhs.backward(grad);
    }
}

reversed_scalar_op_impl! {
    Div; div; "scalar_div_back";
    where
        &'a V | 'a: Recip<Output = V::Alloc>,
        V::Alloc: MulAssign<Self>;
    (self, rhs) => {
        let mut value = rhs.value.recip();
        value.mul_assign(self);
        value
    }
    => move |grad| {
        rhs.backward(grad);
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
        macro_rules! inner_impl_float {
            ($t:ty) => {
                impl<V, G, B> $trait_name for Variable<$t, V, G, B>
                where
                    // Forward operation.
                    for<'a> &'a V: $trait_name<Output = V::Alloc>,
                    
                    // Output variable gradient allocation.
                    V: AllocLike<Scalar = $t>,

                    // Backward call on input in output backward closure.
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
                    fn $fn_name($self) -> Self::Output {
                        let value = $self.value.$fn_name();
                        
                        let grad = {
                            let self_grad = $self.grad.borrow();
                            if let Some(_) = *self_grad {
                                Some($self.value.fill_like(0.0))
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
        
        // NOTE: repetitions in nested macros are not supported yet.
        inner_impl_float! { f64 }
        inner_impl_float! { f32 }
    };
}

fn_impl! {
    Exp; exp; "exp_back";
    where
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        grad.mul_assign(&self.value.exp());
        self.backward(grad);
    }
}

fn_impl! {
    Exp2; exp2; "exp_back";
    where
        V::Alloc: MulAssign<V::Scalar>,    
        B | 'a: MulAssign<&'a V::Alloc>;
    (self) => move |mut grad| {
        let mut part_grad = self.value.exp2();
        part_grad.mul_assign(ln_2!());
        grad.mul_assign(&part_grad);
        self.backward(grad);
    }
}
