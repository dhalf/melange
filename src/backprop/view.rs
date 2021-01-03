//! Implements the viewing functions defined
//! by the traits in [`tensor::view`](crate::tensor::view)
//! for `Variables`.

use super::variable::{InternalVariable, Variable};
use crate::scalar_traits::*;
use crate::tensor::prelude::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::*;
use std::rc::Rc;
use crate::tensor::index::Index;

macro_rules! cross_domain_fn_impl {
    (
        $wrapper_trait_name:ident;
        $alloc_trait_name:ident;
        $trait_name:ident$(<$param_type:ty>)?;
        $fn_name:ident;
        $op_name:expr;
        $(where trait $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        $(where fn $($generic_fn:ty $(| $($lgen_fn:lifetime),+)?: $($bound_fn:path)|*),*;)?
        ($self:ident) => $backward_closure:expr
    ) => {
        macro_rules! fill {
            ($param:ident: $var:ty) => {
                A::fill($param, T::ZERO)
            };
            () => {
                A::fill(T::ZERO)
            };
        }

        macro_rules! op {
            ($self_:ident, $fn_name_:ident, $param:ident: $var:ty) => {
                $self_.value.$fn_name_($param.clone())
            };
            ($self_:ident, $fn_name_:ident,) => {
                $self_.value.$fn_name_()
            };
        }
        
        /// Wrapper trait that allows an adaptive `B` parameter in the output `Variable`.
        pub trait $wrapper_trait_name<'var, Sout: Shape> {
            /// Scalar type `T` of the output `Variable`.
            type OutputScalar;
            /// `V` parameter of the output `Variable`.
            type OutputValue;
            /// `G` parameter of the output `Variable`.
            type OutputGrad;
            /// `B` parameter of the input `Variable`.
            type Back;
            /// `S` parameter of the input `Variable`'s value `Tensor`.
            type InputShape;
            /// Trait function generic over `Bnext`, the adaptive `B` parameter of the output `Variable`.
            fn $fn_name<Bout>(&'var self$(, runtime_shape: $param_type)?) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bout>
            where
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        
        impl<'var, T, X, Y, Z, S, A, D, L, G, B, Sout> $wrapper_trait_name<'var, Sout> for Variable<'var, T, Tensor<X, Y, Z, T, S, A, D, L>, G, B>
        where
            // Forward operation.
            Tensor<X, Y, Z, T, S, A, D, L>: $trait_name<'var, Sout>,
            Sout: Shape,
            
            // Output variable gradient allocation.
            A: $alloc_trait_name<T, Sout>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: for<'b> AddAssign<&'b B>,

            // 'static bounds required by output backward closure.
            T: 'var,
            X: 'var,
            Y: 'var,
            Z: 'var,
            S: 'var,
            A: 'var,
            D: 'var,
            L: 'var,
            G: 'var,
            B: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = T;
            type OutputValue = <Tensor<X, Y, Z, T, S, A, D, L> as $trait_name<'var, Sout>>::Output;
            type OutputGrad = A::Alloc;
            type Back = B;
            type InputShape = S;
            fn $fn_name<Bout>(&'var $self$(, runtime_shape: $param_type)?) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bout>
            where
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = op!($self, $fn_name, $(runtime_shape: $param_type)?);
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some(fill!($(runtime_shape: $param_type)?))
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
    VariableBroadcast; StaticAlloc;
    Broadcast; broadcast; "broadcast_back";
    where fn
        &'b Bout | 'b: Sum<Self::InputShape, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.sum())
    }
}

cross_domain_fn_impl! {
    VariableBroadcastDynamic; DynamicAlloc;
    BroadcastDynamic<Index<Sout::Len>>; broadcast_dynamic; "broadcast_dynamic_back";
    where fn
        &'b Bout | 'b: Sum<Self::InputShape, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.sum())
    }
}
