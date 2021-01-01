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

pub trait BroadcastView<Sout> {
    type Output: for<'a> Gat<'a>;
}

use crate::gat::{Gat, RefGat};
impl<'rhs, X, Y, Z, T, S, A, D, L> Gat<'rhs> for Tensor<X, Y, Z, T, S, A, D, L>
where
    D: for<'a> Gat<'a>,
{
    type Output = Tensor<X, Y, Z, T, S, A, <D as Gat<'rhs>>::Output, L>;
}

use crate::tensor::layout::DynamicLayout;
impl<Y, Z, T, S, A, D, L, Sout> BroadcastView<Sout> for Tensor<Static, Y, Z, T, S, A, D, L>
where
    T: 'static,
    Sout: Shape,
{
    type Output = Tensor<Static, Strided, Z, T, Sout, A, RefGat<[T]>, DynamicLayout<Sout::Len>>;
}

macro_rules! cross_domain_fn_impl {
    (
        $wrapper_trait_name:ident;
        $trait_name:ident;
        $fn_name:ident;
        $op_name:expr;
        $(where trait $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        $(where fn $($generic_fn:ty $(| $($lgen_fn:lifetime),+)?: $($bound_fn:path)|*),*;)?
        ($self:ident) => $backward_closure:expr
    ) => {
        /// Wrapper trait that allows an adaptive `B` parameter in the output `Variable`.
        pub trait $wrapper_trait_name<'var, S, Sout> {
            /// Scalar type `T` of the output `Variable`.
            type OutputScalar;
            /// `V` parameter of the output `Variable`.
            type OutputValue: for<'a> Gat<'a>;
            /// `G` parameter of the output `Variable`.
            type OutputGrad;
            /// `B` parameter of the input `Variable`.
            type Back;
            /// Trait function generic over `Bnext`, the adaptive `B` parameter of the output `Variable`.
            fn $fn_name<Bout>(&'var self) -> Variable<'var, Self::OutputScalar, <Self::OutputValue as Gat<'var>>::Output, Self::OutputGrad, Bout>
            where
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        
        impl<'var, T, S, V, G, B, Sout> $wrapper_trait_name<'var, S, Sout> for Variable<'var, T, V, G, B>
        where
            // Forward operation.
            V: $trait_name<'var, Sout, Output = <<V as BroadcastView<Sout>>::Output as Gat<'var>>::Output>,
            V: BroadcastView<Sout>,
            
            // Output variable gradient allocation.
            V: Allocator,
            V::Allocator: StaticAlloc<T, Sout>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: for<'b> AddAssign<&'b B>,

            // 'static bounds required by output backward closure.
            T: 'var,
            V: 'var,
            G: 'var,
            B: 'var,

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = T;
            type OutputValue = <V as BroadcastView<Sout>>::Output;
            type OutputGrad = <V::Allocator as StaticAlloc<T, Sout>>::Alloc;
            type Back = B;
            fn $fn_name<Bout>(&'var $self) -> Variable<'var, Self::OutputScalar, <Self::OutputValue as Gat<'var>>::Output, Self::OutputGrad, Bout>
            where
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                let value = $self.value.$fn_name();
                let grad = {
                    let self_grad = $self.grad.borrow();
                    if let Some(_) = *self_grad {
                        Some(<V::Allocator as StaticAlloc<T, Sout>>::fill(T::ZERO))
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
    VariableBroadcast;
    Broadcast; broadcast; "broadcast_back";
    where fn
        &'b Bout | 'b: Sum<S, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.sum())
    }
}