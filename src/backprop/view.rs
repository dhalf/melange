//! Implements the viewing functions defined
//! by the traits in [`tensor::view`](crate::tensor::view)
//! for `Variables`.

use super::variable::{InternalVariable, Variable, BackwardCompatible};
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
        $trait_name:ident$(<$static_shape:ident>)?;
        $fn_name:ident$(<$(let $param:ident:)? $param_type:ty>)?;
        $op_name:expr;
        $(where trait $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*;)?
        $(where 'static $($statics:ty),*;)?
        $(where fn $($generic_fn:ty $(| $($lgen_fn:lifetime),+)?: $($bound_fn:path)|*),*;)?
        ($self:ident) => $backward_closure:expr
    ) => {
        macro_rules! isset_or_default {
            ($t:ty) => { $t };
            () => { S };
        }

        macro_rules! op {
            ($self_:ident, $fn_name_:ident, $param_:ident: $var:ty) => {
                $self_.value.$fn_name_($param_.clone())
            };
            ($self_:ident, $fn_name_:ident,) => {
                $self_.value.$fn_name_()
            };
        }
        
        /// Wrapper trait that allows an adaptive `B` parameter in the output `Variable`.
        pub trait $wrapper_trait_name<'var, $($static_shape: Shape)?> {
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
            fn $fn_name<Bnext>(&'var self$(, param: $param_type)?) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?;
        }
        
        impl<'var, T, X, Y, Z, S, A, D, L, G, B, $($static_shape)?> $wrapper_trait_name<'var, $($static_shape)?> for Variable<'var, T, Tensor<X, Y, Z, T, S, A, D, L>, G, B>
        where
            // Forward operation.
            &'var Tensor<X, Y, Z, T, S, A, D, L>: $trait_name<$($static_shape)?>,
            $($static_shape: Shape,)?
            
            // Output variable gradient allocation.
            A: $alloc_trait_name<T, isset_or_default!($($static_shape)?)>,
            T: Zero,

            // Backward call on input in output backward closure.
            G: BackwardCompatible<B>,

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
            //$($param_type: 'static, Strait::Len: 'static,)?
            $($($statics: 'static),*,)?

            // Additionnal bounds needed by either forward or
            // backward passes
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type OutputScalar = T;
            type OutputValue = <&'var Tensor<X, Y, Z, T, S, A, D, L> as $trait_name<$($static_shape)?>>::Output;
            type OutputGrad = A::Alloc;
            type Back = B;
            type InputShape = S;
            fn $fn_name<Bnext>(&'var $self$(, param: $param_type)?) -> Variable<'var, Self::OutputScalar, Self::OutputValue, Self::OutputGrad, Bnext>
            where
                $($($(for<$($lgen_fn),+>)? $generic_fn: $($bound_fn +)*),*)?
            {
                $($(let $param = param.clone();)?)?
                let value = op!($self, $fn_name, $(param: $param_type)?);

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
    VariableBroadcast; StaticAlloc;
    Broadcast<Sout>; broadcast; "broadcast_back";
    where fn
        &'b Bnext | 'b: Sum<Self::InputShape, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.sum());
    }
}

cross_domain_fn_impl! {
    VariableBroadcastDynamic; DynamicAlloc;
    BroadcastDynamic<Sout>; broadcast_dynamic<Index<Sout::Len>>; "broadcast_dynamic_back";
    where trait
        S: Shape,
        L: Layout<S::Len>;
    where fn
        &'b Bnext | 'b: BroadcastDynamicBack<Self::InputShape, Output = Self::Back>,
        Self::InputShape: Shape;
    (self) => move |grad| {
        self.backward(grad.broadcast_dynamic_back(self.value.shape()));
    }
}

cross_domain_fn_impl! {
    VariableStride; StaticAlloc;
    Stride<Strides>; stride; "stride_back";
    where fn
        &'b Bnext | 'b: StrideBack<Self::InputShape, Strides, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.stride_back());
    }
}

cross_domain_fn_impl! {
    VariableStrideDynamic; DynamicAlloc;
    StrideDynamic<Strides>; stride_dynamic<let runtime_strides: Index<Strides::Len>>; "stride_dynamic_back";
    where trait
        S: Shape,
        L: Layout<S::Len>;
    where 'static
        Index<Strides::Len>, Strides::Len;
    where fn
        &'b Bnext | 'b: StrideDynamicBack<Self::InputShape, Strides, Output = Self::Back>,
        Self::InputShape: Shape;
    (self) => move |grad| {
        self.backward(grad.stride_dynamic_back(self.value.shape(), runtime_strides.clone()));
    }
}

// Missing impl in the dynamic case.
cross_domain_fn_impl! {
    VariableTranspose; StaticAlloc;
    Transpose; transpose; "transpose_back";
    where fn
        Bnext: Transpose<Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.transpose());
    }
}

cross_domain_fn_impl! {
    VariableReshape; StaticAlloc;
    Reshape<Sout>; reshape; "reshape_back";
    where fn
        Bnext: Reshape<Self::InputShape, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.reshape());
    }
}

cross_domain_fn_impl! {
    VariableReshapeDynamic; DynamicAlloc;
    ReshapeDynamic<Sout>; reshape_dynamic<Index<Sout::Len>>; "reshape_dynamic_back";
    where trait
        S: Shape,
        L: Layout<S::Len>;
    where fn
        Bnext: ReshapeDynamic<Self::InputShape, Output = Self::Back>,
        Self::InputShape: Shape;
    (self) => move |grad| {
        self.backward(grad.reshape_dynamic(self.value.shape()));
    }
}

cross_domain_fn_impl! {
    VariableAsStatic; StaticAlloc;
    AsStatic<Sout>; as_static; "as_static_back";
    where fn
        Bnext: AsDynamic<Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.as_dynamic());
    }
}

cross_domain_fn_impl! {
    VariableAsDynamic; DynamicAlloc;
    AsDynamic; as_dynamic; "as_dynamic_back";
    where trait
        S: Shape;
    where fn
        Bnext: AsStatic<Self::InputShape, Output = Self::Back>;
    (self) => move |grad| {
        self.backward(grad.as_static());
    }
}
