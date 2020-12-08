//! Defines Variable, a wrapper around tensors
//! that can be used for backpropagating.

use crate::tensor::prelude::*;
use std::cell::RefCell;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;
use crate::algebra::Ring;
use crate::tensor::layout::{StaticLayout, DynamicLayout};

/// Create a new variable that retains its gradient if require_grad is true
/// by moving the given tensor.
pub trait New<T> {
    /// Wraps `tensor` inside a variable that retains its grad
    /// if `require_grad` is true.
    fn new(tensor: T, require_grad: bool) -> Self;
}

/// Groups the value, "interior-mutable" gradient option
/// and backpropagation closure as a sigle entity.
/// 
/// Fields are public in the parent module (`backprop`).
pub struct InternalVariable<V, G, B> {
    pub(super) value: V,
    pub(super) grad: RefCell<Option<G>>,
    pub(super) backward_op_name: &'static str,
    pub(super) backward_closure: Box<dyn Fn(B) -> ()>,
}

impl<V, G, B> fmt::Debug for InternalVariable<V, G, B>
where
    V: fmt::Debug,
    G: fmt::Debug,
    B: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("value", &self.value)
            .field("grad", & *self.grad.borrow())
            .field("backward_op", &self.backward_op_name)
            .finish()
    }
}

/// Core type of `backprop` module that represents a node in the computation
/// graph. It contains a combination of `Rc` and `RefCell` to allow
/// mutable reference counting of the actual `BackpropNode`s.
pub struct Variable<V, G, B>(
    pub(super) Rc<InternalVariable<V, G, B>>,
);

impl<V, G, B> fmt::Debug for Variable<V, G, B>
where
    V: fmt::Debug,
    G: fmt::Debug,
    B: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<V, G, B> Deref for Variable<V, G, B>
{
    type Target = Rc<InternalVariable<V, G, B>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V, G, B> Variable<V, G, B>
where
    G: Clone,
{
    /// Returns an option to a copy of the gradient.
    pub fn grad(&self) -> Option<G> {
        self.grad.borrow().clone()
    }
}

impl<V, G, B> Variable<V, G, B>
where
    G: for<'a> Add_<&'a B>,
{
    /// Add given gradient to the retained gradient if needed and
    /// backpropagate using the closure.
    ///
    /// To initiate backpropagation a tensor full of ones is a good choice.
    pub fn backward(&self, grad: B) {
        let mut current_grad = self.grad.borrow_mut();
        if let Some(g) = &mut *current_grad {
            g.add_(&grad);
        }
        (self.backward_closure)(grad);
    }
}

impl<Y, Z, T, S, D, L, B> New<Tensor<Static, Y, Z, T, S, D, L>>
    for Variable<
        Tensor<Static, Y, Z, T, S, D, L>,
        Tensor<Static, Contiguous, Normal, T, S, Vec<T>, StaticLayout<S>>,
        B,
    >
where
    T: Ring + Copy,
    S: StaticShape,
{
    fn new(tensor: Tensor<Static, Y, Z, T, S, D, L>, require_grad: bool) -> Self {
        Variable(Rc::new(InternalVariable {
            value: tensor,
            grad: RefCell::new(if require_grad {
                Some(Tensor::fill(T::ZERO))
            } else {
                None
            }),
            backward_op_name: "no_op",
            backward_closure: Box::new(|_grad| ()),
        }))
    }
}

impl<Y, Z, T, S, D, L, B> New<Tensor<Dynamic, Y, Z, T, S, D, L>>
    for Variable<
        Tensor<Dynamic, Y, Z, T, S, D, L>,
        Tensor<Dynamic, Contiguous, Normal, T, S, Vec<T>, DynamicLayout<S::Len>>,
        B,
    >
where
    T: Ring + Copy,
    S: Shape,
    L: Layout<S::Len>,
{
    fn new(tensor: Tensor<Dynamic, Y, Z, T, S, D, L>, require_grad: bool) -> Self {
        let shape = tensor.shape();
        Variable(Rc::new(InternalVariable {
            value: tensor,
            grad: RefCell::new(if require_grad {
                Some(Tensor::fill_dynamic(shape, T::ZERO))
            } else {
                None
            }),
            backward_op_name: "no_op",
            backward_closure: Box::new(|_grad| ()),
        }))
    }
}

impl<V, G, B> Clone for Variable<V, G, B>
{
    fn clone(&self) -> Self {
        Variable(Rc::clone(&self.0))
    }
}
