//! Defines Variable, a wrapper around tensors
//! that can be used for backpropagating.

use crate::scalar_traits::{Differentiable, Zero};
use crate::tensor::prelude::*;
use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::ops::Deref;
use std::rc::Rc;

/// Create a new variable that retains its gradient if require_grad is true
/// by moving the given tensor.
pub trait New<V> {
    /// Wraps `tensor` inside a variable that retains its grad
    /// if `require_grad` is true.
    fn new(tensor: V, require_grad: bool) -> Self;
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
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("value", &self.value)
            .field("grad", &*self.grad.borrow())
            .field("backward_op", &self.backward_op_name)
            .finish()
    }
}

/// Core type of `backprop` module that represents a node in the computation
/// graph. It contains a combination of `Rc` and `RefCell` to allow
/// mutable reference counting of the [`InternalVariables`](InternalVariables)s.
pub struct Variable<T, V, G, B>(
    pub(super) Rc<InternalVariable<V, G, B>>,
    pub(super) PhantomData<T>,
);

impl<T, V, G, B> fmt::Debug for Variable<T, V, G, B>
where
    V: fmt::Debug,
    G: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T, V, G, B> Deref for Variable<T, V, G, B> {
    type Target = Rc<InternalVariable<V, G, B>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, V, G, B> Variable<T, V, G, B>
where
    G: Clone,
{
    /// Returns an option to a copy of the gradient.
    pub fn grad(&self) -> Option<G> {
        self.grad.borrow().clone()
    }
}

impl<T, V, G, B> Variable<T, V, G, B>
where
    G: for<'a> AddAssign<&'a B>,
{
    /// Add given gradient to the retained gradient if needed and
    /// backpropagate using the closure.
    ///
    /// To initiate backpropagation a tensor full of ones is a good choice.
    pub fn backward(&self, grad: B) {
        let mut current_grad = self.grad.borrow_mut();
        if let Some(g) = &mut *current_grad {
            g.add_assign(&grad);
        }
        (self.backward_closure)(grad);
    }
}

impl<T, V, B> New<V> for Variable<T, V, V::Alloc, B>
where
    T: Differentiable + Zero,
    V: AllocLike<Scalar = T>,
{
    fn new(value: V, require_grad: bool) -> Self {
        let grad = value.fill_like(T::ZERO);
        Variable(
            Rc::new(InternalVariable {
                value: value,
                grad: RefCell::new(if require_grad { Some(grad) } else { None }),
                backward_op_name: "no_op",
                backward_closure: Box::new(|_grad| ()),
            }),
            PhantomData,
        )
    }
}

impl<T, V, G, B> Clone for Variable<T, V, G, B> {
    fn clone(&self) -> Self {
        Variable(Rc::clone(&self.0), PhantomData)
    }
}
