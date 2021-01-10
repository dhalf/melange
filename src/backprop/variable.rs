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

/// Groups the value, "interior-mutable" gradient option
/// and backpropagation closure as a sigle entity.
///
/// Fields are public in the parent module (`backprop`).
pub struct InternalVariable<'a, V, G, B> {
    pub(super) value: V,
    pub(super) grad: RefCell<Option<G>>,
    pub(super) backward_op_name: &'static str,
    pub(super) backward_closure: Box<dyn Fn(B) -> () + 'a>,
}

impl<'a, V, G, B> fmt::Debug for InternalVariable<'a, V, G, B>
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
pub struct Variable<'a, T, V, G, B>(
    pub(super) Rc<InternalVariable<'a, V, G, B>>,
    pub(super) PhantomData<T>,
);

/// Type alias for `Variable`s that wrap scalar values.
pub type ScalarVariable<'a, T> = Variable<'a, T, T, T, T>;

impl<'a, T, V, G, B> fmt::Debug for Variable<'a, T, V, G, B>
where
    V: fmt::Debug,
    G: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T, V, G, B> Deref for Variable<'a, T, V, G, B> {
    type Target = Rc<InternalVariable<'a, V, G, B>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T, V, G, B> Variable<'a, T, V, G, B>
where
    G: Clone,
{
    /// Returns an option to a copy of the gradient.
    pub fn grad(&self) -> Option<G> {
        self.grad.borrow().clone()
    }
}

impl<'a, T, V, G, B> Variable<'a, T, V, G, B>
where
    G: BackwardCompatible<B>,
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

impl<'a, T, V, B> From<V> for Variable<'a, T, V, V::Alloc, B>
where
    T: Differentiable + Zero,
    V: AllocLike<Scalar = T>,
{
    fn from(value: V) -> Self {
        let grad = value.fill_like(T::ZERO);
        Variable(
            Rc::new(InternalVariable {
                value: value,
                grad: RefCell::new(Some(grad)),
                backward_op_name: "no_op",
                backward_closure: Box::new(|_grad| ()),
            }),
            PhantomData,
        )
    }
}

impl<'a, T, V, G, B> Clone for Variable<'a, T, V, G, B> {
    fn clone(&self) -> Self {
        Variable(Rc::clone(&self.0), PhantomData)
    }
}

pub unsafe trait BackwardCompatible<Rhs>: for<'a> AddAssign<&'a Rhs> {}

unsafe impl<X, Y, Z, T, S, A, D, L, Xrhs, Yrhs, Zrhs, Srhs, Arhs, Drhs, Lrhs>
    BackwardCompatible<Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>>
    for Tensor<X, Y, Z, T, S, A, D, L>
where
    Self: for<'b> AddAssign<&'b Tensor<Xrhs, Yrhs, Zrhs, T, Srhs, Arhs, Drhs, Lrhs>>,
{
}
unsafe impl<X, Y, Z, T, S, A, D, L> BackwardCompatible<Tensor<X, Y, Z, T, S, A, D, L>> for T where
    T: Copy + for<'a> AddAssign<&'a Tensor<X, Y, Z, T, S, A, D, L>>
{
}
unsafe impl<T> BackwardCompatible<T> for T where T: Copy + for<'a> AddAssign<&'a T> {}
