//! Contains backpropagation tools.
//!
//! This module provides the `Variable` type that
//! wraps tensors to enable backpropagation.
//!
//! Varibles contain a tensor value and an optional
//! tensor gradient. Tensor operations are available
//! for variables. But contrary to tensors two passes must
//! be considered:
//! * the forward pass that correspond
//! to the actual tensor operation performed on
//! the value tensors of the involved variables,
//! * the backward pass that computes the partial
//! gradient of the backpropagation tree's root with
//! respect to the input variable of the operation via the
//! chain rule (see [Backpropagation]).
//!
//! The forward pass is computed immediately when the
//! operation is invoked. However, the backward pass is
//! stored in a closure to be called when the upstream
//! (following node closer to the root) gradient is available.
//!
//! Thus only normal tensor operations on the value parts
//! are performed until the [`backward`](Variable::backward)
//! function is called on a variable, starting a backpropagation from
//! that variable onto all the variables that were used to compute
//! it.
//!
//! In practice, backpropagating means building a computation graphs
//! that "remembers" the dependencies between variables. Melange's
//! philosophy regarding computation graph is similar to Pytorch's.
//! The computation graph is entirely implicit and build on the go as
//! variables are used. The actual implementation relies on three key
//! aspects:
//! * variables data are wrapped in an [`Rc`](std::rc::Rc) which makes
//! them shareable without giving up ownership, this means they can be
//! cloned like `Rc`s (no overhead) and moved like them;
//! * the gradient part is also wrapped in a [`RefCell`](std::cell::refCell)
//! to provide internal mutability;
//! * move closures are used to capture each operation environment,
//! this means a variable holds an `Rc`-like clone of the variables it
//! depends on.
//!
//! [Backpropagation] https://en.wikipedia.org/wiki/Backpropagation

pub mod elementwise_ops;
pub mod prelude;
#[doc(hidden)]
pub mod variable;

#[doc(inline)]
pub use variable::{New, Variable};
