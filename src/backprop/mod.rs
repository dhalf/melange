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
//! The backward closure should contain code to compute the gradient of
//! the root with respect to the inputs. This code must be valid for all
//! the scalar types the operation is defined on. This is mainly problematic
//! when the operation is valid for complex numbers.
//! 
//! The classical definition of derivation for functions of a complex variable
//! is inherently limited since it requires the function to be holomorphic.
//! This limitation comes from the fact that the growth rate limit should
//! be the same allong all directions in the complex plane and translate into
//! the Cauchy-Rieman conditions that are a pair of partial derivatives
//! equations. Some usual functions enter in that category like the
//! exponential, logarithm, polynomials amd trigonometric functions.
//! However, all real valued functions of a complex variable that are not
//! constant (so all intreasting functions like norms, argument, etc.) are
//! not holomorphic. We thus need another definition of derivation tailored
//! for optimization.
//! 
//! By convention, ML libraries use the conjugate cogradient which is one
//! of the two Wirtinger derivatives. This is a consequence of the community's
//! objective which can be restricted to optimizing a real valued function
//! of a potentially complex variable (called the loss). Wirtinger derivatives
//! have the great property of being an extension of both real and complex
//! "classical" derivatives.
//! 
//! In the case of holomorphic functions,
//! the conjugate cogradient chain rule simplifies to the product
//! of the output's conjugate cogradient and the conjugate of the operation's
//! cogradient (the other Wirtinger derivative).
//! In practice this translates into adding a conjuagtion operation
//! to the real implementation. Note that conjugation is a no-op for real
//! numbers thus this is consistent with real derivatives.
//! For non holomorphic functions, the full formula should be used.
//! 
//! Please refer to [Pytorch] for a more in depth explanation.
//! [This paper] provides a mathematically correct presentation.
//!
//! [Backpropagation]: https://en.wikipedia.org/wiki/Backpropagation
//! [Pytorch]: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
//! [This paper]: https://arxiv.org/pdf/0906.4835.pdf

pub mod elementwise_ops;
pub mod prelude;
#[doc(hidden)]
pub mod variable;
pub mod view;

#[doc(inline)]
pub use variable::{New, Variable};
