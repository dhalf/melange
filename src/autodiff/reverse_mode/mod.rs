use std::rc::Rc;
use std::cell::Ref;
use as_dyn_trait::as_dyn_trait;

pub mod differentiable;
use differentiable::Differentiable;

pub mod dispatch_tuple;
pub mod accumulate_on;
pub mod rvar;
pub mod ops;
pub mod destructure;
pub mod topological_iter;

use rvar::RVar;

pub fn grad<F, T, U>(f: F, grad: U::RGrad) -> Box<dyn FnOnce(T) -> T>
where
    F: FnOnce(RVar<T>) -> RVar<U> + 'static,
    T: Differentiable,
    U: Differentiable,
{
    Box::new(move |value| {
        let input = RVar::trace(value);
        let graph = f(RVar::clone(&input));
        graph.backward(grad);
        input.0.grad()
    })
}

pub trait ReverseMode<T: Differentiable>: BackwardStep {
    fn accumulate(self: Rc<Self>, value: T::RGrad);
    fn take_value(self: Rc<Self>) -> T;
    fn borrow_value<'a>(&'a self) -> Ref<'a, T>;
    fn grad(self: Rc<Self>) -> T;
    fn allow_merge(&self) -> bool;
}

#[as_dyn_trait(
    enable_ref = false,
    enable_box = false,
    enable_arc = false,
    enable_pin = false
)]
pub trait BackwardStep {
    fn backward_step(self: Rc<Self>);
    fn parents(&self) -> Vec<Rc<dyn BackwardStep>>;
    fn retains_grad(&self) -> bool;
}
