use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, AddAssign};
use super::dispatch_tuple::DispatchTuple;
use super::topological_iter::TopologicalIter;
use super::*;

#[derive(Debug, Clone)]
pub enum Grad<T: Differentiable> {
    Allocated(Box<T::RGrad>),
    NonAllocated(T::AllocData),
}

impl<T: Differentiable> Grad<T> {
    pub fn new(grad: T) -> Self {
        Grad::Allocated(Box::new(T::grad_to_rgrad(grad)))
    }
    pub fn take(self) -> T {
        match self {
            Grad::Allocated(grad) => T::rgrad_to_grad(*grad),
            Grad::NonAllocated(data) => T::zero(data),
        }
    }
}

pub enum GradFn<In, Val, Out> {
    GradOnly(Box<dyn Fn(In) -> Out>),
    GradValue(Box<dyn Fn(In, Val) -> Out>),
    AllowMerge,
    DenyMerge,
}

pub struct InnerRVar<T: Differentiable, P: DispatchTuple> {
    pub(super) value: RefCell<Option<Box<T>>>,
    pub(super) grad: RefCell<Grad<T>>,
    pub(super) parents: P::Refs,
    pub(super) grad_fn: GradFn<T::RGrad, T, P::RGrads>,
    pub(super) retain_grad: bool,
}

pub struct RVar<T: Differentiable>(pub(super) Rc<dyn ReverseMode<T>>);

impl<T: Differentiable, P: DispatchTuple> BackwardStep for InnerRVar<T, P> {
    fn backward_step(self: Rc<Self>) {
        let var = Rc::try_unwrap(self)
            .ok()
            .expect("Cannot take a backward step on a variable that is referenced elsewhere.");
        match (var.grad_fn, var.grad.into_inner()) {
            (GradFn::GradValue(f), Grad::Allocated(g)) => P::dispatch(
                f(
                    *g,
                    *var.value
                        .into_inner()
                        .expect("Cannot use the value of a merged variable."),
                ),
                var.parents,
            ),
            (GradFn::GradOnly(f), Grad::Allocated(g)) => P::dispatch(f(*g), var.parents),
            _ => (),
        }
    }
    fn parents(&self) -> Vec<Rc<dyn BackwardStep>> {
        P::downcast(&self.parents)
    }
    fn retains_grad(&self) -> bool {
        self.retain_grad
    }
}

impl<T: Differentiable, P: DispatchTuple> ReverseMode<T> for InnerRVar<T, P> {
    fn accumulate(self: Rc<Self>, value: T::RGrad) {
        if self.retain_grad {
            match &mut *self.grad.borrow_mut() {
                Grad::Allocated(b) => {
                    **b += value;
                }
                g @ Grad::NonAllocated(_) => *g = Grad::Allocated(Box::new(value)),
            }
        }
    }
    fn take_value(self: Rc<Self>) -> T {
        *self.value.borrow_mut().take().expect("Cannot move out the value of a merged variable.")
    }
    fn borrow_value<'a>(&'a self) -> Ref<'a, T> {
        Ref::map(self.value.borrow(), |x| {
            &**x.as_ref()
                .expect("Cannot borrow the value of a merged variable.")
        })
    }
    fn grad(self: Rc<Self>) -> T {
        (match Rc::try_unwrap(self) {
            Ok(var) => var.grad.into_inner(),
            Err(var) => var.grad.borrow().clone(),
        }).take()
    }
    fn allow_merge(&self) -> bool {
        if let GradFn::GradOnly(_) | GradFn::AllowMerge = &self.grad_fn {
            true
        } else {
            false
        }
    }
}

impl<T: Differentiable> RVar<T> {
    pub fn new(value: T) -> Self {
        RVar(Rc::new(InnerRVar::<T, ()> {
            grad: RefCell::new(Grad::NonAllocated(T::alloc_data(&value))),
            value: RefCell::new(Some(Box::new(value))),
            parents: (),
            grad_fn: GradFn::AllowMerge,
            retain_grad: false,
        }))
    }
    pub fn trace(value: T) -> Self {
        RVar(Rc::new(InnerRVar::<T, ()> {
            grad: RefCell::new(Grad::NonAllocated(T::alloc_data(&value))),
            value: RefCell::new(Some(Box::new(value))),
            parents: (),
            grad_fn: GradFn::DenyMerge,
            retain_grad: true,
        }))
    }
    pub fn new_destructured_field_var<U, F>(value: T, parent: RVar<U>, restructure: F) -> Self
    where
        U: Differentiable,
        F: Fn(T::RGrad) -> U::RGrad + 'static,
    {
        RVar(Rc::new(InnerRVar::<T, (U,)> {
            grad: RefCell::new(Grad::NonAllocated(T::alloc_data(&value))),
            value: RefCell::new(Some(Box::new(value))),
            retain_grad: parent.0.retains_grad(),
            parents: (parent.0,),
            grad_fn: GradFn::GradOnly(Box::new(move |x| (restructure(x),))),
        }))
    }
    pub fn detach(&self) -> Self {
        let value = self.0.borrow_value();
        RVar(Rc::new(InnerRVar::<T, ()> {
            grad: RefCell::new(Grad::NonAllocated(T::alloc_data(&value))),
            value: RefCell::new(Some(Box::new(T::clone(&*value)))),
            parents: (),
            grad_fn: GradFn::AllowMerge,
            retain_grad: self.0.retains_grad(),
        }))
    }
    fn allow_merge(&self) -> bool {
        Rc::strong_count(&self.0) <= 2 && self.0.allow_merge()
    }
    fn op1<U, F: Fn(&T) -> U>(self, op_ref: F) -> Box<U> {
        Box::new(op_ref(&*self.0.borrow_value()))
    }
    pub(super) fn op1_merge<U, F, G>(self, op_ref: F, op_move: G) -> Box<U>
    where
        F: Fn(&T) -> U,
        G: Fn(T) -> U,
    {
        if self.allow_merge() {
            Box::new(op_move(self.0.take_value()))
        } else {
            Box::new(op_ref(&*self.0.borrow_value()))
        }
    }
    pub(super) fn op2<U, V, F>(self, other: RVar<U>, op_ref: F) -> Box<V>
    where
        U: Differentiable,
        F: Fn(&T, &U) -> V,
    {
        Box::new(op_ref(&*self.0.borrow_value(), &*other.0.borrow_value()))
    }
    pub(super) fn op2_merge<U, V, F, G, H>(
        self,
        other: RVar<U>,
        op_ref: F,
        op_move_first: G,
        op_move_second: H,
    ) -> Box<V>
    where
        U: Differentiable,
        F: Fn(&T, &U) -> V,
        G: Fn(T, &U) -> V,
        H: Fn(&T, U) -> V,
    {
        if self.allow_merge() {
            Box::new(op_move_first(self.0.take_value(), &*other.0.borrow_value()))
        } else if other.allow_merge() {
            Box::new(op_move_second(&*self.0.borrow_value(), other.0.take_value()))
        } else {
            Box::new(op_ref(&*self.0.borrow_value(), &*other.0.borrow_value()))
        }
    }
}

impl<T: Differentiable> RVar<T> {
    pub(super) fn backward(self, grad: T::RGrad) {
        Rc::clone(&self.0).accumulate(grad);
        for node in TopologicalIter::new(self.0.as_dyn_backward_step()) {
            node.backward_step();
        }
    }
}

impl<T: Differentiable> Clone for RVar<T> {
    fn clone(&self) -> Self {
        RVar(Rc::clone(&self.0))
    }
}

impl<T: Differentiable> fmt::Debug for RVar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RVar")
    }
}

impl<T: Differentiable, P: DispatchTuple> fmt::Debug for InnerRVar<T, P>
where
    T: fmt::Debug,
    T::RGrad: fmt::Debug,
    T::AllocData: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerRVar")
            .field("value", &self.value)
            .field("grad", &self.grad)
            .field("parents", &"(...dyn Accumulate<_>)")
            .field("grad_fn", &"dyn Fn(_, _) -> (..._)")
            .field("retain_grad", &self.retain_grad)
            .finish()
    }
}

impl<T: Differentiable> Differentiable for RVar<T>
where
    T::RGrad: Differentiable<RGrad = T::RGrad>,
    RVar<T::RGrad>: Add<Output = RVar<T::RGrad>>,
{
    type RGrad = RVar<T::RGrad>;
    type AllocData = T::AllocData;
    fn rgrad_to_grad(rgrad: Self::RGrad) -> Self {
        let value = RVar::clone(&rgrad).op1_merge(|x| T::rgrad_to_grad(x.clone()), |x| T::rgrad_to_grad(x));
        RVar(Rc::new(InnerRVar::<T, (T::RGrad,)> {
            grad: RefCell::new(Grad::NonAllocated(T::alloc_data(&value))),
            value: RefCell::new(Some(value)),
            retain_grad: rgrad.0.retains_grad(),
            parents: (rgrad.0,),
            grad_fn: GradFn::GradOnly(Box::new(move |x| (x,))),
        }))
    }
    fn grad_to_rgrad(self) -> Self::RGrad {
        let value = RVar::clone(&self).op1_merge(|x| T::grad_to_rgrad(x.clone()), |x| T::grad_to_rgrad(x));
        RVar(Rc::new(InnerRVar::<T::RGrad, (T,)> {
            grad: RefCell::new(Grad::NonAllocated(<T::RGrad as Differentiable>::alloc_data(&value))),
            value: RefCell::new(Some(value)),
            retain_grad: self.0.retains_grad(),
            parents: (self.0,),
            grad_fn: GradFn::GradOnly(Box::new(move |x| (x,))),
        }))
    }
    fn alloc_data(&self) -> T::AllocData {
        T::alloc_data(&*self.0.borrow_value())
    }
    fn zero(data: T::AllocData) -> Self {
        RVar::new(T::zero(data))
    }
    fn zero_rgrad(data: T::AllocData) -> Self::RGrad {
        RVar::new(T::zero_rgrad(data))
    }
}

impl<T> AddAssign for RVar<T>
where
    T: Differentiable,
    Self: Add<Output = Self>,
{
    fn add_assign(&mut self, rhs: Self) {
        let res = RVar::clone(self) + rhs;
        self.0 = res.0;
    }
}
