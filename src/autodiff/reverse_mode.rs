use as_dyn_trait::as_dyn_trait;
use std::{cell::RefCell, ops::{AddAssign, Add}, rc::Rc, fmt};

pub fn grad<F, T, U, Q, G>(f: F, grad: G) -> Box<dyn Fn(T) -> T::Grad>
where
    F: Fn(RVar<T, ()>) -> RVar<U, Q> + 'static,
    InnerRVar<U, Q>: Accumulate<G>,
    T: GradAlloc + Clone + 'static,
    U: GradAlloc + 'static,
    Q: DispatchTuple + 'static,
    G: 'static,
{
    Box::new(move |value| {
        let input = RVar::trace(value.clone());
        let graph = f(RVar::clone(&input));
        graph.backward(&grad);
        let input = Rc::try_unwrap(input.0).ok().unwrap();
        *input.grad.unwrap().into_inner()
    })
}

pub trait GradAlloc {
    type Grad;
    fn grad_alloc(&self) -> Self::Grad;
}

impl GradAlloc for i32 {
    type Grad = i32;
    fn grad_alloc(&self) -> i32 {
        0
    }
}

pub trait DispatchTuple {
    type Refs;
    fn dispatch(self, refs: Self::Refs);
    fn downcast(refs: &Self::Refs) -> Vec<Rc<dyn BackwardStep>>;
}

impl DispatchTuple for () {
    type Refs = ();
    fn dispatch(self, _: Self::Refs) {}
    fn downcast(_: &Self::Refs) -> Vec<Rc<dyn BackwardStep>> {
        vec![]
    }
}

impl<A: 'static> DispatchTuple for (A,) {
    type Refs = (Rc<dyn Accumulate<A>>,);
    fn dispatch(self, refs: Self::Refs) {
        refs.0.accumulate(&self.0);
    }
    fn downcast(refs: &Self::Refs) -> Vec<Rc<dyn BackwardStep>> {
        vec![Rc::clone(&refs.0).as_dyn_backward_step()]
    }
}

impl<A: 'static, B: 'static> DispatchTuple for (A, B) {
    type Refs = (Rc<dyn Accumulate<A>>, Rc<dyn Accumulate<B>>);
    fn dispatch(self, refs: Self::Refs) {
        refs.0.accumulate(&self.0);
        refs.1.accumulate(&self.1);
    }
    fn downcast(refs: &Self::Refs) -> Vec<Rc<dyn BackwardStep>> {
        vec![
            Rc::clone(&refs.0).as_dyn_backward_step(),
            Rc::clone(&refs.1).as_dyn_backward_step(),
        ]
    }
}

pub struct InnerRVar<T: GradAlloc, P: DispatchTuple> {
    value: RefCell<Option<Box<T>>>,
    grad: Option<RefCell<Box<T::Grad>>>,
    parents: P::Refs,
    grad_fn: Option<Box<dyn Fn(T::Grad, T) -> P>>,
    allow_merge: bool,
}

pub struct RVar<T: GradAlloc, P: DispatchTuple>(Rc<InnerRVar<T, P>>);

impl<T: GradAlloc> RVar<T, ()> {
    pub fn new(value: T) -> Self {
        RVar(Rc::new(InnerRVar {
            value: RefCell::new(Some(Box::new(value))),
            grad: None,
            parents: (),
            grad_fn: None,
            allow_merge: true,
        }))
    }
    fn trace(value: T) -> Self {
        RVar(Rc::new(InnerRVar {
            grad: Some(RefCell::new(Box::new(value.grad_alloc()))),
            value: RefCell::new(Some(Box::new(value))),
            parents: (),
            grad_fn: None,
            allow_merge: true,
        }))
    }
}

impl<T: GradAlloc + 'static, P: DispatchTuple + 'static> RVar<T, P> {
    fn backward<U: 'static>(self, grad: &U)
    where
        InnerRVar<T, P>: Accumulate<U>,
    {
        self.0.accumulate(grad);
        for node in TopologicalIter::new(self.0) {
            node.backward_step();
        }
    }
}

impl<T: GradAlloc, P: DispatchTuple> InnerRVar<T, P> {
    fn op1<F, G>(self: Rc<Self>, op_ref: F, op_move: G) -> Box<T>
    where
        F: Fn(&T) -> T,
        G: Fn(T) -> T,
    {
        if self.allow_merge && Rc::strong_count(&self) == 1 {
            let value = self
                .value
                .borrow_mut()
                .take()
                .expect("Attempting to use the value of a merged variable.");
            Box::new(op_move(*value))
        } else {
            let value = self.value.borrow();
            let value = value
                .as_ref()
                .expect("Attempting to use the value of a merged variable.");
            Box::new(op_ref(&*value))
        }
    }
    fn op2<F, G, U, Q>(self: Rc<Self>, other: Rc<InnerRVar<U, Q>>, op_ref: F, op_move: G) -> Box<T>
    where
        F: Fn(&T, &U) -> T,
        G: Fn(T, &U) -> T,
        U: GradAlloc,
        Q: DispatchTuple,
    {
        let other_value = other.value.borrow();
        let other_value = other_value
            .as_ref()
            .expect("Attempting to use the value of a merged variable.");
        self.op1(
            |v| op_ref(v, &*other_value),
            |v| op_move(v, &*other_value),
        )
    }
}

impl<T: GradAlloc, P: DispatchTuple> fmt::Debug for InnerRVar<T, P>
where
    T: fmt::Debug,
    T::Grad: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerRVar")
            .field("value", &self.value)
            .field("grad", &self.grad)
            .field("parents", &"(...dyn Accumulate<_>)")
            .field("grad_fn", &"dyn Fn(_, _) -> (..._)")
            .field("allow_merge", &self.allow_merge)
            .finish()
    }
}

impl<T: GradAlloc, P: DispatchTuple> Clone for RVar<T, P> {
    fn clone(&self) -> Self {
        RVar(Rc::clone(&self.0))
    }
}

pub trait Accumulate<T>: BackwardStep {
    fn accumulate(&self, value: &T);
}

impl<T: GradAlloc, P: DispatchTuple, U> Accumulate<U> for InnerRVar<T, P>
where
    T::Grad: for<'a> AddAssign<&'a U>,
{
    fn accumulate(&self, value: &U) {
        match &self.grad {
            Some(b) => **b.borrow_mut() += value,
            None => (),
        }
    }
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
}

impl<T: GradAlloc, P: DispatchTuple> BackwardStep for InnerRVar<T, P> {
    fn backward_step(self: Rc<Self>) {
        let var = Rc::try_unwrap(self).ok().expect(
            "Attempting to take a backward step on a variable that is referenced elsewhere.",
        );
        match (var.grad_fn, var.grad) {
            (Some(f), Some(g)) => f(
                *g.into_inner(),
                *var.value
                    .into_inner()
                    .expect("Attempting to use the value of a merged variable."),
            )
            .dispatch(var.parents),
            _ => (),
        }
    }
    fn parents(&self) -> Vec<Rc<dyn BackwardStep>> {
        P::downcast(&self.parents)
    }
}

struct TopologicalIter {
    stack: Vec<Rc<dyn BackwardStep>>,
}

impl TopologicalIter {
    fn new(graph: Rc<dyn BackwardStep>) -> Self {
        TopologicalIter { stack: vec![graph] }
    }
}

impl Iterator for TopologicalIter {
    type Item = Rc<dyn BackwardStep>;
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.stack.pop()?;
        for p in item.parents() {
            if Rc::strong_count(&p) == 1 {
                self.stack.push(p);
            }
        }
        Some(item)
    }
}

impl<T0, T1, P0, P1> Add<RVar<T1, P1>> for RVar<T0, P0>
where
    for<'a> &'a T0: Add<&'a T1, Output = T0>,
    T0: for<'a> Add<&'a T1, Output = T0> + GradAlloc + 'static,
    T0::Grad: for<'a> AddAssign<&'a T0::Grad> + Clone + 'static,
    T1::Grad: for<'a> AddAssign<&'a T0::Grad>,
    T1: GradAlloc + 'static,
    P0: DispatchTuple + 'static,
    P1: DispatchTuple + 'static,
{
    type Output = RVar<T0, (T0::Grad, T0::Grad)>;
    fn add(self, other: RVar<T1, P1>) -> Self::Output {
        let value = Rc::clone(&self.0).op2(Rc::clone(&other.0), |x, y| x + y, |x, y| x + y);
        RVar(Rc::new(InnerRVar {
            grad: Some(RefCell::new(Box::new(T0::grad_alloc(&*value)))),
            value: RefCell::new(Some(value)),
            parents: (self.0, other.0),
            grad_fn: Some(Box::new(|grad, _| (grad.clone(), grad))),
            allow_merge: true,
        }))
    }    
}
