use super::*;

pub trait DispatchTuple {
    type Refs;
    type RGrads;
    fn dispatch(grads: Self::RGrads, refs: Self::Refs);
    fn downcast(refs: &Self::Refs) -> Vec<Rc<dyn BackwardStep>>;
}

impl DispatchTuple for () {
    type Refs = ();
    type RGrads = ();
    fn dispatch(_: Self::RGrads, _: Self::Refs) {}
    fn downcast(_: &Self::Refs) -> Vec<Rc<dyn BackwardStep>> {
        vec![]
    }
}

impl<A: Differentiable> DispatchTuple for (A,) {
    type Refs = (Rc<dyn ReverseMode<A>>,);
    type RGrads = (A::RGrad,);
    fn dispatch(grads: Self::RGrads, refs: Self::Refs) {
        refs.0.accumulate(grads.0);
    }
    fn downcast(refs: &Self::Refs) -> Vec<Rc<dyn BackwardStep>> {
        vec![Rc::clone(&refs.0).as_dyn_backward_step()]
    }
}

impl<A: Differentiable, B: Differentiable> DispatchTuple for (A, B) {
    type Refs = (Rc<dyn ReverseMode<A>>, Rc<dyn ReverseMode<B>>);
    type RGrads = (A::RGrad, B::RGrad);
    fn dispatch(grads: Self::RGrads, refs: Self::Refs) {
        refs.0.accumulate(grads.0);
        refs.1.accumulate(grads.1);
    }
    fn downcast(refs: &Self::Refs) -> Vec<Rc<dyn BackwardStep>> {
        vec![
            Rc::clone(&refs.0).as_dyn_backward_step(),
            Rc::clone(&refs.1).as_dyn_backward_step(),
        ]
    }
}
