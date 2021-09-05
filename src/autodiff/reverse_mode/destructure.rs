use super::*;

pub trait Destructure: Sized + Differentiable + Clone {
    type Destructured;
    fn internal_destructure(
        self,
        parent: RVar<Self>,
    ) -> Self::Destructured;
    fn destructure(this: RVar<Self>) -> Self::Destructured {
        let value = *RVar::clone(&this).op1_merge(|x| Self::clone(x), |x| x);
        value.internal_destructure(RVar::clone(&this))
    }
}
