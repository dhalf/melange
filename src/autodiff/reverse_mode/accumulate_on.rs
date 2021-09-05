use std::ops::AddAssign;
use crate::hkt::KindTypeTypeType;
use crate::tensor::Tensor;
use crate::axes::Axes;
use crate::scalar_traits::Zero;

pub trait AccumulateOn<T> {
    fn onto(self) -> T;
}

impl<T> AccumulateOn<T> for T {
    #[inline]
    fn onto(self) -> T {
        self
    }
}

impl<B, T, S, C> AccumulateOn<T> for Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem>,
    B::Applied: AsMut<[T]>,
    T: Zero + for<'a> AddAssign<&'a T> + 'static,
    S: Axes,
{
    fn onto(mut self) -> T {
        let mut res = T::ZERO;
        self.for_each(|x| res += x);
        res
    }
}