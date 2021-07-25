use super::*;
use std::ops::Add;
use typenum::IsLessOrEqual;

pub trait Block<O, Z> {}

impl<N, As, M, As2, O, As3> Block<Ax<StatAx<M>, As2>, Ax<StatAx<O>, As3>> for Ax<StatAx<N>, As>
where
    M: Add<O>,
    <M as Add<O>>::Output: IsLessOrEqual<N>,
    <<M as Add<O>>::Output as IsLessOrEqual<N>>::Output: True,
    As: Block<As2, As3>,
{}

impl<N, As, M, As2, D3, As3> Block<Ax<StatAx<M>, As2>, Ax<DynAx<D3>, As3>> for Ax<StatAx<N>, As>
where
    As: Block<As2, As3>,
{}

impl<N, As, D2, As2, A3, As3> Block<Ax<DynAx<D2>, As2>, Ax<A3, As3>> for Ax<StatAx<N>, As>
where
    As: Block<As2, As3>,
{}

impl<D, As, A2, As2, A3, As3> Block<Ax<A2, As2>, Ax<A3, As3>> for Ax<DynAx<D>, As>
where
    As: Block<As2, As3>,
{}

impl Block<Ax0, Ax0> for Ax0 {}
