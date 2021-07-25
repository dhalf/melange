use super::*;
use std::ops::{Div, Rem, Add};
use typenum::{IsGreater, UInt, UTerm};
use typenum::bit::{B0, B1};

#[allow(non_camel_case_types)]
pub struct __stride_op<A, B>(PhantomData<(*const A, *const B)>);

pub trait If<T, E> {
    type Output;
}

impl<T, E> If<T, E> for B0 {
    type Output = E;
}

impl<T, E> If<T, E> for B1 {
    type Output = T;
}

pub trait Stride<Z> {
    type Output;
}

impl<N, As, M, As2> Stride<Ax<StatAx<M>, As2>> for Ax<StatAx<N>, As>
where
    N: Div<M> + Rem<M>,
    <N as Rem<M>>::Output: IsGreater<UInt<UTerm, B0>>,
    <<N as Rem<M>>::Output as IsGreater<UInt<UTerm, B0>>>::Output: If<UInt<UTerm, B1>, UInt<UTerm, B0>>,
    <N as Div<M>>::Output: Add<<<<N as Rem<M>>::Output as IsGreater<UInt<UTerm, B0>>>::Output as If<UInt<UTerm, B1>, UInt<UTerm, B0>>>::Output>,
    As: Stride<As2>,
{
    type Output = Ax<StatAx<<<N as Div<M>>::Output as Add<<<<N as Rem<M>>::Output as IsGreater<UInt<UTerm, B0>>>::Output as If<UInt<UTerm, B1>, UInt<UTerm, B0>>>::Output>>::Output>, As::Output>;
}

impl<N, As, D2, As2> Stride<Ax<DynAx<D2>, As2>> for Ax<StatAx<N>, As>
where
    As: Stride<As2>,
{
    type Output = Ax<DynAx<__stride_op<StatAx<N>, DynAx<D2>>>, As::Output>;
}

impl<D, As, A2, As2> Stride<Ax<A2, As2>> for Ax<DynAx<D>, As>
where
    As: Stride<As2>,
{
    type Output = Ax<DynAx<__stride_op<DynAx<D>, A2>>, As::Output>;
}

impl Stride<Ax0> for Ax0 {
    type Output = Ax0;
}
