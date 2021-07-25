use super::*;
use std::ops::BitOr;
use typenum::{IsEqual, UInt, UTerm};
use typenum::bit::B1;

pub trait Broadcast<Z> {}

impl<N, As, M, As2> Broadcast<Ax<StatAx<M>, As2>> for Ax<StatAx<N>, As>
where
    N: IsEqual<M> + IsEqual<UInt<UTerm, B1>>,
    <N as IsEqual<M>>::Output: BitOr<<N as IsEqual<UInt<UTerm, B1>>>::Output>,
    <<N as IsEqual<M>>::Output as BitOr<<N as IsEqual<UInt<UTerm, B1>>>::Output>>::Output: True,
    As: Broadcast<As2>,
{}

impl<D, As, As2> Broadcast<Ax<DynAx<D>, As2>> for Ax<StatAx<UInt<UTerm, B1>>, As>
where
    As: Broadcast<As2>,
{}

impl<D, As, As2> Broadcast<Ax<DynAx<D>, As2>> for Ax<DynAx<D>, As>
where
    As: Broadcast<As2>,
{}

impl<As> Broadcast<As> for Ax0 {}
