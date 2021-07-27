use super::*;
use crate::stack_buffer::StackBuffer;
use std::ops::Add;
use typenum::bit::B1;
use typenum::{UInt, UTerm};

pub trait StaticAxes: Axes {
    fn _runtime_inner(ax: &mut [usize], idx: usize);
    fn runtime() -> <Self::Len as StackBuffer<[usize; 1]>>::Buffer;
}

impl<N, As> StaticAxes for Ax<StatAx<N>, As>
where
    N: Unsigned,
    As: StaticAxes,
    As::Len: Add<UInt<UTerm, B1>>,
    <As::Len as Add<UInt<UTerm, B1>>>::Output: Unsigned + StackBuffer<[usize; 1]>,
    As::Elem: Mul<N>,
    <As::Elem as Mul<N>>::Output: Unsigned,
{
    #[inline]
    fn _runtime_inner(ax: &mut [usize], idx: usize) {
        ax[idx] = N::USIZE;
        As::_runtime_inner(ax, idx + 1);
    }
    fn runtime() -> <Self::Len as StackBuffer<[usize; 1]>>::Buffer {
        let mut ax = <Self::Len as StackBuffer<[usize; 1]>>::Buffer::default();
        Self::_runtime_inner(ax.as_mut(), 0);
        ax
    }
}

impl StaticAxes for Ax0 {
    #[inline]
    fn _runtime_inner(_: &mut [usize], _: usize) {}
    fn runtime() -> <Self::Len as StackBuffer<[usize; 1]>>::Buffer {
        []
    }
}
