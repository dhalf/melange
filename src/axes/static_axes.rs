use super::*;
use crate::stack_buffer::StackBuffer;
use std::ops::Add;
use typenum::bit::{B0, B1};
use typenum::{UInt, UTerm};

pub trait StaticAxes {
    type Len: StackBuffer<[usize; 1]>;
    fn _runtime_inner(ax: &mut [usize], idx: usize);
    fn runtime() -> <Self::Len as StackBuffer<[usize; 1]>>::Buffer;
}

impl<N, As> StaticAxes for Ax<StatAx<N>, As>
where
    N: Unsigned,
    As: StaticAxes,
    As::Len: Add<UInt<UTerm, B1>>,
    <As::Len as Add<UInt<UTerm, B1>>>::Output: StackBuffer<[usize; 1]>,
{
    type Len = <As::Len as Add<UInt<UTerm, B1>>>::Output;
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
    type Len = UInt<UTerm, B0>;
    #[inline]
    fn _runtime_inner(_: &mut [usize], _: usize) {}
    fn runtime() -> <Self::Len as StackBuffer<[usize; 1]>>::Buffer {
        []
    }
}
