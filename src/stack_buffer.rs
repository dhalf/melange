use typenum::uint::{UInt, UTerm};
use typenum::bit::{B0, B1};

pub trait Scalar {
    type Scalar;
}

impl<T, const N: usize> Scalar for [T; N] {
    type Scalar = T;
}

pub trait PowerOfTwo {
    type Next: Scalar;
    type Null: Scalar;
}

macro_rules! power_of_two_impl_arr_T {
    ($n:literal $m:literal $($tail:tt)*) => {
        impl<T> PowerOfTwo for [T; $n] {
            type Next = [T; $m];
            type Null = [T; 0];
        }
        power_of_two_impl_arr_T! { $m $($tail)* }
    };
    ($last:literal) => {};
}

power_of_two_impl_arr_T! {
    1
    2
    4
    8
    16
    32
    64
    128
    256
    512
    1024
    2048
    4096
    8192
    16384
    32768
    65536
    131072
    262144
    524288
    1048576
    2097152
    4194304
    8388608
    16777216
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Glue<L, R> {
    left: L,
    right: R,
}

pub trait SliceTransmutable<T> {
    const LEN: usize;
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<L, T, const N: usize> SliceTransmutable<T> for Glue<L, [T; N]>
where
    L: SliceTransmutable<T>,
{
    const LEN: usize = N + L::LEN;
    fn as_ptr(&self) -> *const T {
        self.left.as_ptr()
    }
    fn as_mut_ptr(&mut self) -> *mut T {
        self.left.as_mut_ptr()
    }
}

impl<T, const N: usize> SliceTransmutable<T> for [T; N] {
    const LEN: usize = N;
    fn as_ptr(&self) -> *const T {
        self.as_ref().as_ptr()
    }
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut().as_mut_ptr()
    }
}

impl<L, R, T> AsRef<[T]> for Glue<L, R>
where
    Self: SliceTransmutable<T>,
{
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), Self::LEN) }
    }
}

impl<L, R, T> AsMut<[T]> for Glue<L, R>
where
    Self: SliceTransmutable<T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), Self::LEN) }
    }
}

pub trait StackBuffer<T>
where
    T: Scalar,
{
    type Buffer: AsRef<[T::Scalar]> + AsMut<[T::Scalar]> + Default;
}

impl<T, U> StackBuffer<T> for UInt<U, B1>
where
    T: Scalar + PowerOfTwo + Default,
    U: StackBuffer<T::Next>,
    Glue<<U as StackBuffer<T::Next>>::Buffer, T>: SliceTransmutable<T::Scalar>,
{
    type Buffer = Glue<<U as StackBuffer<T::Next>>::Buffer, T>;
}

impl<T, U> StackBuffer<T> for UInt<U, B0>
where
    T: Scalar + PowerOfTwo,
    U: StackBuffer<T::Next>,
    <U as StackBuffer<T::Next>>::Buffer: SliceTransmutable<T::Scalar> + AsRef<[T::Scalar]> + AsMut<[T::Scalar]>,
{
    type Buffer = <U as StackBuffer<T::Next>>::Buffer;
}

impl<T> StackBuffer<T> for UTerm
where
    T: Scalar + PowerOfTwo,
    T::Null: AsRef<[T::Scalar]> + AsMut<[T::Scalar]> + Default,
{
    type Buffer = T::Null;
}
