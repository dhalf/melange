use super::alloc::AllocLike;
use super::Tensor;

pub trait Owned {}

impl<T> Owned for Vec<T> {}

impl<X, Y, Z, T, S, A, D, L> Owned for Tensor<X, Y, Z, T, S, A, D, L> where D: Owned {}

pub trait BinaryOp<Rhs> {
    type Output;
}

impl<X, Y, Z, T, S, A, D, L, Rhs> BinaryOp<Rhs> for Tensor<X, Y, Z, T, S, A, D, L>
where
    Self: AllocLike,
{
    type Output = <Self as AllocLike>::Alloc;
}

impl<T, Rhs> BinaryOp<Rhs> for T
where
    T: Copy,
    Rhs: AllocLike,
{
    type Output = <Rhs as AllocLike>::Alloc;
}
