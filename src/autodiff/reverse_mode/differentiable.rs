use num_complex::{Complex32, Complex64};
use std::ops::AddAssign;
use crate::tensor::{Tensor, linalg::Contiguous, alloc::Alloc};
use crate::hkt::KindTypeTypeType;
use crate::axes::Axes;
use crate::scalar_traits::Zero;
use crate::stack_buffer::StackBuffer;

pub trait Differentiable: Clone + 'static {
    type RGrad: AddAssign + Clone + 'static;
    type AllocData: Clone + 'static;
    fn rgrad_to_grad(rgrad: Self::RGrad) -> Self;
    fn grad_to_rgrad(self) -> Self::RGrad;
    fn alloc_data(&self) -> Self::AllocData;
    fn zero(data: Self::AllocData) -> Self;
    fn zero_rgrad(data: Self::AllocData) -> Self::RGrad;
}

macro_rules! differentiable_impl_scalars {
    ($($t:ty)*) => {$(
        impl Differentiable for $t {
            type RGrad = $t;
            type AllocData = ();
            #[inline]
            fn rgrad_to_grad(rgrad: Self::RGrad) -> Self {
                rgrad
            }
            #[inline]
            fn grad_to_rgrad(self) -> Self::RGrad {
                self
            }
            #[inline]
            fn alloc_data(&self) {}
            #[inline]
            fn zero(_: ()) -> Self {
                <$t>::ZERO
            }
            #[inline]
            fn zero_rgrad(_: ()) -> Self {
                <$t>::ZERO
            }
        }
    )*};
}

differentiable_impl_scalars! { f32 f64 Complex32 Complex64 }

impl<B, T, S> Differentiable for Tensor<B, T, S, Contiguous>
where
    B: KindTypeTypeType<T, S::Elem> + KindTypeTypeType<T::RGrad, S::Elem> + Alloc<T, S::Elem>,
    T: Differentiable<RGrad = T> + Zero,
    S: Axes,
    Self: Clone + 'static,
    Tensor<B, T, S, Contiguous>: AddAssign + Clone + 'static,
{
    type RGrad = Tensor<B, T, S, Contiguous>;
    type AllocData = <S::Len as StackBuffer<[usize; 1]>>::Buffer;
    #[inline]
    fn rgrad_to_grad(rgrad: Self::RGrad) -> Self {
        rgrad
    }
    #[inline]
    fn grad_to_rgrad(self) -> Self::RGrad {
        self
    }
    #[inline]
    fn alloc_data(&self) -> Self::AllocData {
        self.size()
    }
    #[inline]
    fn zero(size: Self::AllocData) -> Self {
        Tensor::alloc(T::ZERO, size)
    }
    #[inline]
    fn zero_rgrad(size: Self::AllocData) -> Self::RGrad {
        Tensor::alloc(T::ZERO, size)
    }
}
