use super::*;
use crate::ops::{MaxAssign, MinAssign, Sqrt};
use crate::scalar_traits::{Cast, Infinity, NegInfinity, One, Zero};
use std::ops::{AddAssign, DivAssign, MulAssign, Sub, Mul};

macro_rules! reduction_impl {
    ($fn:ident, $fn_dyn:ident, $op_bound:ident, $op:expr, $value_bound:ident, $value:expr) => {
        pub fn $fn<Z>(&self) -> Tensor<B::Buffer, T, Z, Contiguous>
        where
            T: Copy + $value_bound + $op_bound + 'static,
            Z: StaticAxes,
            Z: Broadcast<S>,
            B: Realloc<T, Z::Elem>,
            B::Applied: AsRef<[T]>,
            <<B as Realloc<T, Z::Elem>>::Buffer as KindTypeTypeType<T, Z::Elem>>::Applied:
                AsMut<[T]>,
        {
            self.$fn_dyn(Z::runtime())
        }
        pub fn $fn_dyn<Z>(
            &self,
            size: <Z::Len as StackBuffer<[usize; 1]>>::Buffer,
        ) -> Tensor<B::Buffer, T, Z, Contiguous>
        where
            T: Copy + $value_bound + $op_bound + 'static,
            Z: Axes,
            Z: Broadcast<S>,
            B: Realloc<T, Z::Elem>,
            B::Applied: AsRef<[T]>,
            <<B as Realloc<T, Z::Elem>>::Buffer as KindTypeTypeType<T, Z::Elem>>::Applied:
                AsMut<[T]>,
        {
            let mut res = Tensor::alloc($value, size);
            let mut view = res.broadcast_dyn_mut::<S>(self.size);
            view.zip_with_mut(self, $op);
            res
        }
    };
}

macro_rules! derived_reduction_impl {
    ($fn:ident, $fn_dyn:ident, $($bound_ty:ty: $bound_path:path $(|+ $bound_path_opt:path)*,)* |$self:ident, $size:ident| $e:expr) => {
        pub fn $fn<Z>(&self) -> Tensor<B::Buffer, T, Z, Contiguous>
        where
            T: Copy + 'static,
            Z: StaticAxes,
            Z: Broadcast<S>,
            B: Realloc<T, Z::Elem>,
            B::Applied: AsRef<[T]>,
            <<B as Realloc<T, Z::Elem>>::Buffer as KindTypeTypeType<T, Z::Elem>>::Applied: AsMut<[T]>,
            $($bound_ty: $bound_path $(+ $bound_path_opt)*),*
        {
            self.$fn_dyn(Z::runtime())
        }
        pub fn $fn_dyn<Z>(&$self, $size: <Z::Len as StackBuffer<[usize; 1]>>::Buffer) -> Tensor<B::Buffer, T, Z, Contiguous>
        where
            T: Copy + 'static,
            Z: Axes,
            Z: Broadcast<S>,
            B: Realloc<T, Z::Elem>,
            B::Applied: AsRef<[T]>,
            <<B as Realloc<T, Z::Elem>>::Buffer as KindTypeTypeType<T, Z::Elem>>::Applied: AsMut<[T]>,
            $($bound_ty: $bound_path $(+ $bound_path_opt)*),*
        {
            $e
        }
    };
}

impl<B, T, S, C> Tensor<B, T, S, C>
where
    B: KindTypeTypeType<T, S::Elem>,
    S: Axes,
{
    reduction_impl! {
        sum, sum_dyn, AddAssign, |x, &y| *x += y, Zero, T::ZERO
    }
    reduction_impl! {
        product, product_dyn, MulAssign, |x, &y| *x *= y, One, T::ONE
    }
    reduction_impl! {
        max, max_dyn, MaxAssign, |x, &y| x.max_assign(y), NegInfinity, T::NEG_INFINITY
    }
    reduction_impl! {
        min, min_dyn, MinAssign, |x, &y| x.min_assign(y), Infinity, T::INFINITY
    }
    derived_reduction_impl! {
        mean, mean_dyn, T: AddAssign |+ DivAssign |+ Zero, usize: Cast<T>, |self, size| {
            let count = self.size
                .as_ref()
                .iter()
                .zip(pad::<_, Z::Len, S::Len>(&size, 1).as_ref().iter())
                .fold(0, |acc, (&x, &y)| acc + if y == 1 { x } else { 0 });
            let mut res = self.sum_dyn(size);
            res.for_each(|x| *x /= count.as_());
            res
        }
    }
    derived_reduction_impl! {
        var, var_dyn, T: AddAssign |+ DivAssign |+ Sub<Output = T> |+ Mul<Output = T> |+ Zero, usize: Cast<T>, <<B as Realloc<T, Z::Elem>>::Buffer as KindTypeTypeType<T, Z::Elem>>::Applied: AsRef<[T]>, |self, size| {
            let count = self.size
                .as_ref()
                .iter()
                .zip(pad::<_, Z::Len, S::Len>(&size, 1).as_ref().iter())
                .fold(0, |acc, (&x, &y)| acc + if y == 1 { x } else { 0 });
            let mut res = Tensor::alloc(T::ZERO, size);
            let mut view = res.broadcast_dyn_mut::<S>(self.size);
            let mean = self.mean_dyn::<Z>(size);
            view.zip2_with_mut(self, &mean.broadcast_dyn(self.size), |x, &y, &z| *x += (y - z) * (y - z));
            res.for_each(|x| *x /= count.as_());
            res
        }
    }
    derived_reduction_impl! {
        std, std_dyn, T: AddAssign |+ DivAssign |+ Sub<Output = T> |+ Mul<Output = T> |+ Sqrt<Output = T> |+ Zero, usize: Cast<T>, <<B as Realloc<T, Z::Elem>>::Buffer as KindTypeTypeType<T, Z::Elem>>::Applied: AsRef<[T]>, |self, size| {
            let mut res = self.var_dyn(size);
            res.for_each(|x| *x = x.sqrt());
            res
        }
    }
}
