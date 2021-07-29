use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use super::*;

macro_rules! binary_inplace_ops_impl {
    ($($trait:ident $fn:ident $op:expr);*$(;)?) => {$(
        impl<B, B2, T, S, C, C2> $trait<Tensor<B2, T, S, C2>> for Tensor<B, T, S, C>
        where
            B: KindTypeTypeType<T, S::Elem>,
            B2: KindTypeTypeType<T, S::Elem>,
            S: Axes,
            T: $trait + Copy + 'static,
            B::Applied: AsMut<[T]>,
            B2::Applied: AsRef<[T]>,
        {
            fn $fn(&mut self, rhs: Tensor<B2, T, S, C2>) {
                self.zip_with_mut(&rhs, $op);
            }
        }
    )*};
}

binary_inplace_ops_impl! {
    AddAssign add_assign |x, &y| *x += y;

}

// impl<B, B2, T, S, C, C2> AddAssign<Tensor<B2, T, S, C2>> for Tensor<B, T, S, C>
//         where
//             B: KindTypeTypeType<T, S::Elem>,
//             B2: KindTypeTypeType<T, S::Elem>,
//             S: Axes,
//             T: AddAssign + Copy + 'static,
//             B::Applied: AsMut<[T]>,
//             B2::Applied: AsRef<[T]>,
//         {
//             fn add_assign(&mut self, rhs: Tensor<B2, T, S, C2>) {
//                 self.zip_with_mut(&rhs, |x, &y| *x += y);
//             }
//         }