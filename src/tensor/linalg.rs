extern crate cblas;
extern crate openblas_src;

use cblas::{
    cdotu_sub, cgemm, cgemv, ddot, dgemm, dgemv, sdot, sgemm, sgemv, zdotu_sub, zgemm, zgemv,
    Transpose,
};
use typenum::{UInt, UTerm};
use num_complex::{Complex32, Complex64};
use super::*;
use crate::prelude::*;
use crate::scalar_traits::{Zero, One};
use melange_macros::{ax, buf};

#[derive(Debug)]
pub struct Contiguous;

#[derive(Debug)]
pub struct Transposed;

#[derive(Debug)]
pub struct Strided;

/// Defines the constant that should be passed to BLAS operations.
pub trait BlasTranspose {
    /// Constant to be passed to BLAS.
    const BLAS_TRANSPOSE: Transpose;
}

impl BlasTranspose for Contiguous {
    const BLAS_TRANSPOSE: Transpose = Transpose::None;
}

impl BlasTranspose for Transposed {
    const BLAS_TRANSPOSE: Transpose = Transpose::Ordinary;
}

macro_rules! dots_impl_floats_complex {
    ($($t:ty, $vvdot:ident, $mvdot:ident, $mmdot:ident $({ $complex_vv_res:ident })?);*) => {$(
        impl<B, N, C> Tensor<B, $t, ax!(N), C>
        where
            B: KindTypeTypeType<$t, <ax!(N) as Axes>::Elem>,
            B::Applied: AsRef<[$t]>,
            ax!(N): Axes,
            C: BlasTranspose,
        {
            pub fn vvdot<B2, C2>(&self, other: &Tensor<B2, $t, ax!(N), C2>) -> $t
            where
                B2: KindTypeTypeType<$t, <ax!(N) as Axes>::Elem>,
                B2::Applied: AsRef<[$t]>,
                C2: BlasTranspose,
            {
                assert_eq!(
                    self.size.as_ref()[0], other.size.as_ref()[0],
                    "Contracted dimmensions must be equal, got sizes {:?} and {:?}.",
                    self.size.as_ref(), other.size.as_ref()
                );
                $(let mut $complex_vv_res: [$t; 1] = [<$t>::ZERO];)?
                unsafe { $vvdot(self.size.as_ref()[0] as i32, self.buffer.as_ref(), 1, other.buffer.as_ref(), 1$(, &mut $complex_vv_res)?) }
                $(; $complex_vv_res[0])?
            }
        }

        impl<B, N, K, C> Tensor<B, $t, ax!(N, K), C>
        where
            B: KindTypeTypeType<$t, <ax!(N, K) as Axes>::Elem>,
            B::Applied: AsRef<[$t]>,
            ax!(N, K): Axes,
            C: BlasTranspose,
        {
            pub fn mvdot<B2, C2>(&self, other: &Tensor<B2, $t, ax!(K), C2>) -> Tensor<B::Buffer, $t, ax!(N), Contiguous>
            where
                B: Realloc<$t, <ax!(N) as Axes>::Elem>,
                ax!(N): Axes,
                B2: KindTypeTypeType<$t, <ax!(K) as Axes>::Elem>,
                B2::Applied: AsRef<[$t]>,
                ax!(K): Axes,
                <B::Buffer as KindTypeTypeType<$t, <ax!(N) as Axes>::Elem>>::Applied: AsMut<[$t]>,
                <ax!(N) as Axes>::Len: StackBuffer<[usize; 1], Buffer = Glue<[usize; 0], [usize; 1]>>,
            {
                assert_eq!(
                    self.size.as_ref()[1], other.size.as_ref()[0],
                    "Contracted dimmensions must be equal, got sizes {:?} and {:?}.",
                    self.size.as_ref(), other.size.as_ref()
                );
                let mut res = self.realloc(<$t as Zero>::ZERO, buf![self.size.as_ref()[0]; usize]);

                unsafe {
                    $mvdot(
                        cblas::Layout::ColumnMajor,
                        C::BLAS_TRANSPOSE,
                        self.size.as_ref()[0] as i32,
                        self.size.as_ref()[1] as i32,
                        <$t as One>::ONE,
                        self.buffer.as_ref(),
                        self.size.as_ref()[1] as i32,
                        other.buffer.as_ref(),
                        1,
                        <$t as One>::ONE,
                        res.buffer.as_mut(),
                        1,
                    );
                }
                res
            }
            pub fn mmdot<B2, C2, M>(&self, other: &Tensor<B2, $t, ax!(K, M), C2>) -> Tensor<B::Buffer, $t, ax!(N, M), Contiguous>
            where
                B2: KindTypeTypeType<$t, <ax!(K, M) as Axes>::Elem>,
                B2::Applied: AsRef<[$t]>,
                B: Realloc<$t, <ax!(N, M) as Axes>::Elem>,
                ax!(N, M): Axes,
                ax!(K, M): Axes,
                <B::Buffer as KindTypeTypeType<$t, <ax!(N, M) as Axes>::Elem>>::Applied: AsMut<[$t]>,
                <ax!(N, M) as Axes>::Len: StackBuffer<[usize; 1], Buffer = Glue<[usize; 0], [usize; 2]>>,
                C2: BlasTranspose,
            {
                assert_eq!(
                    self.size.as_ref()[1], other.size.as_ref()[0],
                    "Contracted dimmensions must be equal, got sizes {:?} and {:?}.",
                    self.size.as_ref(), other.size.as_ref()
                );
                let mut res = self.realloc(<$t as Zero>::ZERO, buf![self.size.as_ref()[0], other.size.as_ref()[1]; usize]);

                unsafe {
                    $mmdot(
                        cblas::Layout::ColumnMajor,
                        C::BLAS_TRANSPOSE,
                        C2::BLAS_TRANSPOSE,
                        self.size.as_ref()[0] as i32,
                        other.size.as_ref()[1] as i32,
                        self.size.as_ref()[1] as i32,
                        <$t as One>::ONE,
                        self.buffer.as_ref(),
                        self.size.as_ref()[1] as i32,
                        other.buffer.as_ref(),
                        other.size.as_ref()[1] as i32,
                        <$t as One>::ONE,
                        res.buffer.as_mut(),
                        other.size.as_ref()[1] as i32,
                    );
                }
                res
            }
        }
    )*};
}

dots_impl_floats_complex! {
    f32, sdot, sgemv, sgemm;
    f64, ddot, dgemv, dgemm;
    Complex32, cdotu_sub, cgemv, cgemm { res } ;
    Complex64, zdotu_sub, zgemv, zgemm { res }
}
