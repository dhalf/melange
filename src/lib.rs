//! Melange is a minimal Deep Learning library
//! written in Rust. It features dynamic graphs
//! backpropagation and statically shaped (i.e.
//! shapes checked at compile time) tensors.

#![warn(missing_docs)]
#![allow(type_alias_bounds)]

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use typenum::{U1, U2, U4};
    #[test]
    fn iteration() {
        let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 1, 1, 1]).unwrap();

        for chunk in a.strided_iter(2) {
            assert_eq!(chunk, &[1, 1]);
        }
    }

    #[test]
    fn broadcast() {
        let a: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![1, 2]).unwrap();
        let a = Broadcast::<Shape2D<U2, U2>>::broadcast(&a);
        let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn as_static() {
        let data: Vec<i32> = vec![1, 2, 3, 4];
        let a: DynamicTensor<i32, Shape2D<Dyn, U2>> = Tensor::try_from(data).unwrap();
        let a = AsStatic::<Shape2D<U2, U2>>::as_static(&a);
        let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn stride() {
        let a: StaticTensor<i32, Shape2D<U4, U4>> =
            Tensor::try_from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).unwrap();
        let a = Stride::<Shape2D<U2, U2>>::stride(&a);
        let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 3, 9, 11]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn stride_dyn() {
        let a: DynamicTensor<i32, Shape2D<Dyn, U4>> =
            Tensor::try_from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).unwrap();
        let a = StrideDynamic::<Shape2D<U2, Dyn>>::stride_dynamic(
            &a,
            Index::<U2>::try_from(vec![2, 2]).unwrap(),
        );
        let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 3, 9, 11]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mmdot() {
        use std::f64::consts::FRAC_PI_2;

        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
            FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
            FRAC_PI_2.sin(), FRAC_PI_2.cos()
        ]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
        let c: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![0.0, 1.0]).unwrap();
        assert!(a.dot(&b).sub(&c) < f64::EPSILON);
    }

    #[test]
    fn mvdot() {
        use std::f64::consts::FRAC_PI_2;

        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
            FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
            FRAC_PI_2.sin(), FRAC_PI_2.cos()
        ]).unwrap();
        let b: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
        let c: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![0.0, 1.0]).unwrap();
        assert!(a.dot(&b).sub(&c) < f64::EPSILON);
    }

    #[test]
    fn vvdot() {
        let a: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, -4.0]).unwrap();
        let b: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![6.0, 2.0]).unwrap();
        assert_eq!(a.dot(&b), -2.0);
    }

    #[test]
    fn mmdot_add() {
        use std::f64::consts::FRAC_PI_2;
        
        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
            FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
            FRAC_PI_2.sin(), FRAC_PI_2.cos()
        ]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
        let c: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 1.0]).unwrap();
        let d: StaticTensor<f64, Shape2D<U2, U1>> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
        assert!(a.dot_add(&b, &c).sub(&d) < f64::EPSILON);
    }

    #[test]
    fn mvdot_add() {
        use std::f64::consts::FRAC_PI_2;
        
        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![
            FRAC_PI_2.cos(), -FRAC_PI_2.sin(),
            FRAC_PI_2.sin(), FRAC_PI_2.cos()
        ]).unwrap();
        let b: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 0.0]).unwrap();
        let c: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 1.0]).unwrap();
        let d: StaticTensor<f64, Shape1D<U2>> = Tensor::try_from(vec![1.0, 2.0]).unwrap();
        assert!(a.dot_add(&b, &c).sub(&d) < f64::EPSILON);
    }

    #[test]
    fn backprop() {
        use crate::backprop::variable::Variable;
        use crate::backprop::variable::New;

        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        let g: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        let a = Variable::new(a, true);
        let b = Variable::new(b, true);

        let c = Variable::clone(&a) + b;
        c.backward(g);

        let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(a.grad().unwrap(), d);
    }
}

pub mod gat;
pub mod prelude;
pub mod tensor;
pub mod backprop;
pub mod algebra;
