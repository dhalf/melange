#[cfg(test)]
mod tests {
    use super::prelude::*;
    use melange_macros::{ax, rvar};
    #[test]
    fn it_works() {
        let g = grad(|a: RVar<f64>| a + RVar::new(1.0), 1.0);
        assert_eq!(1.0, g(2.0));
    }
    #[test]
    fn c2() {
        fn f(a: RVar<RVar<f64>>) -> RVar<RVar<f64>> {
            let b: RVar<RVar<f64>> = RVar::new(RVar::new(1.0));
            a + b
        }
        let g = grad(grad(f, RVar::new(1.0)), 1.0);
        assert_eq!(0.0, g(2.0));
    }
    #[test]
    fn linear() {
        #[rvar]
        #[derive(PartialEq)]
        struct A {
            a: f64,
            b: f64,
        }

        fn linear(param: RVar<A>, input: f64) -> RVar<f64> {
            let param_pat = A::destructure(param);
            param_pat.b + param_pat.a * RVar::new(input)
        }
        let g = grad(|p| linear(p, 5.0), 1.0);
        assert_eq!(g(A { a: 2.0, b: 0.0 }), A { a: 5.0, b: 1.0 });
    }
    #[test]
    fn linear_tensor() {
        use std::convert::TryFrom;

        #[rvar]
        #[derive(PartialEq)]
        struct Param {
            a: Tensor<VecConstructor, f64, ax!(3), Contiguous>,
            b: f64,
        }

        fn linear(
            param: RVar<Param>,
            input: Tensor<VecConstructor, f64, ax!(3), Contiguous>,
        ) -> RVar<Tensor<VecConstructor, f64, ax!(3), Contiguous>> {
            let ParamDestructured { a, b } = Param::destructure(param);
            b + a * RVar::new(input) //a.vvdot(&RVar::new(input))
        }
        let g = grad(
            |p| linear(p, Tensor::try_from(vec![1.0, -10.0, 3.0]).unwrap()),
            Tensor::fill(1.0),
        );
        assert_eq!(
            g(Param {
                a: Tensor::fill(1.0),
                b: 5.0
            }),
            Param {
                a: Tensor::try_from(vec![1.0, -10.0, 3.0]).unwrap(),
                b: 3.0
            }
        );
    }
}

pub mod autodiff;
pub mod axes;
pub mod hkt;
pub mod iter;
pub mod ops;
pub mod scalar_traits;
pub mod stack_buffer;
pub mod tensor;

pub mod prelude {
    pub use crate::autodiff::reverse_mode::{
        destructure::Destructure,
        differentiable::Differentiable,
        grad,
        rvar::{Grad, RVar},
    };
    pub use crate::axes::{Ax, Ax0, DynAx, StatAx};
    pub use crate::hkt::*;
    pub use crate::stack_buffer::StackBuffer;
    pub use crate::tensor::linalg::{Contiguous, Strided, Transposed};
    pub use crate::tensor::Tensor;
    pub use typenum::bit::{B0, B1};
    pub use typenum::{UInt, UTerm};
}
