#[cfg(test)]
mod tests {
    use crate::autodiff::reverse_mode::*;
    #[test]
    fn it_works() {
        let g = grad(|a| a + RVar::new(1), 1);
        assert_eq!(1, g(2));
    }
}

pub mod stack_buffer;
pub mod hkt;
pub mod axes;
pub mod iter;
pub mod tensor;
pub mod scalar_traits;
pub mod ops;
pub mod autodiff;

pub mod prelude {
    pub use typenum::{UInt, UTerm};
    pub use typenum::bit::{B0, B1};
    pub use crate::axes::{Ax, StatAx, DynAx, Ax0};
    pub use crate::hkt::*;
    pub use crate::tensor::Tensor;
    pub use crate::stack_buffer::StackBuffer;
}
