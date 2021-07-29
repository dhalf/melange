#[cfg(test)]
mod tests {
    use typenum::uint::{UInt, UTerm};
    use typenum::bit::{B0, B1};

    use crate::stack_buffer::{Glue, StackBuffer};

    #[test]
    fn it_works() {
        type Arr5 = <UInt<UInt<UInt<UTerm, B1>, B0>, B1> as StackBuffer<[i32; 1]>>::Buffer;
        let mut a: Arr5 = Glue::default();
        let r = a.as_mut();
        for x in r.iter_mut() {
            *x += 1;
        }
        println!("{:?}", r);
        panic!("Stop");
    }
}

pub mod stack_buffer;
pub mod hkt;
pub mod axes;
pub mod iter;
pub mod tensor;
pub mod scalar_traits;
pub mod ops;

pub mod prelude {
    pub use typenum::{UInt, UTerm};
    pub use typenum::bit::{B0, B1};
    pub use crate::axes::{Ax, StatAx, DynAx, Ax0};
    pub use crate::hkt::*;
    pub use crate::tensor::Tensor;
    pub use crate::stack_buffer::StackBuffer;
}
