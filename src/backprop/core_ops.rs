use crate::tensor::prelude::*;
use super::variable::{Variable, InternalVariable};
use std::rc::Rc;
use std::cell::RefCell;
use crate::algebra::Ring;

impl<V, G, B, Vrhs, Grhs> Add<Variable<Vrhs, Grhs, B::Alloc>> for Variable<V, G, B>
where
    // Forward operation.
    for<'a, 'b> &'a V: Add<&'b Vrhs, Output = V::Alloc>,
    
    // Output variable gradient allocation.
    V: AllocLike,
    V::Scalar: Ring,

    // Backward gradient "copy" in output backward closure.
    B: AllocLike,

    // Backward calls on inputs in output backward closure.
    G: for<'a> Add_<&'a B>,
    Grhs: for<'a> Add_<&'a B::Alloc>,

    // 'static bounds required by output backward closure.
    V: 'static,
    G: 'static,
    B: 'static,
    Vrhs: 'static,
    Grhs: 'static,
    B::Alloc: 'static,
{
    type Output = Variable<V::Alloc, V::Alloc, B>;
    fn add(self, rhs: Variable<Vrhs, Grhs, B::Alloc>) -> Self::Output {
        let value = self.value.add(&rhs.value);
        
        let grad = {
            let self_grad = self.grad.borrow();
            let rhs_grad = rhs.grad.borrow();
            if let Some(_) = *self_grad {
                Some(self.value.fill_like(V::Scalar::ZERO))
            } else if let Some(_) = *rhs_grad {
                Some(self.value.fill_like(V::Scalar::ZERO))
            } else {
                None
            }
        };

        Variable(Rc::new(InternalVariable {
            value,
            grad: RefCell::new(grad),
            backward_op_name: "add_back",
            backward_closure: Box::new(move |grad| {
                rhs.backward(grad.to_contiguous());
                self.backward(grad);
            }),
        }))
    }
}