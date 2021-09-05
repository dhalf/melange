use std::ops::*;
use crate::ops::*;
use std::cell::RefCell;
use std::rc::Weak;
use super::accumulate_on::AccumulateOn;
use super::rvar::*;
use super::*;

macro_rules! op_switch {
    (op2, $self:ident, $other:ident, $fn:ident) => {
        RVar::clone(&$self).op2(
            RVar::clone(&$other),
            |x: &T, y: &U| x.$fn(y),
        )
    };
    (op2_merge, $self:ident, $other:ident, $fn:ident) => {
        RVar::clone(&$self).op2_merge(
            RVar::clone(&$other),
            |x: &T, y: &U| x.$fn(y),
            |x: T, y: &U| x.$fn(y),
            |x: &T, y: U| x.$fn(y),
        )
    };
}

macro_rules! binop {
    ($(
        $trait:ident
        $fn:ident
        $op:ident
        $(where $($generic:ty $(| $($lgen:lifetime),+)?: $($bound:path)|*),*)?
        { $(($a:ident, $b:ident) =>)? $variant:ident $backward_closure:expr }
    )*) => {$(
        impl<T, U, V> $trait<RVar<U>> for RVar<T>
        where
            T: Differentiable + for<'a> $trait<&'a U, Output = V>,
            for<'a> &'a T: $trait<U, Output = V> + $trait<&'a U, Output = V>,
            U: Differentiable,
            V: Differentiable,
            V::RGrad: AccumulateOn<T::RGrad> + AccumulateOn<U::RGrad>,
            $($($(for<$($lgen),+>)? $generic: $($bound +)*),*)?
        {
            type Output = RVar<V>;
            fn $fn(self, other: RVar<U>) -> Self::Output {
                let value = op_switch!($op, self, other, $fn);
                $(
                    let $a = Rc::downgrade(&self.0);
                    let $b = Rc::downgrade(&other.0);
                )?
                RVar(Rc::new(InnerRVar::<V, (T, U)> {
                    grad: RefCell::new(Grad::NonAllocated(V::alloc_data(&value))),
                    value: RefCell::new(Some(value)),
                    retain_grad: self.0.retains_grad() || other.0.retains_grad(),
                    parents: (self.0, other.0),
                    grad_fn: GradFn::$variant(Box::new($backward_closure)),
                }))
            }
        }
        
        impl<T> $trait<&RVar<T>> for RVar<T>
        where
            T: Differentiable,
            RVar<T>: $trait<Output = RVar<T>>,
        {
            type Output = RVar<T>;
            fn $fn(self, other: &RVar<T>) -> Self::Output {
                let other = RVar::clone(other);
                self.$fn(other)
            }
        }

        impl<T> $trait<RVar<T>> for &RVar<T>
        where
            T: Differentiable,
            RVar<T>: $trait<Output = RVar<T>>,
        {
            type Output = RVar<T>;
            fn $fn(self, other: RVar<T>) -> Self::Output {
                let this = RVar::clone(self);
                this.$fn(other)
            }
        }

        impl<T> $trait<&RVar<T>> for &RVar<T>
        where
            T: Differentiable,
            RVar<T>: $trait<Output = RVar<T>>,
        {
            type Output = RVar<T>;
            fn $fn(self, other: &RVar<T>) -> Self::Output {
                let this = RVar::clone(self);
                let other = RVar::clone(other);
                this.$fn(other)
            }
        }
    )*};
}

binop! {
    Add add op2_merge { 
        GradOnly |grad| (grad.clone().onto(), grad.onto())
    }
    
    Sub sub op2_merge where V::RGrad: Neg<Output = V::RGrad> { 
        GradOnly |grad| (grad.clone().onto(), (-grad).onto())
    }

    Mul mul op2
    where
        &'a T | 'a: Conj<Output = T>,
        &'a U | 'a: Conj<Output = U>,
        V::RGrad: Mul<U, Output = V::RGrad> | Mul<T, Output = V::RGrad>
    { 
        (a, b) => GradOnly move |grad| (
            (grad.clone() * Weak::upgrade(&b).unwrap().borrow_value().conj()).onto(),
            (grad * Weak::upgrade(&a).unwrap().borrow_value().conj()).onto()
        )
    }

    Div div op2
    where
        &'a T | 'a: Neg<Output = T>,
        V | 'a: Div<&'a U, Output = V>,
        V: Conj<Output = V>,
        &'a U | 'a: Recip<Output = U>,
        U: Conj<Output = U>,
        V::RGrad: Mul<U, Output = V::RGrad> | Mul<V, Output = V::RGrad>
    { 
        (a, b) => GradOnly move |grad| {
            let a = Weak::upgrade(&a).unwrap();
            let a = a.borrow_value();
            let b = Weak::upgrade(&b).unwrap();
            let b = b.borrow_value();
            (
                (grad.clone() * ((-&*a / &*b) / &*b).conj()).onto(),
                (grad * b.recip().conj()).onto()
            )
        }
    }
}
