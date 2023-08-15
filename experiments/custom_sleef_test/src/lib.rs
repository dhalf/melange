#![feature(test)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub mod sleef_custom;
pub use sleef_custom::{xexpf, increment};

pub trait Vectorizable {
    type Vector;
    const LEN: usize;
}

impl Vectorizable for f32 {
    type Vector = __m128;
    const LEN: usize = 4;
}

pub struct SimdLoop<T, C> {
    pub cls: C,
    pub values: Vec<T>,
}

pub fn simd_loop<T, F>(v: &mut [T], f: F)
where
    T: Copy + Vectorizable,
    F: Composer<T::Vector>,
{
    let ptr = v.as_mut_ptr();
    for i in 0..(v.len() / T::LEN) {
        unsafe {
            let chunk_ptr = ptr.offset((T::LEN * i) as isize) as *mut T::Vector;
            let res = f.run(chunk_ptr.read());
            chunk_ptr.write(res);
        }
    }
}

pub fn chunk_loop<T, C>(v: &mut [T], f: C, chunk_size: usize)
where
    T: Copy,
    C: Sequencer<[T]>,
{
    for chunk in v.chunks_exact_mut(chunk_size) {
        f.run(chunk);
    }
}

#[derive(Clone, Copy)]
pub struct Cons<A, B>(pub A, pub B);

#[derive(Clone, Copy)]
pub struct Nil;

pub trait Composer<T> {
    fn run(&self, x: T) -> T;
}

impl<A, B, T> Composer<T> for Cons<A, B>
where
    A: Fn(T) -> T,
    B: Composer<T>,
{
    #[inline(always)]
    fn run(&self, x: T) -> T {
        (self.0)(self.1.run(x))
    }
}

impl<T> Composer<T> for Nil {
    #[inline(always)]
    fn run(&self, x: T) -> T {
        x
    }
}

pub trait Sequencer<T: ?Sized> {
    fn run(&self, x: &mut T);
}

impl<A, B, T> Sequencer<T> for Cons<A, B>
where
    A: Fn(&mut T),
    B: Sequencer<T>,
    T: ?Sized,
{
    #[inline(always)]
    fn run(&self, x: &mut T) {
        self.1.run(x);
        (self.0)(x);
    }
}

impl<T: ?Sized> Sequencer<T> for Nil {
    #[inline(always)]
    fn run(&self, x: &mut T) {}
}

macro_rules! vectorize {
    ($op:ident as $name:ident for $t:ty) => {
        pub fn $name(v: &mut [$t]) {
            let ptr = v.as_mut_ptr();
            for i in 0..(v.len() / <$t>::LEN) {
                unsafe {
                    let chunk_ptr = ptr.offset((<$t>::LEN * i) as isize) as *mut <$t as Vectorizable>::Vector;
                    let res = $op(chunk_ptr.read());
                    chunk_ptr.write(res);
                }
            }
        }
    };
}

vectorize!(xexpf as xexpf_v for f32);
vectorize!(increment as increment_v for f32);

pub mod test {
    use super::*;

    macro_rules! list {
        ($($op0:ident$(, $ops:ident)*)?) => {
            {
                let f = Nil;
                $(
                    let f = Cons($op0, f);
                    $(
                        let f = Cons($ops, f);
                    )*
                )?
                f
            }
        };
    }

    #[test]
    fn unrolled() {
        let mut v: Vec<_> = (0..4096).map(|x| -(x as f32)).collect();
        let w: Vec<_> = v.iter().map(|x| x + 4.0).collect();
        let f_v = list!(increment_v, increment_v, increment_v, increment_v);
        f_v.run(&mut v);
        assert_eq!(v, w);
    }

    #[test]
    fn chunked() {
        let mut v: Vec<_> = (0..4096).map(|x| -(x as f32)).collect();
        let w: Vec<_> = v.iter().map(|x| x + 4.0).collect();
        let f_v = list!(increment_v, increment_v, increment_v, increment_v);
        chunk_loop(&mut v, f_v, 256);
        assert_eq!(v, w);
    }

    #[test]
    fn fused() {
        let mut v: Vec<_> = (0..4096).map(|x| -(x as f32)).collect();
        let w: Vec<_> = v.iter().map(|x| x + 4.0).collect();
        let f = list!(increment, increment, increment, increment);
        simd_loop(&mut v, f);
        assert_eq!(v, w);
    }
}

// pub struct Fuse<T, C> {
//     pub cls: C,
//     pub values: Vec<T>,
// }

// impl<T, C> Fuse<T, C>
// where
//     T: Copy + Vectorizable,
//     C: Composer<T::Vector>,
// {
//     pub fn run(&mut self) {
//         // for x in self.values.iter_mut() {
//         //     *x = self.cls.run(*x);
//         // }
//         let ptr = self.values.as_mut_ptr();
//         for i in 0..(self.values.len() / T::LEN) {
//             unsafe {
//                 let chunk_ptr = ptr.offset((T::LEN * i) as isize) as *mut T::Vector;
//                 let res = self.cls.run(chunk_ptr.read());
//                 chunk_ptr.write(res);
//             }
//         }
//     }
// }

// #[bench]
// fn main(b: &mut Bencher) {
//     // let a = unsafe { _mm_set_ps(1.0, 2.0, 3.0, 4.0) };
//     // let exp_a = xexpf(a);
//     // println!("{:?}", exp_a);

//     let v: Vec<_> = (0..65536).map(|x| x as f32).collect();

//     let mut v = Fuse {
//         cls: fuse!(xexpf, xexpf, xexpf, xexpf),
//         values: v,
//     };
//     v.run();
// }
