#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use custom_sleef_test::*;

// const VSIZE: usize = 1048576;
// const VSIZE: usize = 65536;
const VSIZE: usize = 4096;

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

fn bench_ops(c: &mut Criterion) {
    // let mut v: Vec<_> = (0..VSIZE).map(|x| -(x as f32)).collect();
    let mut v = vec![0_f32; VSIZE];
    let f = list!(increment, increment, increment, increment);
    let f_v = list!(increment_v, increment_v, increment_v, increment_v);

    let mut group = c.benchmark_group("ops");

    group.bench_function("add_unrolled", |b| b.iter(|| {
        f_v.run(&mut v);
        black_box(&mut v);
    }));

    for chunk_size in [4, 16, 64, 256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::new("add_chunked", chunk_size),chunk_size, |b, &chunk_size| b.iter(|| {
            for chunk in v.chunks_exact_mut(chunk_size) {
                for _ in 0..4 {
                    let ptr = chunk.as_mut_ptr();
                    for i in 0..(chunk.len() / 4) {
                        unsafe {
                            let chunk_ptr = ptr.offset((4 * i) as isize) as *mut __m128;
                            let res = increment(chunk_ptr.read());
                            chunk_ptr.write(res);
                        }
                    }
                }
            }
        }));
    }

    group.bench_function("add_fused", |b| b.iter(|| {
        simd_loop(&mut v, f);
    }));

    // group.bench_function("add_control", |b| b.iter(|| {
    //     for chunk in v.chunks_exact_mut(4) {
    //         for _ in 0..4 {
    //             let ptr = chunk.as_mut_ptr();
    //             for i in 0..(chunk.len() / 4) {
    //                 unsafe {
    //                     let chunk_ptr = ptr.offset((4 * i) as isize) as *mut __m128;
    //                     let res = increment(chunk_ptr.read());
    //                     chunk_ptr.write(res);
    //                 }
    //             }
    //         }
    //     }
    // }));

    group.finish();
}

criterion_group!(benches, bench_ops);
criterion_main!(benches);