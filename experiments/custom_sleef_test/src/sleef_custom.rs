#![allow(non_upper_case_globals)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use core::f32::consts::LOG2_E as R_LN2f;
const L2Uf: f32 = 0.693145751953125_f32;
const L2Lf: f32 = 1.428606765330187045e-06_f32;

#[inline]
pub fn increment(x: __m128) -> __m128 {
    unsafe { _mm_add_ps(x, _mm_set1_ps(1.0_f32)) }
}

#[inline(always)]
fn mlaf(x: __m128, y: __m128, z: __m128) -> __m128 {
    unsafe { _mm_fmadd_ps(x, y, z) }
}

#[inline(always)]
fn rintfk(x: __m128) -> __m128 {
    const NEAREST: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    unsafe { _mm_round_ps::<NEAREST>(x) }
}

#[inline(always)]
pub fn pow2if(q: __m128i) -> __m128 {
    unsafe { _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(q, _mm_set1_epi32(0x7f)), 23)) }
}

#[inline(always)]
fn ldexp2kf(x: __m128, e: __m128i) -> __m128 {
    unsafe {
        _mm_mul_ps(
            _mm_mul_ps(x, pow2if(_mm_srai_epi32(e, 1))),
            pow2if(_mm_sub_epi32(e, _mm_srai_epi32(e, 1))),
        )
    }
}

#[inline]
pub fn xexpf(x: __m128) -> __m128 {
    unsafe {
        let q = rintfk(_mm_mul_ps(x, _mm_set1_ps(R_LN2f)));

        let s = mlaf(q, _mm_set1_ps(-L2Uf), x);
        let s = mlaf(q, _mm_set1_ps(-L2Lf), s);

        let u = _mm_set1_ps(0.000198527617612853646278381_f32);
        let u = mlaf(u, s, _mm_set1_ps(0.00139304355252534151077271_f32));
        let u = mlaf(u, s, _mm_set1_ps(0.00833336077630519866943359_f32));
        let u = mlaf(u, s, _mm_set1_ps(0.0416664853692054748535156_f32));
        let u = mlaf(u, s, _mm_set1_ps(0.166666671633720397949219_f32));
        let u = mlaf(u, s, _mm_set1_ps(0.5_f32));

        let u = _mm_add_ps(_mm_set1_ps(1.0_f32), mlaf(_mm_mul_ps(s, s), u, s));

        let u = ldexp2kf(u, _mm_cvtps_epi32(q));

        // let u = _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(_mm_cmplt_ps(x, _mm_set1_ps(-104.0))), _mm_castps_si128(u)));
        let u = _mm_andnot_ps(_mm_cmplt_ps(x, _mm_set1_ps(-104.0)), u);
        
        // let u = _mm_blendv_ps(u, _mm_set1_ps(f32::INFINITY), _mm_castsi128_ps(_mm_castps_si128(_mm_cmplt_ps(_mm_set1_ps(100.0), x))));
        let u = _mm_blendv_ps(u, _mm_set1_ps(f32::INFINITY), _mm_cmplt_ps(_mm_set1_ps(100.0), x));

        u
    }
}
