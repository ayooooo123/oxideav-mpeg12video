//! Bit-exact eight-lane pixel and four-column integer transform kernels.
#[cfg(target_arch = "aarch64")]
mod arch {
    use core::arch::aarch64::*;
    pub type V = int32x4_t;
    #[inline] pub unsafe fn splat(x: i32) -> V { unsafe { vdupq_n_s32(x) } }
    #[inline] pub unsafe fn add(a: V, b: V) -> V { unsafe { vaddq_s32(a, b) } }
    #[inline] pub unsafe fn sub(a: V, b: V) -> V { unsafe { vsubq_s32(a, b) } }
    #[inline] pub unsafe fn mul(a: V, b: i32) -> V { unsafe { vmulq_n_s32(a, b) } }
    #[inline] pub unsafe fn load(p: *const i16) -> V { unsafe { vmovl_s16(vld1_s16(p)) } }
    #[inline] pub unsafe fn store(p: *mut i32, v: V) { unsafe { vst1q_s32(p, vshrq_n_s32::<20>(v)); } }
    #[inline]
    pub unsafe fn pixels(d: *mut u8, s: *const u8, stride: usize, phase: u8, round: u16, avg: bool) {
        // SAFETY: the caller guarantees the pointers cover the lanes read and written.
        unsafe {
            let a = vld1_u8(s);
            let mut v = match phase {
                0 => a,
                1 | 2 => {
                    let b = vld1_u8(s.add(if phase == 1 { 1 } else { stride }));
                    if round == 1 { vrhadd_u8(a, b) } else { vhadd_u8(a, b) }
                }
                _ => {
                    let b = vld1_u8(s.add(1));
                    let c = vld1_u8(s.add(stride));
                    let e = vld1_u8(s.add(stride + 1));
                    let sum = vaddq_u16(vaddl_u8(a, b), vaddl_u8(c, e));
                    vshrn_n_u16::<2>(vaddq_u16(sum, vdupq_n_u16(round + 1)))
                }
            };
            if avg { v = vrhadd_u8(v, vld1_u8(d)); }
            vst1_u8(d, v);
        }
    }
    #[inline]
    pub unsafe fn residual(d: *mut u8, r: *const i16, add_prediction: bool) {
        // SAFETY: the caller guarantees the pointers cover the lanes read and written.
        unsafe {
            let mut v = vld1q_s16(r);
            if add_prediction { v = vqaddq_s16(v, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(d)))); }
            vst1_u8(d, vqmovun_s16(v));
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod arch {
    use core::arch::wasm32::*;
    pub type V = v128;
    #[inline] pub unsafe fn splat(x: i32) -> V { i32x4_splat(x) }
    #[inline] pub unsafe fn add(a: V, b: V) -> V { i32x4_add(a, b) }
    #[inline] pub unsafe fn sub(a: V, b: V) -> V { i32x4_sub(a, b) }
    #[inline] pub unsafe fn mul(a: V, b: i32) -> V { i32x4_mul(a, i32x4_splat(b)) }
    #[inline] pub unsafe fn load(p: *const i16) -> V { unsafe { i32x4_extend_low_i16x8(v128_load64_zero(p.cast())) } }
    #[inline] pub unsafe fn store(p: *mut i32, v: V) { unsafe { v128_store(p.cast(), i32x4_shr(v, 20)); } }
    #[inline]
    pub unsafe fn pixels(d: *mut u8, s: *const u8, stride: usize, phase: u8, round: u16, avg: bool) {
        // SAFETY: the caller guarantees the pointers cover the lanes read and written.
        unsafe {
            let a = v128_load64_zero(s.cast());
            let mut v = match phase {
                0 => a,
                1 | 2 => {
                    let b = v128_load64_zero(s.add(if phase == 1 { 1 } else { stride }).cast());
                    if round == 1 { u8x16_avgr(a, b) } else {
                        u8x16_sub(u8x16_avgr(a, b), v128_and(v128_xor(a, b), u8x16_splat(1)))
                    }
                }
                _ => {
                    let b = v128_load64_zero(s.add(1).cast());
                    let c = v128_load64_zero(s.add(stride).cast());
                    let e = v128_load64_zero(s.add(stride + 1).cast());
                    let sum = i16x8_add(i16x8_add(u16x8_extend_low_u8x16(a), u16x8_extend_low_u8x16(b)),
                        i16x8_add(u16x8_extend_low_u8x16(c), u16x8_extend_low_u8x16(e)));
                    let v = u16x8_shr(i16x8_add(sum, i16x8_splat((round + 1) as i16)), 2);
                    u8x16_narrow_i16x8(v, v)
                }
            };
            if avg { v = u8x16_avgr(v, v128_load64_zero(d.cast())); }
            v128_store64_lane::<0>(v, d.cast());
        }
    }
    #[inline]
    pub unsafe fn residual(d: *mut u8, r: *const i16, add_prediction: bool) {
        // SAFETY: the caller guarantees the pointers cover the lanes read and written.
        unsafe {
            let mut v = v128_load(r.cast());
            if add_prediction { v = i16x8_add_sat(v, u16x8_extend_low_u8x16(v128_load64_zero(d.cast()))); }
            v128_store64_lane::<0>(u8x16_narrow_i16x8(v, v), d.cast());
        }
    }
}

/// Caller validates complete source/destination rectangles before entering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixels(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize, phase: u8, round: u16, avg: bool) {
    if w == 0 || h == 0 { return; }
    assert!(w.is_multiple_of(8) && phase < 4 && round <= 1);
    assert!((h - 1) * ds + w <= dst.len());
    assert!((h - 1 + usize::from(phase & 2 != 0)) * ss + w + usize::from(phase & 1 != 0) <= src.len());
    unsafe {
        for y in 0..h {
            for x in (0..w).step_by(8) {
                arch::pixels(dst.as_mut_ptr().add(y * ds + x), src.as_ptr().add(y * ss + x), ss, phase, round, avg);
            }
        }
    }
}

pub(crate) fn residual(dst: &mut [u8], row: &[i16; 8], add_prediction: bool) {
    assert!(dst.len() >= 8);
    unsafe { arch::residual(dst.as_mut_ptr(), row.as_ptr(), add_prediction); }
}

/// The column pass preserves wrapping i32 arithmetic before its signed shift.
pub(crate) fn columns(block: &[i16; 64]) -> [[i32; 8]; 8] {
    use arch::*;
    let mut out = [[0i32; 8]; 8];
    unsafe {
        for c in [0, 4] {
            let r = |n: usize| load(block.as_ptr().add(n * 8 + c));
            let base = mul(add(r(0), splat(32)), 16383);
            let a0 = add(add(base, mul(r(2), 21407)), add(mul(r(4), 16383), mul(r(6), 8867)));
            let a1 = sub(sub(add(base, mul(r(2), 8867)), mul(r(4), 16383)), mul(r(6), 21407));
            let a2 = add(sub(sub(base, mul(r(2), 8867)), mul(r(4), 16383)), mul(r(6), 21407));
            let a3 = sub(add(sub(base, mul(r(2), 21407)), mul(r(4), 16383)), mul(r(6), 8867));
            let b0 = add(add(mul(r(1), 22725), mul(r(3), 19266)), add(mul(r(5), 12873), mul(r(7), 4520)));
            let b1 = sub(sub(sub(mul(r(1), 19266), mul(r(3), 4520)), mul(r(5), 22725)), mul(r(7), 12873));
            let b2 = add(sub(mul(r(1), 12873), mul(r(3), 22725)), add(mul(r(5), 4520), mul(r(7), 19266)));
            let b3 = sub(add(sub(mul(r(1), 4520), mul(r(3), 12873)), mul(r(5), 19266)), mul(r(7), 22725));
            let rows = [add(a0,b0), add(a1,b1), add(a2,b2), add(a3,b3), sub(a3,b3), sub(a2,b2), sub(a1,b1), sub(a0,b0)];
            for (row, v) in out.iter_mut().zip(rows) { store(row.as_mut_ptr().add(c), v); }
        }
    }
    out
}

#[cfg(test)]
#[path = "simd_tests.rs"]
mod tests;
