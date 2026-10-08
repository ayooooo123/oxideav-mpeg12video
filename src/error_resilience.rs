// Ported from FFmpeg (commit 2da55bf): libavcodec/error_resilience.c
// (ff_er_frame_start, ff_er_add_slice, ff_er_frame_end, put_dc, filter181,
// guess_dc, h_block_filter, v_block_filter, guess_mv, is_intra_more_likely)
// for MPEG-1/2 frame pictures, with libavcodec/me_cmp.c (pix_abs16_c),
// libavcodec/mpeg_er.c (ff_mpeg_er_frame_start, mpeg_er_decode_mb) and
// libavcodec/mpegvideo.c (mb_index2xy).
// License: LGPL-2.1-or-later
// Copyright (c) 2000, 2001 Fabrice Bellard
// Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>

//! Error concealment as FFmpeg's MPEG-1/2 decoder runs it (error_concealment
//! guess_mvs + deblock, its default). Each slice marks the macroblocks it
//! decoded and where it failed; at the end of the picture the damaged
//! ones (with up to 50 before each error in its slice) are rebuilt: by a
//! zero-vector prediction from the references (FFmpeg's MPEG-1/2 decoder
//! keeps no motion vectors, so its search has only zero vectors to try),
//! or, where intra coding is likelier, as flat blocks of DC values
//! interpolated from the undamaged blocks; then the block edges next to
//! damage are smoothed.

use crate::ff_decode::{
    is_inter, is_intra, motion, Cur, MbPrediction, MvType, Pic, MB_TYPE_16X16, MB_TYPE_FORWARD_MV, MB_TYPE_INTRA, MV_DIR_BACKWARD,
    MV_DIR_FORWARD, PICT_B, PICT_I,
};
use crate::frame_assembly::FrameBuffer;

const VP_START: u8 = 1;
pub(crate) const ER_AC_ERROR: u8 = 2;
pub(crate) const ER_DC_ERROR: u8 = 4;
pub(crate) const ER_MV_ERROR: u8 = 8;
pub(crate) const ER_AC_END: u8 = 16;
pub(crate) const ER_DC_END: u8 = 32;
pub(crate) const ER_MV_END: u8 = 64;
const ER_MB_ERROR: u8 = ER_AC_ERROR | ER_DC_ERROR | ER_MV_ERROR;
const ER_MB_END: u8 = ER_AC_END | ER_DC_END | ER_MV_END;
const INT_MAX: i64 = i32::MAX as i64;

/// ERContext: the error status of each macroblock of the picture.
pub(crate) struct Er {
    mb_width: usize,
    mb_height: usize,
    mb_stride: usize,
    mb_num: usize,
    status: Vec<u8>,
    error_count: i64,
}

impl Er {
    pub(crate) fn new(mb_width: usize, mb_height: usize) -> Self {
        let mb_stride = mb_width + 1;
        Self { mb_width, mb_height, mb_stride, mb_num: mb_width * mb_height, status: vec![0; mb_stride * mb_height], error_count: 0 }
    }

    /// mb_index2xy, whose entry past the end is (mb_height - 1) *
    /// mb_stride + mb_width.
    fn xy(&self, i: usize) -> usize {
        if i == self.mb_num { (self.mb_height - 1) * self.mb_stride + self.mb_width } else { (i % self.mb_width) + (i / self.mb_width) * self.mb_stride }
    }

    /// ff_er_frame_start: every macroblock damaged until a slice says
    /// otherwise.
    pub(crate) fn frame_start(&mut self) {
        self.status.fill(ER_MB_ERROR | VP_START | ER_MB_END);
        self.error_count = 3 * self.mb_num as i64;
    }

    /// ff_er_add_slice, without slice threads.
    pub(crate) fn add_slice(&mut self, startx: i32, starty: i32, endx: i32, endy: i32, status: u8) {
        let mb_w = self.mb_width as i64;
        let start_i = (i64::from(startx) + i64::from(starty) * mb_w).clamp(0, self.mb_num as i64 - 1) as usize;
        let end_i = (i64::from(endx) + i64::from(endy) * mb_w).clamp(0, self.mb_num as i64) as usize;
        let (start_xy, end_xy) = (self.xy(start_i), self.xy(end_i));
        if start_i > end_i || start_xy > end_xy {
            return; // internal error, slice end before start
        }
        let mut mask: u8 = !VP_START;
        let covered = start_i as i64 - end_i as i64 - 1;
        for (error, end) in [(ER_AC_ERROR, ER_AC_END), (ER_DC_ERROR, ER_DC_END), (ER_MV_ERROR, ER_MV_END)] {
            if status & (error | end) != 0 {
                mask &= !(error | end);
                self.error_count += covered;
            }
        }
        if status & ER_MB_ERROR != 0 {
            self.error_count = INT_MAX;
        }
        for s in &mut self.status[start_xy..end_xy] {
            *s &= mask;
        }
        if end_i == self.mb_num {
            self.error_count = INT_MAX;
        } else {
            self.status[end_xy] &= mask;
            self.status[end_xy] |= status;
        }
        self.status[start_xy] |= VP_START;
        if start_xy > 0 && self.status[self.xy(start_i - 1)] & !VP_START != ER_MB_END {
            self.error_count = INT_MAX;
        }
    }
}

/// mpeg_er_decode_mb: a zero-residual prediction of one macroblock.
fn er_decode_mb(pic: &Pic, cur: &mut FrameBuffer, mb_x: usize, mb_y: usize, mv_dir: u8) {
    let zero = [[[0i32; 2]; 4]; 2];
    let prediction = MbPrediction { mv_dir, mv_type: MvType::M16x16, mv: &zero, field_select: &[[0; 2]; 2] };
    motion(pic, cur, mb_x, mb_y, &prediction);
}

/// pix_abs16_c over two 16x16 blocks of one stride.
fn sad16(a: &[u8], ao: usize, b: &[u8], bo: usize, stride: usize) -> i64 {
    (0..16)
        .map(|y| a[ao + y * stride..ao + y * stride + 16].iter().zip(&b[bo + y * stride..bo + y * stride + 16]).map(|(&p, &q)| (i64::from(p) - i64::from(q)).abs()).sum::<i64>())
        .sum()
}

/// is_intra_more_likely.
fn is_intra_more_likely(er: &Er, pic: &Pic, cur: &Cur) -> bool {
    let Some(last) = pic.last else {
        return true; // no previous frame: spatial prediction
    };
    let damaged = |e: u8| e & ER_DC_ERROR != 0 && e & ER_MV_ERROR != 0;
    let undamaged = (0..er.mb_num).filter(|&i| !damaged(er.status[er.xy(i)])).count();
    if undamaged < 5 {
        return false; // almost all damaged: temporal prediction
    }
    let skip_amount = (undamaged / 50).max(1);
    let mut likely: i64 = 0;
    let mut j = 0usize;
    let linesize = cur.fb.y.width();
    for mb_y in 0..er.mb_height - 1 {
        for mb_x in 0..er.mb_width {
            let mb_xy = mb_x + mb_y * er.mb_stride;
            if damaged(er.status[mb_xy]) {
                continue;
            }
            j += 1;
            if j % skip_amount != 0 {
                continue;
            }
            if pic.pict_type == PICT_I {
                let at = mb_x * 16 + mb_y * 16 * linesize;
                likely += sad16(last.y.samples(), at, cur.fb.y.samples(), at, linesize);
                likely -= sad16(last.y.samples(), at, last.y.samples(), at + linesize * 16, linesize);
            } else if is_intra(cur.mb_type[mb_xy]) {
                likely += 1;
            } else {
                likely -= 1;
            }
        }
    }
    likely > 0
}

/// put_dc: a macroblock as flat blocks of its DC values.
fn put_dc(fb: &mut FrameBuffer, dc: &[Vec<i16>; 3], b8_stride: usize, mb_stride: usize, mb_x: usize, mb_y: usize) {
    let linesize = fb.y.width();
    let uvlinesize = fb.cb.width();
    let flat = |v: i16| (i32::from(v).clamp(0, 2040) / 8) as u8;
    let y = fb.y.samples_mut();
    for i in 0..4 {
        let v = flat(dc[0][mb_x * 2 + (i & 1) + (mb_y * 2 + (i >> 1)) * b8_stride]);
        for r in 0..8 {
            let o = mb_x * 16 + (i & 1) * 8 + (mb_y * 16 + (i >> 1) * 8 + r) * linesize;
            y[o..o + 8].fill(v);
        }
    }
    let (u, v) = (flat(dc[1][mb_x + mb_y * mb_stride]), flat(dc[2][mb_x + mb_y * mb_stride]));
    for r in 0..8 {
        let o = mb_x * 8 + (mb_y * 8 + r) * uvlinesize;
        fb.cb.samples_mut()[o..o + 8].fill(u);
        fb.cr.samples_mut()[o..o + 8].fill(v);
    }
}

/// filter181 on the luma DC values.
fn filter181(data: &mut [i16], width: usize, height: usize, stride: usize) {
    let filter = |prev: i32, x: i16, next: i16| -> i16 {
        let dc = -prev + i32::from(x) * 8 - i32::from(next);
        ((dc.clamp(i32::MIN / 10923, i32::MAX / 10923 - 32768) * 10923 + 32768) >> 16) as i16
    };
    for y in 1..height - 1 {
        let mut prev = i32::from(data[y * stride]);
        for x in 1..width - 1 {
            let dc = filter(prev, data[x + y * stride], data[x + 1 + y * stride]);
            prev = i32::from(data[x + y * stride]);
            data[x + y * stride] = dc;
        }
    }
    for x in 1..width - 1 {
        let mut prev = i32::from(data[x]);
        for y in 1..height - 1 {
            let dc = filter(prev, data[x + y * stride], data[x + (y + 1) * stride]);
            prev = i32::from(data[x + y * stride]);
            data[x + y * stride] = dc;
        }
    }
}

/// guess_dc: each damaged intra block's DC from the nearest undamaged
/// ones in its row and column, weighted by inverse distance.
fn guess_dc(er: &Er, mb_type: &[u32], dc: &mut [i16], w: usize, h: usize, stride: usize, is_luma: bool) {
    let sh = usize::from(is_luma);
    let mb = |b_x: usize, b_y: usize| (b_x >> sh) + (b_y >> sh) * er.mb_stride;
    let source = |b_x: usize, b_y: usize| !is_intra(mb_type[mb(b_x, b_y)]) || er.status[mb(b_x, b_y)] & ER_DC_ERROR == 0;
    let mut col = vec![[0i16; 4]; stride * h];
    let mut dist = vec![[0u32; 4]; stride * h];
    // One direction: `cells` in scan order, the slot to fill, the
    // position along the scan of each cell.
    let mut sweep = |cells: &mut dyn Iterator<Item = (usize, usize)>, slot: usize, along: fn(usize, usize) -> i64| {
        let (mut color, mut distance) = (1024i16, -1i64);
        for (b_x, b_y) in cells {
            if source(b_x, b_y) {
                color = dc[b_x + b_y * stride];
                distance = along(b_x, b_y);
            }
            col[b_x + b_y * stride][slot] = color;
            dist[b_x + b_y * stride][slot] = if distance >= 0 { (along(b_x, b_y) - distance).unsigned_abs() as u32 } else { 9999 };
        }
    };
    for b_y in 0..h {
        sweep(&mut (0..w).map(|b_x| (b_x, b_y)), 1, |x, _| x as i64);
        sweep(&mut (0..w).rev().map(|b_x| (b_x, b_y)), 0, |x, _| x as i64);
    }
    for b_x in 0..w {
        sweep(&mut (0..h).map(|b_y| (b_x, b_y)), 3, |_, y| y as i64);
        sweep(&mut (0..h).rev().map(|b_y| (b_x, b_y)), 2, |_, y| y as i64);
    }
    for b_y in 0..h {
        for b_x in 0..w {
            let m = mb(b_x, b_y);
            if is_inter(mb_type[m]) || er.status[m] & ER_DC_ERROR == 0 {
                continue;
            }
            let (mut guess, mut weight_sum) = (0i64, 0i64);
            for j in 0..4 {
                let weight = i64::from(268_435_456 / dist[b_x + b_y * stride][j].max(1));
                guess += weight * i64::from(col[b_x + b_y * stride][j]);
                weight_sum += weight;
            }
            dc[b_x + b_y * stride] = ((guess + weight_sum / 2) / weight_sum) as i16;
        }
    }
}

/// h_block_filter (`horizontal`: the edges between left and right
/// neighbours) or v_block_filter over w x h blocks of 8. Two inter
/// neighbours have equal (zero) vectors and are left alone.
#[allow(clippy::too_many_arguments)]
fn block_filter(er: &Er, mb_type: &[u32], px: &mut [u8], w: usize, h: usize, stride: usize, is_luma: bool, horizontal: bool) {
    let sh = usize::from(is_luma);
    let (bw, bh) = if horizontal { (w - 1, h) } else { (w, h - 1) };
    for b_y in 0..bh {
        for b_x in 0..bw {
            let (n_x, n_y) = if horizontal { (b_x + 1, b_y) } else { (b_x, b_y + 1) };
            let first = (b_x >> sh) + (b_y >> sh) * er.mb_stride;
            let second = (n_x >> sh) + (n_y >> sh) * er.mb_stride;
            let first_damaged = er.status[first] & ER_MB_ERROR != 0;
            let second_damaged = er.status[second] & ER_MB_ERROR != 0;
            if !(first_damaged || second_damaged) {
                continue;
            }
            if !is_intra(mb_type[first]) && !is_intra(mb_type[second]) {
                continue;
            }
            let offset = b_x * 8 + b_y * stride * 8;
            let (step, across) = if horizontal { (stride, 1) } else { (1, stride) };
            for k in 0..8 {
                let at = |j: usize| offset + k * step + j * across;
                let s = |j: usize| i32::from(px[at(j)]);
                let (a, b, c) = (s(7) - s(6), s(8) - s(7), s(9) - s(8));
                let mut d = (b.abs() - ((a.abs() + c.abs() + 1) >> 1)).max(0);
                if b < 0 {
                    d = -d;
                }
                if d == 0 {
                    continue;
                }
                if !(first_damaged && second_damaged) {
                    d = d * 16 / 9;
                }
                if first_damaged {
                    for (j, m) in [(7, 7), (6, 5), (5, 3), (4, 1)] {
                        let o = at(j);
                        px[o] = (i32::from(px[o]) + ((d * m) >> 4)).clamp(0, 255) as u8;
                    }
                }
                if second_damaged {
                    for (j, m) in [(8, 7), (9, 5), (10, 3), (11, 1)] {
                        let o = at(j);
                        px[o] = (i32::from(px[o]) - ((d * m) >> 4)).clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
}

/// ff_er_frame_end for a frame picture of MPEG-1/2. `mbskip` is the
/// picture's mbskip_table.
pub(crate) fn frame_end(er: &mut Er, pic: &Pic, cur: &mut Cur, mbskip: &[u8]) {
    if er.error_count == 0 {
        return;
    }
    let (mb_width, mb_height, mb_stride, mb_num) = (er.mb_width, er.mb_height, er.mb_stride, er.mb_num);
    if pic.mpeg2 && pic.height.next_multiple_of(16) & 16 != 0 && er.error_count == 3 * mb_width as i64 {
        let row = (mb_height - 1) * mb_stride;
        if (0..mb_width).all(|x| er.status[row + x] == 0x7F) {
            return; // ignoring last missing slice
        }
    }

    // Overlapping slices.
    for error_type in 1..=3u8 {
        let mut end_ok = false;
        for i in (0..mb_num).rev() {
            let mb_xy = er.xy(i);
            let error = er.status[mb_xy];
            if error & (1 << error_type) != 0 || error & (8 << error_type) != 0 {
                end_ok = true;
            }
            if !end_ok {
                er.status[mb_xy] |= 1 << error_type;
            }
            if error & VP_START != 0 {
                end_ok = false;
            }
        }
    }
    // Backward: up to 50 macroblocks (not skipped) before an error.
    let mut distance: i64 = 9_999_999;
    for error_type in 1..=3u8 {
        for i in (0..mb_num).rev() {
            let mb_xy = er.xy(i);
            let error = er.status[mb_xy];
            if mbskip[mb_xy] == 0 {
                distance += 1;
            }
            if error & (1 << error_type) != 0 {
                distance = 0;
            }
            if distance < 50 {
                er.status[mb_xy] |= 1 << error_type;
            }
            if error & VP_START != 0 {
                distance = 9_999_999;
            }
        }
    }
    // Forward: the rest of a damaged slice.
    let mut error = 0u8;
    for i in 0..mb_num {
        let mb_xy = er.xy(i);
        let old = er.status[mb_xy];
        if old & VP_START != 0 {
            error = old & ER_MB_ERROR;
        } else {
            error |= old & ER_MB_ERROR;
            er.status[mb_xy] |= error;
        }
    }
    // Not partitioned: any damage is all three kinds.
    for i in 0..mb_num {
        let mb_xy = er.xy(i);
        if er.status[mb_xy] & ER_MB_ERROR != 0 {
            er.status[mb_xy] |= ER_MB_ERROR;
        }
    }

    let guessed = if is_intra_more_likely(er, pic, cur) { MB_TYPE_INTRA } else { MB_TYPE_16X16 | MB_TYPE_FORWARD_MV };
    for i in 0..mb_num {
        let mb_xy = er.xy(i);
        let e = er.status[mb_xy];
        if e & ER_DC_ERROR != 0 && e & ER_MV_ERROR != 0 {
            cur.mb_type[mb_xy] = guessed;
        }
    }
    // An I- or P-picture's next_pic is itself; a B-picture has its own.
    let have_last = pic.last.is_some();
    let have_next = pic.pict_type != PICT_B || pic.next.is_some();
    if !have_last && !have_next {
        for t in &mut cur.mb_type {
            if !is_intra(*t) {
                *t = MB_TYPE_INTRA;
            }
        }
    }

    // Inter macroblocks with damaged AC only: their (zero) vector again.
    let one_way = if have_last { MV_DIR_FORWARD } else { MV_DIR_BACKWARD };
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let mb_xy = mb_x + mb_y * mb_stride;
            let e = er.status[mb_xy];
            if is_intra(cur.mb_type[mb_xy]) || e & ER_MV_ERROR != 0 || e & ER_AC_ERROR == 0 {
                continue;
            }
            er_decode_mb(pic, &mut cur.fb, mb_x, mb_y, one_way);
        }
    }

    if pic.pict_type == PICT_B {
        // pp_time is MPEG-4's: zero vectors, both ways where they exist.
        let mut mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;
        if !have_last {
            mv_dir &= !MV_DIR_FORWARD;
        }
        if !have_next {
            mv_dir &= !MV_DIR_BACKWARD;
        }
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                let mb_xy = mb_x + mb_y * mb_stride;
                let e = er.status[mb_xy];
                if is_intra(cur.mb_type[mb_xy]) || e & ER_MV_ERROR == 0 || e & ER_AC_ERROR == 0 {
                    continue;
                }
                er_decode_mb(pic, &mut cur.fb, mb_x, mb_y, mv_dir);
            }
        }
    } else {
        // guess_mv: every candidate it weighs is the zero vector, so each
        // inter macroblock with a damaged vector is predicted with it, in
        // the rows both references cover.
        let mut rows = mb_height.min(cur.fb.height.div_ceil(16));
        if let Some(last) = pic.last {
            rows = rows.min(last.height.div_ceil(16));
        }
        for mb_y in 0..rows {
            for mb_x in 0..mb_width {
                let mb_xy = mb_x + mb_y * mb_stride;
                if is_intra(cur.mb_type[mb_xy]) || er.status[mb_xy] & ER_MV_ERROR == 0 {
                    continue;
                }
                er_decode_mb(pic, &mut cur.fb, mb_x, mb_y, one_way);
            }
        }
    }

    // The DC of every block as the picture now is.
    let b8_stride = mb_width * 2 + 1;
    let mut dc = [vec![0i16; b8_stride * mb_height * 2], vec![0i16; mb_stride * mb_height], vec![0i16; mb_stride * mb_height]];
    let linesize = cur.fb.y.width();
    let uvlinesize = cur.fb.cb.width();
    let block_dc = |plane: &[u8], at: usize, stride: usize| -> i16 {
        let sum: i32 = (0..8).map(|y| plane[at + y * stride..at + y * stride + 8].iter().map(|&v| i32::from(v)).sum::<i32>()).sum();
        ((sum + 4) >> 3) as i16
    };
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            for n in 0..4 {
                let at = mb_x * 16 + (n & 1) * 8 + (mb_y * 16 + (n >> 1) * 8) * linesize;
                dc[0][mb_x * 2 + (n & 1) + (mb_y * 2 + (n >> 1)) * b8_stride] = block_dc(cur.fb.y.samples(), at, linesize);
            }
            let at = mb_x * 8 + mb_y * 8 * uvlinesize;
            dc[1][mb_x + mb_y * mb_stride] = block_dc(cur.fb.cb.samples(), at, uvlinesize);
            dc[2][mb_x + mb_y * mb_stride] = block_dc(cur.fb.cr.samples(), at, uvlinesize);
        }
    }
    guess_dc(er, &cur.mb_type, &mut dc[0], mb_width * 2, mb_height * 2, b8_stride, true);
    guess_dc(er, &cur.mb_type, &mut dc[1], mb_width, mb_height, mb_stride, false);
    guess_dc(er, &cur.mb_type, &mut dc[2], mb_width, mb_height, mb_stride, false);
    filter181(&mut dc[0], mb_width * 2, mb_height * 2, b8_stride);

    // Damaged intra macroblocks as their DC values.
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let mb_xy = mb_x + mb_y * mb_stride;
            if is_inter(cur.mb_type[mb_xy]) || er.status[mb_xy] & ER_AC_ERROR == 0 {
                continue;
            }
            put_dc(&mut cur.fb, &dc, b8_stride, mb_stride, mb_x, mb_y);
        }
    }

    // Deblock the edges next to damage.
    let Cur { fb, mb_type } = cur;
    block_filter(er, mb_type, fb.y.samples_mut(), mb_width * 2, mb_height * 2, linesize, true, true);
    block_filter(er, mb_type, fb.y.samples_mut(), mb_width * 2, mb_height * 2, linesize, true, false);
    block_filter(er, mb_type, fb.cb.samples_mut(), mb_width, mb_height, uvlinesize, false, true);
    block_filter(er, mb_type, fb.cr.samples_mut(), mb_width, mb_height, uvlinesize, false, true);
    block_filter(er, mb_type, fb.cb.samples_mut(), mb_width, mb_height, uvlinesize, false, false);
    block_filter(er, mb_type, fb.cr.samples_mut(), mb_width, mb_height, uvlinesize, false, false);
}
