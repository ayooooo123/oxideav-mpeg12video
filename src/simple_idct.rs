// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg 2da55bf libavcodec/simple_idct_template.c (8-bit / int16).
// Copyright (c) 2001 Michael Niedermayer <michaelni@gmx.at>
// Based upon code from mpeg2dec by Aaron Holtzman <aholtzma@ess.engr.uvic.ca>.
//
// This file is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 2.1, or (at your option) any later version.
// It is distributed WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE-LGPL.

#[inline(always)]
fn m(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }
const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19266;
const W4: i32 = 16383;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;
const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 20;
const DC_SHIFT: u32 = 3;

/// `idctRowCondDC_int16_8bit(row, 0)` (64-bit build: the DC shortcut tests
/// coefficients 1..7 of the row).
fn idct_row_cond_dc(row: &mut [i16]) {
    if row[1..8].iter().all(|&v| v == 0) {
        let t = ((row[0] as i32) << DC_SHIFT) as i16;
        row[..8].fill(t);
        return;
    }
    let r = |i: usize| row[i] as i32;
    let mut a0 = m(W4, r(0)).wrapping_add(1 << (ROW_SHIFT - 1));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 = a0.wrapping_add(m(W2, r(2)));
    a1 = a1.wrapping_add(m(W6, r(2)));
    a2 = a2.wrapping_sub(m(W6, r(2)));
    a3 = a3.wrapping_sub(m(W2, r(2)));
    let mut b0 = m(W1, r(1)).wrapping_add(m(W3, r(3)));
    let mut b1 = m(W3, r(1)).wrapping_add(m(-W7, r(3)));
    let mut b2 = m(W5, r(1)).wrapping_add(m(-W1, r(3)));
    let mut b3 = m(W7, r(1)).wrapping_add(m(-W5, r(3)));
    if row[4] != 0 || row[5] != 0 || row[6] != 0 || row[7] != 0 {
        a0 = a0.wrapping_add(m(W4, r(4))).wrapping_add(m(W6, r(6)));
        a1 = a1.wrapping_add(m(-W4, r(4))).wrapping_sub(m(W2, r(6)));
        a2 = a2.wrapping_add(m(-W4, r(4))).wrapping_add(m(W2, r(6)));
        a3 = a3.wrapping_add(m(W4, r(4))).wrapping_sub(m(W6, r(6)));
        b0 = b0.wrapping_add(m(W5, r(5))).wrapping_add(m(W7, r(7)));
        b1 = b1.wrapping_add(m(-W1, r(5))).wrapping_add(m(-W5, r(7)));
        b2 = b2.wrapping_add(m(W7, r(5))).wrapping_add(m(W3, r(7)));
        b3 = b3.wrapping_add(m(W3, r(5))).wrapping_add(m(-W1, r(7)));
    }
    row[0] = (a0.wrapping_add(b0) >> ROW_SHIFT) as i16;
    row[7] = (a0.wrapping_sub(b0) >> ROW_SHIFT) as i16;
    row[1] = (a1.wrapping_add(b1) >> ROW_SHIFT) as i16;
    row[6] = (a1.wrapping_sub(b1) >> ROW_SHIFT) as i16;
    row[2] = (a2.wrapping_add(b2) >> ROW_SHIFT) as i16;
    row[5] = (a2.wrapping_sub(b2) >> ROW_SHIFT) as i16;
    row[3] = (a3.wrapping_add(b3) >> ROW_SHIFT) as i16;
    row[4] = (a3.wrapping_sub(b3) >> ROW_SHIFT) as i16;
}

/// `IDCT_COLS` of the template: returns the eight column outputs (after
/// the final shift), in output row order.
fn idct_cols(b: &[i16], c: usize) -> [i32; 8] {
    let col = |r: usize| b[8 * r + c] as i32;
    let mut a0 = m(W4, col(0).wrapping_add((1 << (COL_SHIFT - 1)) / W4));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 = a0.wrapping_add(m(W2, col(2)));
    a1 = a1.wrapping_add(m(W6, col(2)));
    a2 = a2.wrapping_add(m(-W6, col(2)));
    a3 = a3.wrapping_add(m(-W2, col(2)));
    let mut b0 = m(W1, col(1));
    let mut b1 = m(W3, col(1));
    let mut b2 = m(W5, col(1));
    let mut b3 = m(W7, col(1));
    b0 = b0.wrapping_add(m(W3, col(3)));
    b1 = b1.wrapping_add(m(-W7, col(3)));
    b2 = b2.wrapping_add(m(-W1, col(3)));
    b3 = b3.wrapping_add(m(-W5, col(3)));
    if col(4) != 0 {
        a0 = a0.wrapping_add(m(W4, col(4)));
        a1 = a1.wrapping_add(m(-W4, col(4)));
        a2 = a2.wrapping_add(m(-W4, col(4)));
        a3 = a3.wrapping_add(m(W4, col(4)));
    }
    if col(5) != 0 {
        b0 = b0.wrapping_add(m(W5, col(5)));
        b1 = b1.wrapping_add(m(-W1, col(5)));
        b2 = b2.wrapping_add(m(W7, col(5)));
        b3 = b3.wrapping_add(m(W3, col(5)));
    }
    if col(6) != 0 {
        a0 = a0.wrapping_add(m(W6, col(6)));
        a1 = a1.wrapping_add(m(-W2, col(6)));
        a2 = a2.wrapping_add(m(W2, col(6)));
        a3 = a3.wrapping_add(m(-W6, col(6)));
    }
    if col(7) != 0 {
        b0 = b0.wrapping_add(m(W7, col(7)));
        b1 = b1.wrapping_add(m(-W5, col(7)));
        b2 = b2.wrapping_add(m(W3, col(7)));
        b3 = b3.wrapping_add(m(-W1, col(7)));
    }
    [
        a0.wrapping_add(b0) >> COL_SHIFT,
        a1.wrapping_add(b1) >> COL_SHIFT,
        a2.wrapping_add(b2) >> COL_SHIFT,
        a3.wrapping_add(b3) >> COL_SHIFT,
        a3.wrapping_sub(b3) >> COL_SHIFT,
        a2.wrapping_sub(b2) >> COL_SHIFT,
        a1.wrapping_sub(b1) >> COL_SHIFT,
        a0.wrapping_sub(b0) >> COL_SHIFT,
    ]
}

pub(crate) fn idct(input: &[[i16; 8]; 8]) -> [[i16; 8]; 8] {
    let mut block = [0i16; 64];
    for (row, source) in block.chunks_exact_mut(8).zip(input) {
        row.copy_from_slice(source);
        idct_row_cond_dc(row);
    }
    let mut output = [[0i16; 8]; 8];
    for c in 0..8 {
        let values = idct_cols(&block, c);
        for r in 0..8 { output[r][c] = values[r].clamp(-256, 255) as i16; }
    }
    output
}
