// Ported from FFmpeg (commit 2da55bf): libavcodec/mpeg12dec.c
// (mpeg_decode_motion, mpeg1_decode_block_inter,
// mpeg2_decode_block_non_intra, mpeg2_decode_block_intra, get_dmv,
// mpeg_decode_mb, mpeg_decode_slice, the slice loop of decode_chunks,
// mpeg_field_start's size check, slice_end), libavcodec/mpeg12dec.h
// (decode_dc, mpeg_get_qscale), libavcodec/mpeg12.c
// (ff_mpeg1_decode_block_intra, ff_init_2d_vlc_rl),
// libavcodec/mpegvideo_dec.c (ff_mpv_reconstruct_mb for MPEG-1/2),
// libavcodec/mpegvideo_motion.c (mpeg_motion_internal and the MPEG-1/2
// frame cases of mpv_motion_internal), libavcodec/hpeldsp.c (put and avg
// pixels), libavcodec/get_bits.h (the checked reader, get_xbits,
// skip_1stop_8data_bits, check_marker, GET_VLC, GET_RL_VLC) and
// libavutil/common.h (sign_extend).
// License: LGPL-2.1-or-later
// Copyright (c) 2000, 2001 Fabrice Bellard
// Copyright (c) 2002-2013 Michael Niedermayer <michaelni@gmx.at>
// Copyright (c) 2007 Aurelien Jacobs <aurel@gnuage.org>

//! MPEG-1/2 frame pictures decoded as FFmpeg decodes them, including when
//! they are damaged: a slice that fails stops at the macroblock FFmpeg's
//! parser stops at (it reads zeros past the data, takes a lost motion
//! code as 0xffff, wraps levels to 16 bits, skips the prediction of a
//! vector outside the reference), decoding goes on at the next slice,
//! and error_resilience conceals what is missing before the picture is
//! shown or used as a reference. 4:2:0 only.

use crate::error_resilience::{self, Er, ER_AC_END, ER_AC_ERROR, ER_DC_END, ER_DC_ERROR, ER_MV_END, ER_MV_ERROR};
use crate::ff_tables as t;
use crate::frame_assembly::FrameBuffer;
use crate::simple_idct::idct;

pub(crate) const PICT_I: u8 = 1;
pub(crate) const PICT_P: u8 = 2;
pub(crate) const PICT_B: u8 = 3;

// mb_type flags (mpegutils.h, mpeg12dec.h).
pub(crate) const MB_TYPE_INTRA: u32 = 1;
pub(crate) const MB_TYPE_16X16: u32 = 1 << 3;
const MB_TYPE_16X8: u32 = 1 << 4;
const MB_TYPE_8X16: u32 = 1 << 5;
const MB_TYPE_8X8: u32 = 1 << 6;
const MB_TYPE_INTERLACED: u32 = 1 << 7;
const MB_TYPE_ZERO_MV: u32 = 1 << 9;
const MB_TYPE_CBP: u32 = 1 << 10;
const MB_TYPE_QUANT: u32 = 1 << 11;
pub(crate) const MB_TYPE_FORWARD_MV: u32 = 1 << 12;
const MB_TYPE_BACKWARD_MV: u32 = 1 << 13;
const MB_TYPE_SKIP: u32 = 1 << 17;

/// ptype2mb_type and btype2mb_type (mpeg12.c).
const P_TYPES: [u32; 7] = [
    MB_TYPE_INTRA,
    MB_TYPE_FORWARD_MV | MB_TYPE_CBP | MB_TYPE_ZERO_MV | MB_TYPE_16X16,
    MB_TYPE_FORWARD_MV,
    MB_TYPE_FORWARD_MV | MB_TYPE_CBP,
    MB_TYPE_QUANT | MB_TYPE_INTRA,
    MB_TYPE_QUANT | MB_TYPE_FORWARD_MV | MB_TYPE_CBP | MB_TYPE_ZERO_MV | MB_TYPE_16X16,
    MB_TYPE_QUANT | MB_TYPE_FORWARD_MV | MB_TYPE_CBP,
];
const B_TYPES: [u32; 11] = [
    MB_TYPE_INTRA,
    MB_TYPE_BACKWARD_MV,
    MB_TYPE_BACKWARD_MV | MB_TYPE_CBP,
    MB_TYPE_FORWARD_MV,
    MB_TYPE_FORWARD_MV | MB_TYPE_CBP,
    MB_TYPE_FORWARD_MV | MB_TYPE_BACKWARD_MV,
    MB_TYPE_FORWARD_MV | MB_TYPE_BACKWARD_MV | MB_TYPE_CBP,
    MB_TYPE_QUANT | MB_TYPE_INTRA,
    MB_TYPE_QUANT | MB_TYPE_BACKWARD_MV | MB_TYPE_CBP,
    MB_TYPE_QUANT | MB_TYPE_FORWARD_MV | MB_TYPE_CBP,
    MB_TYPE_QUANT | MB_TYPE_FORWARD_MV | MB_TYPE_BACKWARD_MV | MB_TYPE_CBP,
];

/// IS_INTRA.
pub(crate) fn is_intra(mb_type: u32) -> bool {
    mb_type & 7 != 0
}

/// IS_INTER.
pub(crate) fn is_inter(mb_type: u32) -> bool {
    mb_type & (MB_TYPE_16X16 | MB_TYPE_16X8 | MB_TYPE_8X16 | MB_TYPE_8X8) != 0
}

pub(crate) const MV_DIR_FORWARD: u8 = 1;
pub(crate) const MV_DIR_BACKWARD: u8 = 2;

/// mv_type of a frame picture's macroblock.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MvType {
    M16x16,
    Field,
    Dmv,
}

// ───────────────────────── bits ─────────────────────────

/// get_bits.h's checked reader over a packet: zeros past its end (FFmpeg's
/// input padding), the position held 8 bits past the end.
#[derive(Clone)]
pub(crate) struct Gb<'a> {
    buf: &'a [u8],
    index: usize,
    size: usize,
}

impl<'a> Gb<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, index: 0, size: buf.len() * 8 }
    }

    /// The next `n` (at most 32) bits, not consumed.
    pub(crate) fn show(&self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let at = self.index >> 3;
        let w = match self.buf.get(at..at + 8) {
            Some(bytes) => u64::from_be_bytes(bytes.try_into().unwrap_or([0; 8])),
            None => (0..8).fold(0u64, |w, k| (w << 8) | u64::from(self.buf.get(at + k).copied().unwrap_or(0))),
        };
        ((w << (self.index & 7)) >> (64 - n)) as u32
    }

    /// SHOW_SBITS: the next `n` bits as a signed number.
    fn show_signed(&self, n: u32) -> i32 {
        ((self.show(n) << (32 - n)) as i32) >> (32 - n)
    }

    pub(crate) fn skip(&mut self, n: usize) {
        self.index = (self.index + n).min(self.size + 8);
    }

    pub(crate) fn get(&mut self, n: u32) -> u32 {
        let v = self.show(n);
        self.skip(n as usize);
        v
    }

    fn get_signed(&mut self, n: u32) -> i32 {
        let v = self.show_signed(n);
        self.skip(n as usize);
        v
    }

    pub(crate) fn get1(&mut self) -> u32 {
        self.get(1)
    }

    /// get_bits_left.
    pub(crate) fn left(&self) -> i64 {
        self.size as i64 - self.index as i64
    }

    pub(crate) fn count(&self) -> usize {
        self.index
    }

    fn align(&mut self) {
        let r = (8 - (self.index & 7)) & 7;
        self.skip(r);
    }

    /// get_xbits.
    fn get_xbits(&mut self, n: u32) -> i32 {
        let v = self.get(n) as i32;
        if (v >> (n - 1)) & 1 == 1 { v } else { v - ((1 << n) - 1) }
    }

    /// The next two bits are the end-of-block code `10`.
    fn at_eob(&self) -> bool {
        self.show(2) == 2
    }

    /// skip_1stop_8data_bits: false where FFmpeg returns an error.
    fn skip_1stop_8data_bits(&mut self) -> bool {
        if self.left() <= 0 {
            return false;
        }
        while self.get1() == 1 {
            self.skip(8);
            if self.left() <= 0 {
                return false;
            }
        }
        true
    }
}

const LEAF: u32 = 1 << 31;

/// A VLC as a two-level lookup table, read as get_vlc2 reads it: a code
/// the table lacks is None, and when its first `bits` bits already miss,
/// nothing is consumed.
pub(crate) struct Vlc {
    bits: u32,
    /// LEAF | symbol << 8 | length; or subtable offset << 8 | its bits; 0 lacks.
    table: Vec<u32>,
}

impl Vlc {
    fn new(codes: &[(u32, u8)], symbols: Option<&[u32]>, bits: u32) -> Self {
        let mut table = vec![0u32; 1 << bits];
        // The longest code under each long prefix sizes its subtable.
        let mut sub_bits = vec![0u32; 1 << bits];
        for &(code, len) in codes {
            let len = u32::from(len);
            if len > bits {
                let prefix = (code >> (len - bits)) as usize;
                sub_bits[prefix] = sub_bits[prefix].max(len - bits);
            }
        }
        for (prefix, &n) in sub_bits.iter().enumerate() {
            if n > 0 {
                table[prefix] = ((table.len() as u32) << 8) | n;
                table.extend(std::iter::repeat_n(0, 1 << n));
            }
        }
        for (i, &(code, len)) in codes.iter().enumerate() {
            let symbol = symbols.map_or(i as u32, |s| s[i]);
            let len = u32::from(len);
            if len <= bits {
                let first = (code << (bits - len)) as usize;
                for e in &mut table[first..first + (1 << (bits - len))] {
                    *e = LEAF | (symbol << 8) | len;
                }
            } else {
                let rest = len - bits;
                let prefix = (code >> rest) as usize;
                let (offset, n) = ((table[prefix] >> 8) as usize, table[prefix] & 0xFF);
                let first = offset + (((code & ((1 << rest) - 1)) << (n - rest)) as usize);
                for e in &mut table[first..first + (1 << (n - rest))] {
                    *e = LEAF | (symbol << 8) | rest;
                }
            }
        }
        Self { bits, table }
    }

    fn read(&self, gb: &mut Gb) -> Option<u32> {
        let e = self.table[gb.show(self.bits) as usize];
        if e & LEAF != 0 {
            gb.skip((e & 0xFF) as usize);
            return Some((e & !LEAF) >> 8);
        }
        if e == 0 {
            return None;
        }
        gb.skip(self.bits as usize);
        let e = self.table[(e >> 8) as usize + gb.show(e & 0xFF) as usize];
        if e & LEAF == 0 {
            return None;
        }
        gb.skip((e & 0xFF) as usize);
        Some((e & !LEAF) >> 8)
    }
}

struct Vlcs {
    dc_lum: Vlc,
    dc_chroma: Vlc,
    motion: Vlc,
    mbincr: Vlc,
    mb_pat: Vlc,
    ptype: Vlc,
    btype: Vlc,
    rl_b14: Vlc,
    rl_b15: Vlc,
}

/// mpeg12_init_vlcs. The motion table is one level as FFmpeg's misses
/// are: a lost code consumes nothing.
static VLCS: std::sync::LazyLock<Vlcs> = std::sync::LazyLock::new(|| Vlcs {
    dc_lum: Vlc::new(&t::DC_LUM, None, 9),
    dc_chroma: Vlc::new(&t::DC_CHROMA, None, 10),
    motion: Vlc::new(&t::MOTION, None, 10),
    mbincr: Vlc::new(&t::MB_ADDR_INCR, None, 11),
    mb_pat: Vlc::new(&t::MB_PAT, None, 9),
    ptype: Vlc::new(&t::MB_PTYPE, Some(&P_TYPES), 6),
    btype: Vlc::new(&t::MB_BTYPE, Some(&B_TYPES), 6),
    rl_b14: Vlc::new(&t::RL_B14, None, 10),
    rl_b15: Vlc::new(&t::RL_B15, None, 10),
});

/// GET_RL_VLC over ff_init_2d_vlc_rl's table: (level, run) with run one
/// more than the coded run; escape (0, 65); end of block (127, 0); a code
/// the table lacks (64, 65).
fn rl(vlc: &Vlc, gb: &mut Gb) -> (i32, i32) {
    match vlc.read(gb) {
        None => (64, 65),
        Some(111) => (0, 65),
        Some(112) => (127, 0),
        Some(s) => (i32::from(t::RL_LEVEL[s as usize]), i32::from(t::RL_RUN[s as usize]) + 1),
    }
}

/// sign_extend.
fn sign_extend(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((val as u32).wrapping_shl(shift) as i32) >> shift
}

/// avpriv_find_start_code from `p`: just past the start code's value
/// byte with the code, or the end of `buf` with None.
pub(crate) fn find_start_code(buf: &[u8], mut p: usize) -> (usize, Option<u32>) {
    let mut state: u32 = u32::MAX;
    while p < buf.len() {
        state = (state << 8) | u32::from(buf[p]);
        p += 1;
        if state & 0xFFFF_FF00 == 0x100 {
            return (p, Some(state));
        }
    }
    (p, None)
}

// ───────────────────────── picture ─────────────────────────

/// What a frame picture's decode needs from its headers and references.
pub(crate) struct Pic<'a> {
    pub mpeg2: bool,
    pub pict_type: u8,
    /// mpeg_f_code[dir][component], 0 already read as 1.
    pub f_code: [[u8; 2]; 2],
    pub full_pel: [bool; 2],
    pub intra_dc_precision: u8,
    pub frame_pred_frame_dct: bool,
    pub concealment_motion_vectors: bool,
    pub q_scale_type: bool,
    pub intra_vlc_format: bool,
    pub alternate_scan: bool,
    pub top_field_first: bool,
    pub progressive_sequence: bool,
    /// vertical_size.
    pub height: usize,
    pub mb_width: usize,
    pub mb_height: usize,
    /// Raster order (the C IDCT's permutation is the identity).
    pub intra_matrix: [u16; 64],
    pub inter_matrix: [u16; 64],
    pub chroma_intra_matrix: [u16; 64],
    pub chroma_inter_matrix: [u16; 64],
    /// last_pic: the forward reference.
    pub last: Option<&'a FrameBuffer>,
    /// next_pic of a B-picture: the backward reference. An I- or
    /// P-picture's next_pic is the picture itself.
    pub next: Option<&'a FrameBuffer>,
}

impl Pic<'_> {
    fn mb_stride(&self) -> usize {
        self.mb_width + 1
    }

    fn scan(&self) -> &'static [u8; 64] {
        if self.alternate_scan { &t::ALTERNATE_VERTICAL } else { &t::ZIGZAG }
    }
}

/// How a picture's decode ended.
pub(crate) enum Decoded {
    /// slice_end ran, concealment included.
    Complete(FrameBuffer),
    /// decode_chunks failed after the picture started ("slice too small",
    /// "slice below image"): not concealed and not shown, though an I- or
    /// P-picture already is the newest reference.
    Abandoned(FrameBuffer),
    /// No slice started the picture: nothing changes.
    NotStarted,
}

/// The picture under reconstruction with its macroblock types.
pub(crate) struct Cur {
    pub fb: FrameBuffer,
    pub mb_type: Vec<u32>,
}

/// Where FFmpeg's mpegvideo parser ends a frame picture's packet
/// (ff_mpeg1_find_frame_end): at the first start code after its first
/// slice that is not a slice's, a sequence end code included.
pub(crate) fn parsed_frame_end(region: &[u8]) -> usize {
    let (mut p, mut slices) = (0, false);
    loop {
        let (q, code) = find_start_code(region, p);
        let Some(code) = code else { return region.len() };
        if (0x101..=0x1AF).contains(&code) {
            slices = true;
        } else if slices {
            return if code == 0x1B7 { q } else { q - 4 };
        }
        p = q;
    }
}

/// decode_chunks over one picture: `packet` runs from its picture start
/// code to where the parser ends it ([`parsed_frame_end`]).
pub(crate) fn decode_picture(pic: &Pic, packet: &[u8], fresh: FrameBuffer) -> Result<Decoded, crate::Error> {
    let mb_stride = pic.mb_stride();
    // mbskip_table: concealment reads only what this picture wrote.
    let mut mbskip = vec![0u8; mb_stride * pic.mb_height];
    let mut cur = Cur { fb: fresh, mb_type: vec![0; mb_stride * pic.mb_height] };
    let mut er = Er::new(pic.mb_width, pic.mb_height);
    let mut slice = SliceState {
        macroblocks_left: 2 * pic.mb_width * pic.mb_height,
        ..SliceState::default()
    };
    let mut started = false;
    let mut p = 0usize;
    loop {
        let (q, code) = find_start_code(packet, p);
        p = q;
        let Some(code) = code else { break };
        if !(0x101..=0x1AF).contains(&code) {
            continue;
        }
        let mut mb_y = (code - 0x101) as usize;
        if pic.mpeg2 && pic.mb_height > 2800 / 16 {
            mb_y += usize::from(packet.get(p).copied().unwrap_or(0) & 0xE0) << 2;
        }
        if packet.len() - p < 2 || mb_y >= pic.mb_height {
            // "slice too small", "slice below image"
            return Ok(if started { Decoded::Abandoned(cur.fb) } else { Decoded::NotStarted });
        }
        if !started {
            started = true;
            // mpeg_field_start: too few bytes for the picture's size.
            if pic.mb_width * pic.mb_height * 11 / (33 * 2 * 8) > packet.len() {
                return Ok(Decoded::NotStarted);
            }
            er.frame_start();
        }
        let mut gb = Gb::new(&packet[p..]);
        match slice.decode_slice(pic, &mut gb, mb_y, &mut cur, &mut mbskip) {
            Ok(()) => {
                er.add_slice(slice.resync_mb_x, slice.resync_mb_y, slice.mb_x as i32 - 1, slice.mb_y as i32, ER_AC_END | ER_DC_END | ER_MV_END);
                p += (gb.count() - 1) / 8;
            }
            Err(()) => {
                if slice.exceeded_work {
                    return Err(crate::Error::InvalidBitstream("slice work exceeds the picture macroblock budget"));
                }
                if slice.resync_mb_x >= 0 && slice.resync_mb_y >= 0 {
                    er.add_slice(slice.resync_mb_x, slice.resync_mb_y, slice.mb_x as i32, slice.mb_y as i32, ER_AC_ERROR | ER_DC_ERROR | ER_MV_ERROR);
                }
            }
        }
    }
    if !started {
        return Ok(Decoded::NotStarted);
    }
    error_resilience::frame_end(&mut er, pic, &mut cur, &mbskip);
    Ok(Decoded::Complete(cur.fb))
}

// ───────────────────────── slices ─────────────────────────

/// Mpeg12SliceContext: what carries from macroblock to macroblock.
struct SliceState {
    mb_x: usize,
    mb_y: usize,
    resync_mb_x: i32,
    resync_mb_y: i32,
    qscale: i32,
    last_dc: [i32; 3],
    last_mv: [[[i32; 2]; 2]; 2],
    mv: [[[i32; 2]; 4]; 2],
    field_select: [[u8; 2]; 2],
    mv_dir: u8,
    mv_type: Option<MvType>,
    mb_intra: bool,
    mb_skipped: bool,
    interlaced_dct: bool,
    block: [[i16; 64]; 6],
    block_last_index: [i32; 6],
    // Per-picture budget, deliberately not reset at slice boundaries.
    macroblocks_left: usize,
    exceeded_work: bool,
}

impl Default for SliceState {
    fn default() -> Self {
        Self {
            mb_x: 0,
            mb_y: 0,
            resync_mb_x: -1,
            resync_mb_y: -1,
            qscale: 0,
            last_dc: [0; 3],
            last_mv: [[[0; 2]; 2]; 2],
            mv: [[[0; 2]; 4]; 2],
            field_select: [[0; 2]; 2],
            mv_dir: 0,
            mv_type: None,
            mb_intra: false,
            mb_skipped: false,
            interlaced_dct: false,
            block: [[0; 64]; 6],
            block_last_index: [-1; 6],
            macroblocks_left: 0,
            exceeded_work: false,
        }
    }
}

/// mpeg_get_qscale.
fn get_qscale(gb: &mut Gb, q_scale_type: bool) -> i32 {
    if q_scale_type { i32::from(t::NON_LINEAR_QSCALE[gb.get(5) as usize]) } else { (gb.get(5) as i32) << 1 }
}

/// decode_dc. Both DC tables are complete prefix codes.
fn decode_dc(gb: &mut Gb, component: usize) -> i32 {
    let vlc = if component == 0 { &VLCS.dc_lum } else { &VLCS.dc_chroma };
    match vlc.read(gb).unwrap_or(0) {
        0 => 0,
        size => gb.get_xbits(size),
    }
}

impl SliceState {
    /// mpeg_decode_motion.
    fn decode_motion(gb: &mut Gb, fcode: u8, pred: i32) -> i32 {
        let Some(code) = VLCS.motion.read(gb) else {
            return 0xffff;
        };
        if code == 0 {
            return pred;
        }
        let sign = gb.get1();
        let shift = u32::from(fcode) - 1;
        let mut val = code as i32;
        if shift != 0 {
            val = ((val - 1) << shift) | gb.get(shift) as i32;
            val += 1;
        }
        if sign != 0 {
            val = -val;
        }
        sign_extend(val.wrapping_add(pred), 5 + shift)
    }

    /// get_dmv.
    fn get_dmv(gb: &mut Gb) -> i32 {
        if gb.get1() != 0 { 1 - ((gb.get1() as i32) << 1) } else { 0 }
    }

    /// mpeg_decode_slice for a frame picture.
    fn decode_slice(&mut self, pic: &Pic, gb: &mut Gb, mb_y: usize, cur: &mut Cur, mbskip: &mut [u8]) -> Result<(), ()> {
        self.resync_mb_x = -1;
        self.resync_mb_y = -1;
        if pic.mpeg2 && pic.mb_height > 2800 / 16 {
            gb.skip(3);
        }
        self.interlaced_dct = false;
        self.qscale = get_qscale(gb, pic.q_scale_type);
        if self.qscale == 0 {
            return Err(());
        }
        if !gb.skip_1stop_8data_bits() {
            return Err(());
        }
        self.mb_x = 0;
        while gb.left() > 0 {
            let Some(code) = VLCS.mbincr.read(gb) else {
                return Err(()); // first mb_incr damaged
            };
            if code >= 33 {
                if code == 33 {
                    self.mb_x += 33;
                }
            } else {
                self.mb_x += code as usize;
                break;
            }
        }
        if self.mb_x >= pic.mb_width {
            return Err(()); // initial skip overflow
        }
        self.resync_mb_x = self.mb_x as i32;
        self.resync_mb_y = mb_y as i32;
        self.mb_y = mb_y;
        let dc = 128 << pic.intra_dc_precision;
        self.last_dc = [dc; 3];
        self.last_mv = [[[0; 2]; 2]; 2];

        let mut mb_skip_run: i32 = 0;
        loop {
            let Some(left) = self.macroblocks_left.checked_sub(1) else {
                self.exceeded_work = true;
                return Err(());
            };
            self.macroblocks_left = left;
            self.decode_mb(pic, gb, &mut mb_skip_run, cur)?;
            reconstruct_mb(self, pic, cur, mbskip);
            self.mb_x += 1;
            if self.mb_x >= pic.mb_width {
                self.mb_x = 0;
                self.mb_y += 1;
                if self.mb_y >= pic.mb_height {
                    let left = gb.left();
                    let mut d10 = false;
                    if left >= 32 {
                        let mut g = gb.clone();
                        g.align();
                        if g.show(24) == 0x060E2B {
                            d10 = true; // Invalid MXF data found in video stream
                        }
                        if left > 32 && g.show(32) == 0x201 {
                            break; // skipping m704 alpha
                        }
                    }
                    if left < 0 || (left > 0 && gb.show(left.min(23) as u32) != 0 && !d10) {
                        return Err(()); // end mismatch
                    }
                    break;
                }
                // Files missing their last slice outside the visible area.
                let left = gb.left();
                if self.mb_y >= pic.height.div_ceil(16)
                    && !pic.progressive_sequence
                    && (0..=25).contains(&left)
                    && mb_skip_run == -1
                    && (left == 0 || gb.show(left as u32) == 0)
                {
                    break;
                }
            }
            if mb_skip_run == -1 {
                mb_skip_run = 0;
                loop {
                    let Some(code) = VLCS.mbincr.read(gb) else {
                        return Err(()); // mb incr damaged
                    };
                    if code >= 33 {
                        if code == 33 {
                            mb_skip_run += 33;
                        } else if code == 35 {
                            if mb_skip_run != 0 || gb.show(15) != 0 {
                                return Err(()); // slice mismatch
                            }
                            return Self::end_of_slice(gb);
                        }
                    } else {
                        mb_skip_run += code as i32;
                        break;
                    }
                }
                if mb_skip_run != 0 {
                    if pic.pict_type == PICT_I {
                        return Err(()); // skipped MB in I-frame
                    }
                    self.mb_intra = false;
                    self.block_last_index = [-1; 6];
                    let dc = 128 << pic.intra_dc_precision;
                    self.last_dc = [dc; 3];
                    self.mv_type = Some(MvType::M16x16);
                    if pic.pict_type == PICT_P {
                        self.mv_dir = MV_DIR_FORWARD;
                        self.mv[0][0] = [0, 0];
                        self.last_mv[0][0] = [0, 0];
                        self.last_mv[0][1] = [0, 0];
                        self.field_select[0][0] = 0;
                    } else {
                        self.mv[0][0] = self.last_mv[0][0];
                        self.mv[1][0] = self.last_mv[1][0];
                        self.field_select[0][0] = 0;
                        self.field_select[1][0] = 0;
                    }
                }
            }
        }
        Self::end_of_slice(gb)
    }

    /// eos: an overread is an error.
    fn end_of_slice(gb: &Gb) -> Result<(), ()> {
        if gb.left() < 0 { Err(()) } else { Ok(()) }
    }

    /// mpeg_decode_mb.
    fn decode_mb(&mut self, pic: &Pic, gb: &mut Gb, mb_skip_run: &mut i32, cur: &mut Cur) -> Result<(), ()> {
        let mb_stride = pic.mb_stride();
        let mb_xy = self.mb_x + self.mb_y * mb_stride;
        let run = *mb_skip_run;
        *mb_skip_run -= 1;
        if run != 0 {
            if pic.pict_type == PICT_P {
                self.mb_skipped = true;
                cur.mb_type[mb_xy] = MB_TYPE_SKIP | MB_TYPE_FORWARD_MV | MB_TYPE_16X16;
            } else {
                // A skip run reaches a row start only from the row above.
                let previous = if self.mb_x > 0 { mb_xy - 1 } else { pic.mb_width + (self.mb_y - 1) * mb_stride - 1 };
                let mb_type = cur.mb_type[previous];
                if is_intra(mb_type) {
                    return Err(()); // skip with previntra
                }
                cur.mb_type[mb_xy] = mb_type | MB_TYPE_SKIP;
                if (self.mv[0][0][0] | self.mv[0][0][1] | self.mv[1][0][0] | self.mv[1][0][1]) == 0 {
                    self.mb_skipped = true;
                }
            }
            return Ok(());
        }

        let mut mb_type = match pic.pict_type {
            PICT_P => VLCS.ptype.read(gb).ok_or(())?,
            PICT_B => VLCS.btype.read(gb).ok_or(())?,
            _ => {
                if gb.get1() == 0 {
                    if gb.get1() == 0 {
                        return Err(()); // invalid mb type in I-frame
                    }
                    MB_TYPE_QUANT | MB_TYPE_INTRA
                } else {
                    MB_TYPE_INTRA
                }
            }
        };
        if is_intra(mb_type) {
            self.block = [[0; 64]; 6];
            if !pic.frame_pred_frame_dct {
                self.interlaced_dct = gb.get1() != 0;
            }
            if mb_type & MB_TYPE_QUANT != 0 {
                self.qscale = get_qscale(gb, pic.q_scale_type);
            }
            if pic.concealment_motion_vectors {
                let x = Self::decode_motion(gb, pic.f_code[0][0], self.last_mv[0][0][0]);
                self.mv[0][0][0] = x;
                self.last_mv[0][0][0] = x;
                self.last_mv[0][1][0] = x;
                let y = Self::decode_motion(gb, pic.f_code[0][1], self.last_mv[0][0][1]);
                self.mv[0][0][1] = y;
                self.last_mv[0][0][1] = y;
                self.last_mv[0][1][1] = y;
                gb.skip(1); // check_marker
            } else {
                self.last_mv = [[[0; 2]; 2]; 2];
            }
            self.mb_intra = true;
            for i in 0..6 {
                if pic.mpeg2 {
                    self.mpeg2_block_intra(pic, gb, i)?;
                } else {
                    self.mpeg1_block_intra(pic, gb, i)?;
                }
            }
        } else {
            if mb_type & MB_TYPE_ZERO_MV != 0 {
                self.mv_dir = MV_DIR_FORWARD;
                if !pic.frame_pred_frame_dct {
                    self.interlaced_dct = gb.get1() != 0;
                }
                self.mv_type = Some(MvType::M16x16);
                if mb_type & MB_TYPE_QUANT != 0 {
                    self.qscale = get_qscale(gb, pic.q_scale_type);
                }
                self.last_mv[0][0] = [0, 0];
                self.last_mv[0][1] = [0, 0];
                self.mv[0][0] = [0, 0];
            } else {
                let motion_type = if pic.frame_pred_frame_dct {
                    2
                } else {
                    let mt = gb.get(2);
                    if mb_type & MB_TYPE_CBP != 0 {
                        self.interlaced_dct = gb.get1() != 0;
                    }
                    mt
                };
                if mb_type & MB_TYPE_QUANT != 0 {
                    self.qscale = get_qscale(gb, pic.q_scale_type);
                }
                self.mv_dir = ((mb_type >> 12) & 3) as u8;
                match motion_type {
                    2 => {
                        mb_type |= MB_TYPE_16X16;
                        self.mv_type = Some(MvType::M16x16);
                        for i in 0..2 {
                            if mb_type & (MB_TYPE_FORWARD_MV << i) != 0 {
                                for k in 0..2 {
                                    let v = Self::decode_motion(gb, pic.f_code[i][k], self.last_mv[i][0][k]);
                                    self.mv[i][0][k] = v;
                                    self.last_mv[i][0][k] = v;
                                    self.last_mv[i][1][k] = v;
                                }
                                if pic.full_pel[i] {
                                    self.mv[i][0][0] = self.mv[i][0][0].wrapping_mul(2);
                                    self.mv[i][0][1] = self.mv[i][0][1].wrapping_mul(2);
                                }
                            }
                        }
                    }
                    1 => {
                        self.mv_type = Some(MvType::Field);
                        mb_type |= MB_TYPE_16X8 | MB_TYPE_INTERLACED;
                        for i in 0..2 {
                            if mb_type & (MB_TYPE_FORWARD_MV << i) != 0 {
                                for j in 0..2 {
                                    self.field_select[i][j] = gb.get1() as u8;
                                    let x = Self::decode_motion(gb, pic.f_code[i][0], self.last_mv[i][j][0]);
                                    self.last_mv[i][j][0] = x;
                                    self.mv[i][j][0] = x;
                                    let y = Self::decode_motion(gb, pic.f_code[i][1], self.last_mv[i][j][1] >> 1);
                                    self.last_mv[i][j][1] = y.wrapping_mul(2);
                                    self.mv[i][j][1] = y;
                                }
                            }
                        }
                    }
                    3 => {
                        if pic.progressive_sequence {
                            return Err(()); // MT_DMV in progressive_sequence
                        }
                        self.mv_type = Some(MvType::Dmv);
                        for i in 0..2 {
                            if mb_type & (MB_TYPE_FORWARD_MV << i) != 0 {
                                let mx = Self::decode_motion(gb, pic.f_code[i][0], self.last_mv[i][0][0]);
                                self.last_mv[i][0][0] = mx;
                                self.last_mv[i][1][0] = mx;
                                let dmx = Self::get_dmv(gb);
                                let my = Self::decode_motion(gb, pic.f_code[i][1], self.last_mv[i][0][1] >> 1);
                                let dmy = Self::get_dmv(gb);
                                self.last_mv[i][0][1] = my.wrapping_mul(2);
                                self.last_mv[i][1][1] = my.wrapping_mul(2);
                                self.mv[i][0] = [mx, my];
                                self.mv[i][1] = [mx, my];
                                mb_type |= MB_TYPE_16X16 | MB_TYPE_INTERLACED;
                                let pos = |v: i32| i32::from(v > 0);
                                let m = if pic.top_field_first { 1 } else { 3 };
                                self.mv[i][2][0] = ((mx.wrapping_mul(m) + pos(mx)) >> 1) + dmx;
                                self.mv[i][2][1] = ((my.wrapping_mul(m) + pos(my)) >> 1) + dmy - 1;
                                let m = 4 - m;
                                self.mv[i][3][0] = ((mx.wrapping_mul(m) + pos(mx)) >> 1) + dmx;
                                self.mv[i][3][1] = ((my.wrapping_mul(m) + pos(my)) >> 1) + dmy + 1;
                            }
                        }
                    }
                    _ => return Err(()), // 00 motion_type
                }
            }
            self.mb_intra = false;
            let dc = 128 << pic.intra_dc_precision;
            self.last_dc = [dc; 3];
            if mb_type & MB_TYPE_CBP != 0 {
                self.block = [[0; 64]; 6];
                let cbp = VLCS.mb_pat.read(gb).map_or(-1, |c| c as i32);
                if cbp <= 0 {
                    return Err(()); // invalid cbp
                }
                if pic.mpeg2 {
                    let mut cbp = cbp << 6;
                    for i in 0..6 {
                        if cbp & (1 << 11) != 0 {
                            self.mpeg2_block_non_intra(pic, gb, i)?;
                        } else {
                            self.block_last_index[i] = -1;
                        }
                        cbp += cbp;
                    }
                } else {
                    let mut cbp = cbp;
                    for i in 0..6 {
                        if cbp & 32 != 0 {
                            self.mpeg1_block_inter(pic, gb, i)?;
                        } else {
                            self.block_last_index[i] = -1;
                        }
                        cbp += cbp;
                    }
                }
            } else {
                self.block_last_index = [-1; 6];
            }
        }
        cur.mb_type[mb_xy] = mb_type;
        Ok(())
    }

    /// The sign bit after a coefficient code, applied as FFmpeg does.
    fn signed(gb: &mut Gb, level: i32) -> i32 {
        let s = gb.show_signed(1);
        gb.skip(1);
        (level ^ s) - s
    }

    /// mpeg1_decode_block_inter.
    fn mpeg1_block_inter(&mut self, pic: &Pic, gb: &mut Gb, n: usize) -> Result<(), ()> {
        let scan = pic.scan();
        let qm = &pic.inter_matrix;
        let q = self.qscale;
        let block = &mut self.block[n];
        let mut i: i32 = -1;
        let mut done = false;
        if gb.show(1) == 1 {
            let mut level = (3 * q * i32::from(qm[0])) >> 5;
            level = (level - 1) | 1;
            if gb.show(2) & 1 == 1 {
                level = -level;
            }
            block[0] = level as i16;
            i += 1;
            gb.skip(2);
            done = gb.at_eob();
        }
        if !done {
            loop {
                let (mut level, run) = rl(&VLCS.rl_b14, gb);
                let j;
                if level != 0 {
                    i += run;
                    if i > 63 {
                        break;
                    }
                    j = usize::from(scan[i as usize]);
                    level = ((level * 2 + 1) * q).wrapping_mul(i32::from(qm[j])) >> 5;
                    level = (level - 1) | 1;
                    level = Self::signed(gb, level);
                } else {
                    let run = gb.get(6) as i32 + 1;
                    level = gb.get_signed(8);
                    if level == -128 {
                        level = gb.get(8) as i32 - 256;
                    } else if level == 0 {
                        level = gb.get(8) as i32;
                    }
                    i += run;
                    if i > 63 {
                        break;
                    }
                    j = usize::from(scan[i as usize]);
                    let magnitude = ((level.abs() * 2 + 1) * q).wrapping_mul(i32::from(qm[j])) >> 5;
                    level = if level < 0 { -((magnitude - 1) | 1) } else { (magnitude - 1) | 1 };
                }
                block[j] = level as i16;
                if gb.at_eob() {
                    break;
                }
            }
        }
        gb.skip(2);
        if i > 63 {
            return Err(()); // ac-tex damaged
        }
        self.block_last_index[n] = i;
        Ok(())
    }

    /// mpeg2_decode_block_non_intra.
    fn mpeg2_block_non_intra(&mut self, pic: &Pic, gb: &mut Gb, n: usize) -> Result<(), ()> {
        let scan = pic.scan();
        let qm = if n < 4 { &pic.inter_matrix } else { &pic.chroma_inter_matrix };
        let q = self.qscale;
        let block = &mut self.block[n];
        let mut mismatch: i32 = 1;
        let mut i: i32 = -1;
        let mut done = false;
        if gb.show(1) == 1 {
            let mut level = (3 * q * i32::from(qm[0])) >> 5;
            if gb.show(2) & 1 == 1 {
                level = -level;
            }
            block[0] = level as i16;
            mismatch ^= level;
            i += 1;
            gb.skip(2);
            done = gb.at_eob();
        }
        if !done {
            loop {
                let (mut level, run) = rl(&VLCS.rl_b14, gb);
                let j;
                if level != 0 {
                    i += run;
                    if i > 63 {
                        break;
                    }
                    j = usize::from(scan[i as usize]);
                    level = ((level * 2 + 1) * q).wrapping_mul(i32::from(qm[j])) >> 5;
                    level = Self::signed(gb, level);
                } else {
                    let run = gb.get(6) as i32 + 1;
                    level = gb.get_signed(12);
                    i += run;
                    if i > 63 {
                        break;
                    }
                    j = usize::from(scan[i as usize]);
                    let magnitude = ((level.abs() * 2 + 1) * q).wrapping_mul(i32::from(qm[j])) >> 5;
                    level = if level < 0 { -magnitude } else { magnitude };
                }
                mismatch ^= level;
                block[j] = level as i16;
                if gb.at_eob() {
                    break;
                }
            }
        }
        gb.skip(2);
        block[63] ^= (mismatch & 1) as i16;
        if i > 63 {
            return Err(()); // ac-tex damaged
        }
        self.block_last_index[n] = i;
        Ok(())
    }

    /// mpeg2_decode_block_intra.
    fn mpeg2_block_intra(&mut self, pic: &Pic, gb: &mut Gb, n: usize) -> Result<(), ()> {
        let scan = pic.scan();
        let (qm, component) = if n < 4 { (&pic.intra_matrix, 0) } else { (&pic.chroma_intra_matrix, (n & 1) + 1) };
        let diff = decode_dc(gb, component);
        let dc = self.last_dc[component].wrapping_add(diff);
        self.last_dc[component] = dc;
        let q = self.qscale;
        let block = &mut self.block[n];
        block[0] = dc.wrapping_mul(1 << (3 - pic.intra_dc_precision.min(3))) as i16;
        let mut mismatch = i32::from(block[0]) ^ 1;
        let mut i: i32 = 0;
        let vlc = if pic.intra_vlc_format { &VLCS.rl_b15 } else { &VLCS.rl_b14 };
        loop {
            let (mut level, run) = rl(vlc, gb);
            let j;
            if level == 127 {
                break;
            } else if level != 0 {
                i += run;
                if i > 63 {
                    break;
                }
                j = usize::from(scan[i as usize]);
                level = (level * q).wrapping_mul(i32::from(qm[j])) >> 4;
                level = Self::signed(gb, level);
            } else {
                let run = gb.get(6) as i32 + 1;
                level = gb.get_signed(12);
                i += run;
                if i > 63 {
                    break;
                }
                j = usize::from(scan[i as usize]);
                let magnitude = (level.abs() * q).wrapping_mul(i32::from(qm[j])) >> 4;
                level = if level < 0 { -magnitude } else { magnitude };
            }
            mismatch ^= level;
            block[j] = level as i16;
        }
        block[63] ^= (mismatch & 1) as i16;
        if i > 63 {
            return Err(()); // ac-tex damaged
        }
        Ok(())
    }

    /// ff_mpeg1_decode_block_intra.
    fn mpeg1_block_intra(&mut self, pic: &Pic, gb: &mut Gb, n: usize) -> Result<(), ()> {
        let scan = pic.scan();
        let qm = &pic.intra_matrix;
        let component = if n <= 3 { 0 } else { n - 4 + 1 };
        let diff = decode_dc(gb, component);
        let dc = self.last_dc[component].wrapping_add(diff);
        self.last_dc[component] = dc;
        let q = self.qscale;
        let block = &mut self.block[n];
        block[0] = dc.wrapping_mul(i32::from(qm[0])) as i16;
        let mut i: i32 = 0;
        if !gb.at_eob() {
            loop {
                let (mut level, run) = rl(&VLCS.rl_b14, gb);
                let j;
                if level != 0 {
                    i += run;
                    if i > 63 {
                        break;
                    }
                    j = usize::from(scan[i as usize]);
                    level = (level * q).wrapping_mul(i32::from(qm[j])) >> 4;
                    level = (level - 1) | 1;
                    level = Self::signed(gb, level);
                } else {
                    let run = gb.get(6) as i32 + 1;
                    level = gb.get_signed(8);
                    if level == -128 {
                        level = gb.get(8) as i32 - 256;
                    } else if level == 0 {
                        level = gb.get(8) as i32;
                    }
                    i += run;
                    if i > 63 {
                        break;
                    }
                    j = usize::from(scan[i as usize]);
                    let magnitude = (level.abs() * q).wrapping_mul(i32::from(qm[j])) >> 4;
                    level = if level < 0 { -((magnitude - 1) | 1) } else { (magnitude - 1) | 1 };
                }
                block[j] = level as i16;
                if gb.at_eob() {
                    break;
                }
            }
        }
        gb.skip(2);
        if i > 63 {
            return Err(()); // ac-tex damaged
        }
        Ok(())
    }
}

// ───────────────────────── reconstruction ─────────────────────────

/// A macroblock's prediction as ff_mpv_reconstruct_mb reads it.
pub(crate) struct MbPrediction<'m> {
    pub mv_dir: u8,
    pub mv_type: MvType,
    pub mv: &'m [[[i32; 2]; 4]; 2],
    pub field_select: &'m [[u8; 2]; 2],
}

/// ff_mpv_reconstruct_mb for the decoded macroblock at the slice position.
fn reconstruct_mb(s: &mut SliceState, pic: &Pic, cur: &mut Cur, mbskip: &mut [u8]) {
    let (mb_x, mb_y) = (s.mb_x, s.mb_y);
    // mbskip_table: a skipped macroblock, or any of a picture that is
    // not a reference.
    mbskip[mb_x + mb_y * pic.mb_stride()] = u8::from(s.mb_skipped || pic.pict_type == PICT_B);
    s.mb_skipped = false;
    let linesize = cur.fb.y.width();
    let uvlinesize = cur.fb.cb.width();
    let dest_y = mb_y * 16 * linesize + mb_x * 16;
    let dest_c = mb_y * 8 * uvlinesize + mb_x * 8;
    let (dct_linesize, dct_offset) = if s.interlaced_dct { (linesize * 2, linesize) } else { (linesize, linesize * 8) };
    let luma = [dest_y, dest_y + 8, dest_y + dct_offset, dest_y + dct_offset + 8];
    if !s.mb_intra {
        let prediction = MbPrediction {
            mv_dir: s.mv_dir,
            mv_type: s.mv_type.unwrap_or(MvType::M16x16),
            mv: &s.mv,
            field_select: &s.field_select,
        };
        motion(pic, &mut cur.fb, mb_x, mb_y, &prediction);
        for (i, &d) in luma.iter().enumerate() {
            if s.block_last_index[i] >= 0 {
                idct_add(cur.fb.y.samples_mut(), d, dct_linesize, &s.block[i]);
            }
        }
        if s.block_last_index[4] >= 0 {
            idct_add(cur.fb.cb.samples_mut(), dest_c, uvlinesize, &s.block[4]);
        }
        if s.block_last_index[5] >= 0 {
            idct_add(cur.fb.cr.samples_mut(), dest_c, uvlinesize, &s.block[5]);
        }
    } else {
        for (i, &d) in luma.iter().enumerate() {
            idct_put(cur.fb.y.samples_mut(), d, dct_linesize, &s.block[i]);
        }
        idct_put(cur.fb.cb.samples_mut(), dest_c, uvlinesize, &s.block[4]);
        idct_put(cur.fb.cr.samples_mut(), dest_c, uvlinesize, &s.block[5]);
    }
}

fn idct8(block: &[i16; 64]) -> [[i16; 8]; 8] {
    let mut rows = [[0i16; 8]; 8];
    for (r, row) in rows.iter_mut().enumerate() {
        row.copy_from_slice(&block[r * 8..r * 8 + 8]);
    }
    idct(&rows)
}

/// idct_put: the transform clipped to 0..=255.
fn idct_put(dst: &mut [u8], at: usize, stride: usize, block: &[i16; 64]) {
    let out = idct8(block);
    for (r, row) in out.iter().enumerate() {
        let o = at + r * stride;
        if let Some(d) = dst.get_mut(o..o + 8) {
            for (p, &v) in d.iter_mut().zip(row) {
                *p = v.clamp(0, 255) as u8;
            }
        }
    }
}

/// idct_add: the transform added to the prediction, clipped.
fn idct_add(dst: &mut [u8], at: usize, stride: usize, block: &[i16; 64]) {
    let out = idct8(block);
    for (r, row) in out.iter().enumerate() {
        let o = at + r * stride;
        if let Some(d) = dst.get_mut(o..o + 8) {
            for (p, &v) in d.iter_mut().zip(row) {
                *p = (i32::from(*p) + i32::from(v)).clamp(0, 255) as u8;
            }
        }
    }
}

/// The forward and backward predictions of a macroblock (put, then avg):
/// mpv_reconstruct_mb_internal's motion handling.
pub(crate) fn motion(pic: &Pic, cur: &mut FrameBuffer, mb_x: usize, mb_y: usize, p: &MbPrediction) {
    let mut avg = false;
    if p.mv_dir & MV_DIR_FORWARD != 0 {
        if let Some(last) = pic.last {
            motion_dir(pic, cur, last, mb_x, mb_y, p, 0, avg);
        }
        avg = true;
    }
    if p.mv_dir & MV_DIR_BACKWARD != 0 {
        // An I- or P-picture's next_pic is itself: concealment's zero
        // vector from it leaves the block as it is.
        if let (Some(next), PICT_B) = (pic.next, pic.pict_type) {
            motion_dir(pic, cur, next, mb_x, mb_y, p, 1, avg);
        }
    }
}

/// mpv_motion_internal for a frame picture.
#[allow(clippy::too_many_arguments)]
fn motion_dir(pic: &Pic, cur: &mut FrameBuffer, refp: &FrameBuffer, mb_x: usize, mb_y: usize, p: &MbPrediction, dir: usize, avg: bool) {
    match p.mv_type {
        MvType::M16x16 => mpeg_motion(pic, cur, refp, mb_x, mb_y, false, false, false, p.mv[dir][0], 16, avg),
        MvType::Field => {
            for f in 0..2 {
                mpeg_motion(pic, cur, refp, mb_x, mb_y, true, f == 1, p.field_select[dir][f] != 0, p.mv[dir][f], 8, avg);
            }
        }
        MvType::Dmv => {
            let mut a = avg;
            for i in 0..2 {
                for j in 0..2 {
                    mpeg_motion(pic, cur, refp, mb_x, mb_y, true, j == 1, (j ^ i) == 1, p.mv[dir][2 * i + j], 8, a);
                }
                a = true;
            }
        }
    }
}

/// mpeg_motion_internal (MPEG-1/2, 4:2:0, frame picture): a vector that
/// leaves the reference skips the prediction ("MPEG motion vector out of
/// boundary").
#[allow(clippy::too_many_arguments)]
fn mpeg_motion(
    pic: &Pic,
    cur: &mut FrameBuffer,
    refp: &FrameBuffer,
    mb_x: usize,
    mb_y: usize,
    field_based: bool,
    bottom_field: bool,
    field_select: bool,
    mv: [i32; 2],
    h: i32,
    avg: bool,
) {
    let fb = usize::from(field_based);
    let (motion_x, motion_y) = (mv[0], mv[1]);
    let linesize = cur.y.width();
    let uvlinesize = cur.cb.width();
    let h_edge_pos = (16 * pic.mb_width) as i32;
    let v_edge_pos = ((16 * pic.mb_height) >> fb) as i32;
    let ls = linesize << fb;
    let uvls = uvlinesize << fb;
    let dxy = ((motion_y & 1) << 1) | (motion_x & 1);
    let src_x = (mb_x as i32 * 16).wrapping_add(motion_x >> 1);
    let src_y = ((mb_y as i32) << (4 - fb)).wrapping_add(motion_y >> 1);
    let mx = motion_x / 2;
    let my = motion_y / 2;
    let uvdxy = ((my & 1) << 1) | (mx & 1);
    let uvsrc_x = (mb_x as i32 * 8).wrapping_add(mx >> 1);
    let uvsrc_y = ((mb_y as i32) << (3 - fb)).wrapping_add(my >> 1);
    if src_x as u32 >= (h_edge_pos - (motion_x & 1) - 15).max(0) as u32
        || src_y as u32 >= (v_edge_pos - (motion_y & 1) - h + 1).max(0) as u32
    {
        return;
    }
    let mut dy = mb_y * 16 * linesize + mb_x * 16;
    let mut dc = mb_y * 8 * uvlinesize + mb_x * 8;
    if bottom_field {
        dy += linesize;
        dc += uvlinesize;
    }
    let mut sy = src_y as usize * ls + src_x as usize;
    let (Ok(ux), Ok(uy)) = (usize::try_from(uvsrc_x), usize::try_from(uvsrc_y)) else {
        return;
    };
    let mut sc = uy * uvls + ux;
    if field_select {
        sy += linesize;
        sc += uvlinesize;
    }
    pixels(cur.y.samples_mut(), dy, ls, refp.y.samples(), sy, 16, h as usize, dxy, avg);
    pixels(cur.cb.samples_mut(), dc, uvls, refp.cb.samples(), sc, 8, (h >> 1) as usize, uvdxy, avg);
    pixels(cur.cr.samples_mut(), dc, uvls, refp.cr.samples(), sc, 8, (h >> 1) as usize, uvdxy, avg);
}

/// hpeldsp put/avg_pixels{16,8}{,_x2,_y2,_xy2}: rounded half-sample
/// averages, then a rounded average with the destination for avg. Both
/// pictures share the stride `stride`.
#[allow(clippy::too_many_arguments)]
fn pixels(dst: &mut [u8], d: usize, stride: usize, src: &[u8], s: usize, w: usize, h: usize, dxy: i32, avg: bool) {
    let (dx, dy) = (usize::from(dxy & 1 != 0), usize::from(dxy & 2 != 0));
    if h == 0 || s + (h - 1 + dy) * stride + w + dx > src.len() || d + (h - 1) * stride + w > dst.len() {
        return;
    }
    for y in 0..h {
        let r = s + y * stride;
        let o = d + y * stride;
        for x in 0..w {
            let a = u32::from(src[r + x]);
            let v = match dxy {
                0 => a,
                1 => (a + u32::from(src[r + x + 1]) + 1) >> 1,
                2 => (a + u32::from(src[r + x + stride]) + 1) >> 1,
                _ => (a + u32::from(src[r + x + 1]) + u32::from(src[r + x + stride]) + u32::from(src[r + x + stride + 1]) + 2) >> 2,
            };
            let p = &mut dst[o + x];
            *p = if avg { ((u32::from(*p) + v + 1) >> 1) as u8 } else { v as u8 };
        }
    }
}
