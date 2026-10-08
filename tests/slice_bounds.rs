//! A picture's slices cannot expand decode work or memory past its
//! macroblock grid. Allocation is measured per thread, so the bound is
//! observed rather than inferred from error text.
use oxideav_core::bits::BitWriter;
use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_mpeg12video::picture_header::PictureCodingType;
use oxideav_mpeg12video::sequence_extension::ChromaFormat;
use oxideav_mpeg12video::stream_writer::{
    write_picture_coding_extension, write_picture_header, write_sequence_extension,
    write_sequence_header, write_slice_header, PictureCodingExtensionParams, SequenceHeaderParams,
};
use oxideav_mpeg12video::{write_mpeg1_sequence_header, Mpeg12Decoder, Mpeg1SequenceParams};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Tracking;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
    static TOTAL: Cell<usize> = const { Cell::new(0) };
}

fn track(delta: isize) {
    let _ = LIVE.try_with(|live| {
        let now = live.get() + delta;
        live.set(now);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
    if delta > 0 {
        let _ = TOTAL.try_with(|total| total.set(total.get() + delta as usize));
    }
}

// SAFETY: every call forwards to `System` unchanged; only counters are added.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            track(layout.size() as isize);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        track(-(layout.size() as isize));
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = System.realloc(p, layout, size);
        if !q.is_null() {
            track(size as isize - layout.size() as isize);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

/// Run `f` and report its peak extra live bytes and total bytes allocated
/// on this thread.
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let base = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(base));
    let total = TOTAL.with(Cell::get);
    let value = f();
    let peak = PEAK.with(Cell::get) - base;
    (value, peak.max(0) as usize, TOTAL.with(Cell::get) - total)
}

/// One intra macroblock carrying DC-only blocks: address increment 1,
/// intra `macroblock_type`, then per block DC size 0 and end_of_block.
/// The codes are identical in ISO/IEC 11172-2 and 13818-2 at 4:2:0.
fn dc_only_intra_macroblock(bw: &mut BitWriter) {
    bw.write_bit(true);
    bw.write_bit(true);
    for _ in 0..4 {
        bw.write_u32(0b100, 3);
        bw.write_u32(0b10, 2);
    }
    for _ in 0..2 {
        bw.write_u32(0b00, 2);
        bw.write_u32(0b10, 2);
    }
}

/// A 16×16 intra picture (one macroblock) whose slices all start at row 0
/// and carry the given macroblock counts.
fn intra_picture_16x16(mpeg1: bool, slices: &[usize]) -> Vec<u8> {
    let mut bw = BitWriter::new();
    if mpeg1 {
        let seq = Mpeg1SequenceParams { horizontal_size: 16, vertical_size: 16, ..Default::default() };
        write_mpeg1_sequence_header(&mut bw, &seq).unwrap();
    } else {
        let seq = SequenceHeaderParams { horizontal_size: 16, vertical_size: 16, ..Default::default() };
        write_sequence_header(&mut bw, &seq);
        write_sequence_extension(&mut bw, ChromaFormat::Yuv420, true);
    }
    write_picture_header(&mut bw, 0, PictureCodingType::Intra, 7, 7);
    if !mpeg1 {
        let ext = PictureCodingExtensionParams { progressive_frame: true, ..Default::default() };
        write_picture_coding_extension(&mut bw, &ext);
    }
    for &count in slices {
        write_slice_header(&mut bw, 0, 8);
        for _ in 0..count {
            dc_only_intra_macroblock(&mut bw);
        }
        bw.align_to_byte_zero();
    }
    let mut stream = bw.finish();
    stream.extend_from_slice(&[0, 0, 1, 0xB7]);
    stream
}

fn decode(mpeg1: bool, stream: Vec<u8>) -> (oxideav_core::Result<oxideav_core::Frame>, usize, usize) {
    let mut dec = Mpeg12Decoder::new(CodecId::new(if mpeg1 { "mpeg1video" } else { "mpeg2video" }));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream)).unwrap();
    dec.flush().unwrap();
    let (result, peak, total) = measured(|| dec.receive_frame());
    (result, peak, total)
}

#[test]
fn a_slice_cannot_expand_past_the_picture_grid() {
    for mpeg1 in [false, true] {
        // A legal picture is one macroblock; the slice carries 100,000.
        let (result, peak, _) = decode(mpeg1, intra_picture_16x16(mpeg1, &[100_000]));
        let oxideav_core::Frame::Video(frame) = result.unwrap() else { panic!("video") };
        assert_eq!(frame.planes.len(), 3);
        for (plane, length) in frame.planes.iter().zip([256, 64, 64]) {
            assert_eq!(plane.data.len(), length);
        }
        assert!(frame.planes.iter().all(|p| p.data.iter().all(|&v| v == 128)), "mpeg1={mpeg1}");
        // Extra macroblocks are not reconstructed; the one-picture result
        // survives concealment without expanding the retained working set.
        assert!(peak < 1 << 20, "mpeg1={mpeg1}: peak {peak} bytes");
    }
}

#[test]
fn repeated_slices_cannot_decode_the_picture_again_and_again() {
    for mpeg1 in [false, true] {
        let (result, _, total) = decode(mpeg1, intra_picture_16x16(mpeg1, &vec![1; 100_000]));
        assert!(result.is_err(), "mpeg1={mpeg1}: duplicate coverage must be bounded");
        // At most twice the picture grid may be attempted across all slices.
        assert!(total < 8 << 20, "mpeg1={mpeg1}: {total} bytes allocated across slices");
    }
}

#[test]
fn a_single_complete_slice_still_decodes() {
    for mpeg1 in [false, true] {
        let mut dec = Mpeg12Decoder::new(CodecId::new(if mpeg1 { "mpeg1video" } else { "mpeg2video" }));
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), intra_picture_16x16(mpeg1, &[1]))).unwrap();
        dec.flush().unwrap();
        let oxideav_core::Frame::Video(frame) = dec.receive_frame().unwrap() else { panic!("video") };
        // DC-only blocks at the reset predictor: flat mid-grey.
        assert!(frame.planes.iter().all(|p| p.data.iter().all(|&v| v == 128)), "mpeg1={mpeg1}");
    }
}
