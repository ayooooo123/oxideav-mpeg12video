//! A frame coded as an intra first field and a predicted second field
//! decodes at startup and after reset: the second field may predict from
//! the first. Prediction from a field that was never decoded is rejected.
//! Streams come from the crate's field encoders; FFmpeg decodes the same
//! bytes as the reference.
use oxideav_core::bits::BitWriter;
use oxideav_core::{CodecId, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_mpeg12video::field_picture_encoder::{
    encode_field_intra_picture, encode_field_p_picture, second_p_field_reference,
};
use oxideav_mpeg12video::picture_header::PictureStructure;
use oxideav_mpeg12video::sequence_extension::ChromaFormat;
use oxideav_mpeg12video::stream_writer::{
    write_sequence_extension, write_sequence_header, SequenceHeaderParams,
};
use oxideav_mpeg12video::{
    assemble_frame_from_fields, write_gop_header, FrameBuffer, IntraPictureParams, Mpeg12Decoder,
    Mpeg2Gop, TimeCode,
};
use std::{path::PathBuf, process::Command};

/// One 16×16 field of a 16×32 interlaced frame.
fn field_params() -> IntraPictureParams {
    IntraPictureParams {
        width: 16,
        height: 16,
        chroma_format: ChromaFormat::Yuv420,
        frame_pred_frame_dct: false,
        intra_dc_precision: 0,
        intra_vlc_format: false,
        alternate_scan: false,
        q_scale_type: false,
        progressive_sequence: true,
    }
}

/// Textured content, so a prediction from the wrong field is visible.
fn field(seed: usize) -> FrameBuffer {
    let mut f = FrameBuffer::new(16, 16, ChromaFormat::Yuv420);
    for y in 0..16 {
        for x in 0..16 {
            f.y.put_sample(x, y, (40 + (x * 11 + y * 7 + seed * 37) % 170) as u8);
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            f.cb.put_sample(x, y, (90 + (x * 5 + y * 3 + seed * 13) % 70) as u8);
            f.cr.put_sample(x, y, (100 + (x * 3 + y * 5 + seed * 29) % 70) as u8);
        }
    }
    f
}

fn inverted(f: &FrameBuffer) -> FrameBuffer {
    let mut out = FrameBuffer::new(f.width, f.height, f.chroma_format);
    for (dst, src) in [(&mut out.y, &f.y), (&mut out.cb, &f.cb), (&mut out.cr, &f.cr)] {
        for y in 0..src.height() {
            for x in 0..src.width() {
                dst.put_sample(x, y, 255 - src.get(x, y).unwrap());
            }
        }
    }
    out
}

fn headers() -> BitWriter {
    let mut bw = BitWriter::new();
    let seq = SequenceHeaderParams { horizontal_size: 16, vertical_size: 32, ..Default::default() };
    write_sequence_header(&mut bw, &seq);
    write_sequence_extension(&mut bw, ChromaFormat::Yuv420, false);
    let gop = Mpeg2Gop { time_code: TimeCode::from_display_index(0, 3).unwrap(), closed_gop: true, broken_link: false };
    write_gop_header(&mut bw, &gop);
    bw
}

fn finish(bw: BitWriter) -> Vec<u8> {
    let mut stream = bw.finish();
    stream.extend_from_slice(&[0, 0, 1, 0xB7]);
    stream
}

/// Top I field, then a bottom P field equal to the decoded top field. The
/// slot of the previous frame's bottom field, absent at startup, holds a
/// decoy, so the encoder predicts from the top field with zero motion.
fn startup_pair() -> Vec<u8> {
    let mut bw = headers();
    let top = encode_field_intra_picture(&mut bw, &field(1), &field_params(), PictureStructure::TopField, 0, 4).unwrap();
    let reference = assemble_frame_from_fields(&top, &inverted(&top)).unwrap();
    encode_field_p_picture(&mut bw, &top, &reference, &field_params(), PictureStructure::BottomField, 0, 4, 3).unwrap();
    finish(bw)
}

/// Frame 0: two intra fields. Frame 1: an intra top field and a P bottom
/// field equal to frame 0's decoded bottom field, which it predicts from.
/// Returns the two-frame stream and frame 1 alone.
fn previous_frame_pair() -> (Vec<u8>, Vec<u8>) {
    let mut bw = headers();
    let top0 = encode_field_intra_picture(&mut bw, &field(2), &field_params(), PictureStructure::TopField, 0, 4).unwrap();
    let bottom0 = encode_field_intra_picture(&mut bw, &field(3), &field_params(), PictureStructure::BottomField, 0, 4).unwrap();
    let frame0 = assemble_frame_from_fields(&top0, &bottom0).unwrap();
    let frame1 = |bw: &mut BitWriter| {
        let top1 = encode_field_intra_picture(bw, &field(4), &field_params(), PictureStructure::TopField, 1, 4).unwrap();
        let reference = second_p_field_reference(&top1, PictureStructure::TopField, &frame0).unwrap();
        encode_field_p_picture(bw, &bottom0, &reference, &field_params(), PictureStructure::BottomField, 1, 4, 3).unwrap();
    };
    frame1(&mut bw);
    let both = finish(bw);
    let mut bw = headers();
    frame1(&mut bw);
    (both, finish(bw))
}

fn decode(dec: &mut Mpeg12Decoder, stream: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream.to_vec()))?;
    dec.flush()?;
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => frames.push(frame.planes.iter().flat_map(|p| p.data.clone()).collect()),
            Ok(_) => panic!("non-video output"),
            Err(Error::Eof | Error::NeedMore) => return Ok(frames),
            Err(err) => return Err(err),
        }
    }
}

fn ffmpeg(name: &str, stream: &[u8]) -> Vec<u8> {
    let dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir).join("evidence")
        .join(format!("mpeg12-fields-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, stream).unwrap();
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-f", "mpegvideo", "-idct", "simple", "-i"])
        .arg(&path).args(["-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output().expect("FFmpeg is required for the independent oracle");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

/// Luma rows `parity, parity + 2, …` of a packed 16×32 frame.
fn field_rows(frame: &[u8], parity: usize) -> Vec<&[u8]> {
    (0..16).map(|r| &frame[(2 * r + parity) * 16..][..16]).collect()
}

#[test]
fn an_opening_i_p_field_pair_predicts_its_second_field_from_its_first() {
    let stream = startup_pair();
    let mut dec = Mpeg12Decoder::new(CodecId::new("mpeg2video"));
    let frames = decode(&mut dec, &stream).expect("startup I/P field pair decodes");
    assert_eq!(frames.concat(), ffmpeg("startup.m2v", &stream));
    assert_eq!(frames.len(), 1);
    // The bottom field copies the top field, which is not a flat dummy.
    assert_eq!(field_rows(&frames[0], 1), field_rows(&frames[0], 0));
    assert!(frames[0][..16].iter().any(|&v| v != frames[0][0]));
    // After reset, following other content, the result is identical.
    dec.reset().unwrap();
    decode(&mut dec, &previous_frame_pair().0).unwrap();
    dec.reset().unwrap();
    assert_eq!(decode(&mut dec, &stream).unwrap(), frames);
}

#[test]
fn a_second_field_reading_a_missing_older_field_is_rejected() {
    let (both, alone) = previous_frame_pair();
    let mut dec = Mpeg12Decoder::new(CodecId::new("mpeg2video"));
    // With frame 0 decoded, frame 1's bottom field reproduces frame 0's.
    let frames = decode(&mut dec, &both).unwrap();
    assert_eq!(frames.concat(), ffmpeg("previous.m2v", &both));
    assert_eq!(frames.len(), 2);
    assert_eq!(field_rows(&frames[1], 1), field_rows(&frames[0], 1));
    // Without frame 0 that field was never decoded: reject, from startup
    // and after reset, and recover for valid input afterwards.
    for _ in 0..2 {
        dec.reset().unwrap();
        assert!(matches!(decode(&mut dec, &alone), Err(Error::InvalidData(_))));
    }
    dec.reset().unwrap();
    assert_eq!(decode(&mut dec, &startup_pair()).unwrap().len(), 1);
}
