//! Complete independent pixel oracles and packet-boundary/state regressions.
use oxideav_core::{CodecId, Decoder, Error, Frame, Packet, PixelFormat, TimeBase};
use oxideav_mpeg12video::Mpeg12Decoder;
use std::{path::PathBuf, process::Command};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conformance").join(name)
}
fn packet(data: &[u8]) -> Packet { Packet::new(0, TimeBase::new(1, 25), data.to_vec()) }
fn decoder() -> Mpeg12Decoder { Mpeg12Decoder::new(CodecId::new("mpeg2video")) }
fn reference(name: &str) -> Vec<u8> {
    let format = if name.contains("422") { "yuv422p" } else { "yuv420p" };
    reference_path(&fixture(name), format)
}
fn reference_path(path: &std::path::Path, format: &str) -> Vec<u8> {
    let out = Command::new("ffmpeg").args(["-v", "error", "-nostdin", "-idct", "simple", "-i"])
        .arg(path).args(["-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", format, "-f", "rawvideo", "-"])
        .output().expect("FFmpeg is required for the independent oracle");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}
fn drain(dec: &mut dyn Decoder, bytes: &mut Vec<u8>) -> Result<usize, Error> {
    let mut frames = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                for plane in frame.planes { bytes.extend_from_slice(&plane.data); }
                frames += 1;
            }
            Ok(_) => panic!("non-video output"),
            Err(Error::NeedMore | Error::Eof) => return Ok(frames),
            Err(err) => return Err(err),
        }
    }
}
const CASES: &[(&str, usize)] = &[
    ("mpeg1-ibbp-96x64.m1v", 30), ("mpeg1-bigmv-160x128.m1v", 24),
    ("mpeg1-vcd-352x240.m1v", 9), ("mpeg2-ibbp-96x64.m2v", 30),
    ("mpeg2-ilaced-96x64.m2v", 20), ("mpeg2-ivlc-96x64.m2v", 20),
    ("mpeg2-422-96x64.m2v", 15), ("mpeg2-100x62.m2v", 15),
    ("mpeg2-ilaced48hm-96x48.m2v", 18), ("mpeg2-qmat-96x64.m2v", 20),
    ("fieldpics-48x64.m2v", 5),
];
#[test]
fn all_complete_frames_match_ffmpeg_simple() {
    let mut failures = Vec::new();
    for &(name, expected_frames) in CASES {
        let data = std::fs::read(fixture(name)).unwrap();
        let expected = reference(name);
        let mut dec = decoder();
        let mut actual = Vec::new();
        let mut count = 0;
        let result = (|| {
            for chunk in data.chunks(997) {
                dec.send_packet(&packet(chunk))?;
                count += drain(&mut dec, &mut actual)?;
            }
            dec.flush()?;
            count += drain(&mut dec, &mut actual)?;
            Ok::<_, Error>(())
        })();
        let mismatch = actual.iter().zip(&expected).filter(|(a,b)| a != b).count();
        eprintln!("{name}: frames={count}/{expected_frames} bytes={}/{} differing={mismatch} result={result:?}", actual.len(), expected.len());
        if result.is_err() || count != expected_frames || actual != expected { failures.push(name); }
    }
    assert!(failures.is_empty(), "complete pixel/count failures: {failures:?}");
}
#[test]
fn validated_geometry_precedes_first_picture_and_output_precedes_eof() {
    let name = "mpeg2-100x62.m2v";
    let data = std::fs::read(fixture(name)).unwrap();
    let first_picture = data.windows(4).position(|w| w == [0,0,1,0]).unwrap();
    let mut dec = decoder();
    assert_eq!(dec.output_video_dimensions(), None);
    // Leading zero stuffing does not require an ever-growing ES buffer.
    for _ in 0..300 { dec.send_packet(&packet(&[0;1024])).unwrap(); }
    for byte in &data[..first_picture] { dec.send_packet(&packet(&[*byte])).unwrap(); }
    assert_eq!(dec.output_video_dimensions(), Some((100,62)));
    assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv420P));
    let mut actual = Vec::new();
    let mut before_eof = 0;
    for chunk in data[first_picture..].chunks(113) {
        dec.send_packet(&packet(chunk)).unwrap();
        before_eof += drain(&mut dec, &mut actual).unwrap();
    }
    assert!(before_eof > 0, "first picture must be returned before flush");
    dec.flush().unwrap();
    let count = before_eof + drain(&mut dec, &mut actual).unwrap();
    assert_eq!(count, 15);
    assert_eq!(actual, reference(name));
}
#[test]
fn geometry_describes_returned_frame_not_later_sequence() {
    let a = std::fs::read(fixture("mpeg2-100x62.m2v")).unwrap();
    let b = std::fs::read(fixture("mpeg2-422-96x64.m2v")).unwrap();
    let mut dec = decoder();
    dec.send_packet(&packet(&a)).unwrap();
    dec.send_packet(&packet(&b)).unwrap();
    dec.flush().unwrap();
    for index in 0..30 {
        let Frame::Video(_) = dec.receive_frame().unwrap() else { panic!("video"); };
        assert_eq!(dec.output_video_dimensions(), Some(if index < 15 {(100,62)} else {(96,64)}));
        assert_eq!(dec.output_pixel_format(), Some(if index < 15 {PixelFormat::Yuv420P} else {PixelFormat::Yuv422P}));
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    dec.reset().unwrap();
    assert_eq!(dec.output_video_dimensions(), None);
    assert_eq!(dec.output_pixel_format(), None);
}
#[test]
fn stateful_mutations_and_reset_recover_exact_output() {
    let name = "fieldpics-48x64.m2v";
    let data = std::fs::read(fixture(name)).unwrap();
    let expected = reference(name);
    let mut dec = decoder();
    let mut seed = 0x12345678u32;
    for iteration in 0..256 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let mut damaged = data.clone();
        if iteration % 2 == 0 { damaged.truncate(seed as usize % data.len()); }
        else { let pos = seed as usize % data.len(); damaged[pos] ^= 1 << (iteration % 8); }
        for chunk in damaged.chunks(31) {
            let _ = dec.send_packet(&packet(chunk));
            for _ in 0..8 { if dec.receive_frame().is_err() { break; } }
        }
        let _ = dec.flush();
        dec.reset().unwrap();
        let mut actual = Vec::new();
        for chunk in data.chunks(47) {
            dec.send_packet(&packet(chunk)).unwrap();
            drain(&mut dec, &mut actual).unwrap();
        }
        dec.flush().unwrap();
        drain(&mut dec, &mut actual).unwrap();
        assert_eq!(actual, expected, "reset recovery after mutation {iteration}");
        dec.reset().unwrap();
    }
}

#[test]
fn reset_then_seek_to_sequence_or_mid_gop_matches_independent_output() {
    let data = std::fs::read(fixture("mpeg2-ibbp-96x64.m2v")).unwrap();
    let mut points: Vec<_> = data.windows(4).enumerate()
        .filter_map(|(i,w)| (w == [0,0,1,0xB3]).then_some(i)).skip(1).collect();
    assert_eq!(points.len(), 2, "fixture must exercise both later GOPs");
    // A mid-GOP P-picture: no sequence header or reference precedes it.
    let pictures: Vec<_> = data.windows(4).enumerate()
        .filter_map(|(i,w)| (w == [0,0,1,0]).then_some(i)).collect();
    assert_eq!((data[pictures[4] + 5] >> 3) & 7, 2, "fifth coded picture is P");
    points.push(pictures[4]);
    let dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir).join("evidence")
        .join(format!("mpeg12-seek-{}",std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut dec = decoder();
    for offset in points {
        // Populate references and leave a partial picture before seeking.
        for chunk in data[..offset-3].chunks(997) {
            dec.send_packet(&packet(chunk)).unwrap();
            drain(&mut dec,&mut Vec::new()).unwrap();
        }
        dec.reset().unwrap();
        assert_eq!(dec.output_video_dimensions(),None);
        let path = dir.join(format!("{offset}.m2v"));
        std::fs::write(&path,&data[offset..]).unwrap();
        let expected = reference_path(&path,"yuv420p");
        let mut actual = Vec::new();
        for chunk in data[offset..].chunks(113) {
            dec.send_packet(&packet(chunk)).unwrap();
            drain(&mut dec,&mut actual).unwrap();
        }
        dec.flush().unwrap();
        drain(&mut dec,&mut actual).unwrap();
        std::fs::write(dir.join(format!("{offset}.decoded.yuv")),&actual).unwrap();
        std::fs::write(dir.join(format!("{offset}.ffmpeg.yuv")),&expected).unwrap();
        assert_eq!(actual,expected,"complete output after seek to byte {offset}");
        dec.reset().unwrap();
    }
}

#[test]
fn extended_geometry_is_validated_without_silent_mpeg1_fallback() {
    let data = std::fs::read(fixture("mpeg2-100x62.m2v")).unwrap();
    let picture = data.windows(4).position(|w| w == [0,0,1,0]).unwrap();
    let extension = data.windows(4).position(|w| w == [0,0,1,0xB5]).unwrap();
    let mut header = data[..picture].to_vec();
    // horizontal_size_extension's low bit precedes vertical_size_extension.
    header[extension + 6] |= 0x80;
    let mut dec = decoder();
    for byte in &header { dec.send_packet(&packet(&[*byte])).unwrap(); }
    assert_eq!(dec.output_video_dimensions(),Some((4196,62)));
    assert_eq!(dec.output_pixel_format(),Some(PixelFormat::Yuv420P));
    dec.reset().unwrap();
    // A missing required extension marker is malformed, not MPEG-1.
    header[extension + 7] &= !1;
    assert!(dec.send_packet(&packet(&header)).is_err());
    assert_eq!(dec.output_video_dimensions(),None);
    dec.reset().unwrap();
    dec.send_packet(&packet(&data[..picture])).unwrap();
    assert_eq!(dec.output_video_dimensions(),Some((100,62)));
}
