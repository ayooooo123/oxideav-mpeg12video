//! Display-order presentation timestamps through B-picture reordering:
//! sparse PES stamps, coded-order labels and untimed display durations.
//! Expected values come from the bitstream's own GOP/temporal_reference
//! structure or from FFmpeg, never from this decoder.
use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_mpeg12video::{make_encoder, Mpeg12Decoder};
use std::{path::{Path, PathBuf}, process::Command};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conformance").join(name)
}

fn decoder() -> Mpeg12Decoder {
    Mpeg12Decoder::new(CodecId::new("mpeg2video"))
}

fn collect_pts(dec: &mut dyn Decoder, out: &mut Vec<Option<i64>>) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => out.push(frame.pts),
            Ok(_) => panic!("non-video output"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(err) => panic!("decode: {err}"),
        }
    }
}

/// A coded picture: where its access unit begins (its first sequence/GOP
/// header, else its start code), its start code, picture_coding_type and
/// display index from the §6.3.9 GOP reset plus temporal_reference.
struct Picture {
    begin: usize,
    start: usize,
    kind: u8,
    display: usize,
}

fn coded_pictures(data: &[u8]) -> Vec<Picture> {
    let (mut pictures, mut header, mut gop_base, mut in_gop) = (Vec::new(), None, 0, 0);
    for i in 0..data.len().saturating_sub(5) {
        if data[i..i + 3] != [0, 0, 1] {
            continue;
        }
        match data[i + 3] {
            0xB3 => { header.get_or_insert(i); }
            0xB8 => {
                header.get_or_insert(i);
                gop_base += in_gop;
                in_gop = 0;
            }
            0x00 => {
                let temporal_reference = (usize::from(data[i + 4]) << 2) | usize::from(data[i + 5] >> 6);
                pictures.push(Picture {
                    begin: header.take().unwrap_or(i),
                    start: i,
                    kind: (data[i + 5] >> 3) & 7,
                    display: gop_base + temporal_reference,
                });
                in_gop += 1;
            }
            _ => {}
        }
    }
    pictures
}

#[test]
fn sparse_pes_timestamps_survive_b_reordering_and_gaps_are_interpolated() {
    let data = std::fs::read(fixture("mpeg2-ibbp-96x64.m2v")).unwrap();
    let pictures = coded_pictures(&data);
    // A closed GOP, then open GOPs whose leading B-pictures precede the I.
    assert_eq!(pictures.len(), 30);
    assert!(pictures.iter().any(|p| p.kind == 3 && p.display % 12 == 10));
    const PERIOD: i64 = 3600; // frame_rate_code 3 (25 Hz) at 90 kHz
    let pts = |p: &Picture| 90_000 + PERIOD * p.display as i64;
    // ISO/IEC 13818-1 decoder model: a B-picture is decoded when presented,
    // an anchor when the previous anchor is presented.
    let mut previous_anchor = None;
    let stamps: Vec<(i64, i64)> = pictures.iter().map(|p| {
        let pts = pts(p);
        if p.kind == 3 {
            return (pts, pts);
        }
        let dts = previous_anchor.unwrap_or(pts - PERIOD);
        previous_anchor = Some(pts);
        (pts, dts)
    }).collect();

    let mut dec = decoder();
    let mut out = Vec::new();
    let (mut first, mut index) = (0, 0);
    while first < pictures.len() {
        let count = [1, 2, 3, 1, 2][index % 5].min(pictures.len() - first);
        let begin = pictures[first].begin;
        let end = pictures.get(first + count).map_or(data.len(), |p| p.begin);
        let split = (begin + (end - begin) / 2).max(pictures[first].start + 4).min(end);
        // Every third PES is unstamped; the rest stamp the first picture
        // commencing in them, with a DTS only where it differs from the PTS.
        let (pts, dts) = stamps[first];
        for (part, range) in [(0, begin..split), (1, split..end)] {
            let mut packet = Packet::new(0, TimeBase::new(1, 90_000), data[range].to_vec());
            if part == 0 && index % 3 != 2 {
                packet.pts = Some(pts);
                packet.dts = (dts != pts).then_some(dts);
            }
            dec.send_packet(&packet).unwrap();
            collect_pts(&mut dec, &mut out);
        }
        first += count;
        index += 1;
    }
    dec.flush().unwrap();
    collect_pts(&mut dec, &mut out);
    let expected: Vec<_> = (0..30).map(|k| Some(90_000 + PERIOD * k)).collect();
    assert_eq!(out, expected);
}

#[test]
fn coded_order_labels_still_give_increasing_display_times() {
    // A demuxer numbering packets in coded order labels decode times as PTS.
    let data = std::fs::read(fixture("mpeg2-ibbp-96x64.m2v")).unwrap();
    let pictures = coded_pictures(&data);
    let mut dec = decoder();
    let mut out = Vec::new();
    for (index, picture) in pictures.iter().enumerate() {
        let end = pictures.get(index + 1).map_or(data.len(), |next| next.begin);
        let mut packet = Packet::new(0, TimeBase::new(1, 25), data[picture.begin..end].to_vec());
        packet.pts = Some(index as i64);
        packet.dts = Some(index as i64);
        dec.send_packet(&packet).unwrap();
        collect_pts(&mut dec, &mut out);
    }
    dec.flush().unwrap();
    collect_pts(&mut dec, &mut out);
    let times: Vec<i64> = out.iter().map(|t| t.expect("every frame is timed")).collect();
    assert_eq!(times.len(), 30);
    assert!(times.windows(2).all(|w| w[0] < w[1]), "{times:?}");
    assert!(times[1..].windows(2).all(|w| w[1] - w[0] == 1), "{times:?}");
}

fn decode_untimed(data: &[u8], time_base: TimeBase) -> Vec<Option<i64>> {
    let mut dec = decoder();
    let mut out = Vec::new();
    for chunk in data.chunks(997) {
        dec.send_packet(&Packet::new(0, time_base, chunk.to_vec())).unwrap();
        collect_pts(&mut dec, &mut out);
    }
    dec.flush().unwrap();
    collect_pts(&mut dec, &mut out);
    out
}

#[test]
fn untimed_input_starts_at_zero_in_the_packet_time_base() {
    let data = std::fs::read(fixture("mpeg2-ibbp-96x64.m2v")).unwrap();
    let expected: Vec<_> = (0..30).map(|k| Some(3600 * k)).collect();
    assert_eq!(decode_untimed(&data, TimeBase::new(1, 90_000)), expected);
}

/// FFmpeg's raw-ES timing advances by each displayed frame's duration,
/// including `repeat_pict`. Its timeline is offset by the decoding delay and
/// its first frame depends on when it detects reordering, so frame-to-frame
/// differences from the second frame on are compared.
fn ffmpeg_timing(path: &Path) -> (TimeBase, Vec<Option<i64>>) {
    let probe = |entries: &str| {
        let out = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of", "csv=p=0"])
            .arg(path).output().expect("FFmpeg is required for the independent oracle");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };
    let base = probe("stream=time_base");
    let (num, den) = base.trim().trim_end_matches(',').split_once('/').unwrap();
    let times = probe("frame=best_effort_timestamp").lines()
        .map(|line| line.trim().trim_end_matches(',').parse().ok()).collect();
    (TimeBase::new(num.parse().unwrap(), den.parse().unwrap()), times)
}

fn generated(name: &str, frames: usize, options: &[(&str, &str)]) -> PathBuf {
    let mut params = CodecParameters::video(CodecId::new("mpeg2video"));
    params.width = Some(64);
    params.height = Some(48);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30_000, 1001));
    for (key, value) in options {
        params.options.insert(*key, *value);
    }
    let mut enc = make_encoder(&params).unwrap();
    for t in 0..frames {
        let plane = |w: usize, h: usize, seed: usize| VideoPlane {
            stride: w,
            data: (0..w * h).map(|i| (40 + (i * 7 + t * 13 + seed) % 170) as u8).collect(),
        };
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(t as i64),
            planes: vec![plane(64, 48, 0), plane(32, 24, 50), plane(32, 24, 90)],
        })).unwrap();
    }
    enc.flush().unwrap();
    let stream = enc.receive_packet().unwrap().data;
    let dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir).join("evidence")
        .join(format!("mpeg12-timing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, stream).unwrap();
    path
}

#[test]
fn display_durations_match_ffmpeg_for_fields_mpeg1_and_repeat_first_field() {
    let inputs = [
        fixture("mpeg2-ibbp-96x64.m2v"),
        fixture("mpeg2-ilaced-96x64.m2v"),
        fixture("fieldpics-48x64.m2v"),
        fixture("mpeg1-ibbp-96x64.m1v"),
        // Interlaced 3:2 pulldown: three- and two-field frames.
        generated("pulldown.m2v", 12, &[("interlaced", "true"), ("pulldown", "3:2"), ("b_between", "1")]),
        // Progressive top_field_first + repeat_first_field: three frames each.
        generated("tripled.m2v", 12, &[("top_field_first", "true"), ("repeat_first_field", "true"), ("b_between", "1")]),
    ];
    for path in inputs {
        let (base, reference) = ffmpeg_timing(&path);
        let ours = decode_untimed(&std::fs::read(&path).unwrap(), base);
        assert_eq!(ours.len(), reference.len(), "{}", path.display());
        assert_eq!(ours[0], Some(0), "{}", path.display());
        let mut compared = 0;
        for i in 2..ours.len() {
            if let (Some(a), Some(b)) = (reference[i - 1], reference[i]) {
                assert_eq!(ours[i].unwrap() - ours[i - 1].unwrap(), b - a, "{} frame {i}", path.display());
                compared += 1;
            }
        }
        assert!(compared + 3 >= ours.len(), "{}: only {compared} reference durations", path.display());
    }
}

#[test]
fn an_unreduced_time_base_times_frames_like_its_reduced_form() {
    // (M, M) and (1, 1) both mean one second per tick. 30 frames at 25 Hz
    // are followed by 12 at 30000/1001 Hz, starting M - 100 seconds in.
    let m = i64::MAX;
    let mut data = std::fs::read(fixture("mpeg2-ibbp-96x64.m2v")).unwrap();
    let mut ntsc = std::fs::read(generated("ntsc.m2v", 12, &[("b_between", "1")])).unwrap();
    // Declare 30000/1001 Hz (Table 6-4 code 4) in its sequence headers.
    for i in 0..ntsc.len() - 7 {
        if ntsc[i..i + 4] == [0, 0, 1, 0xB3] {
            ntsc[i + 7] = (ntsc[i + 7] & 0xF0) | 4;
        }
    }
    data.extend(ntsc);
    let decode = |base: TimeBase| {
        let mut dec = decoder();
        let mut out = Vec::new();
        for (index, chunk) in data.chunks(997).enumerate() {
            let mut packet = Packet::new(0, base, chunk.to_vec());
            packet.pts = (index == 0).then_some(m - 100);
            dec.send_packet(&packet).unwrap();
            collect_pts(&mut dec, &mut out);
        }
        dec.flush().unwrap();
        collect_pts(&mut dec, &mut out);
        out
    };
    let reduced = decode(TimeBase::new(1, 1));
    assert_eq!(decode(TimeBase::new(m, m)), reduced);
    // Display k of 30 at 1/25 s, then j of 12 at 1001/30000 s, in units of
    // 1/750000 s from M - 100, rounded to the nearest second. At j = 8 the
    // two rates round differently, so the rate change is observable.
    let units = (0..30).map(|k| k * 30_000).chain((0..12).map(|j| 900_000 + j * 25_025));
    let expected: Vec<_> = units.map(|u: i64| Some(m - 100 + (2 * u + 750_000) / 1_500_000)).collect();
    assert_eq!(reduced, expected);
}

#[test]
fn an_unrepresentable_interpolated_time_is_an_error() {
    let data = std::fs::read(fixture("mpeg2-ibbp-96x64.m2v")).unwrap();
    let mut packet = Packet::new(0, TimeBase::new(1, 90_000), data);
    packet.pts = Some(i64::MAX - 10);
    let mut dec = decoder();
    dec.send_packet(&packet).unwrap();
    dec.flush().unwrap();
    let Ok(Frame::Video(first)) = dec.receive_frame() else { panic!("first frame") };
    assert_eq!(first.pts, Some(i64::MAX - 10));
    // The next frame would start 3600 ticks later, past i64::MAX.
    assert!(matches!(dec.receive_frame(), Err(Error::InvalidData(_))));
}

fn ffmpeg_frames(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-f", "mpegvideo", "-idct", "simple", "-i"])
        .arg(path).args(["-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output().expect("FFmpeg is required for the independent oracle");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

#[test]
fn low_delay_pictures_leave_at_their_own_decode_time() {
    // An I/P/P stream, flagged low_delay in its sequence_extension: no
    // B-pictures, so no picture waits for a later anchor.
    let path = generated("lowdelay.m2v", 3, &[("b_between", "0")]);
    let mut data = std::fs::read(&path).unwrap();
    let extension = data.windows(5).position(|w| w[..4] == [0, 0, 1, 0xB5] && w[4] >> 4 == 1).unwrap();
    data[extension + 9] |= 0x80;
    let parsed = oxideav_mpeg12video::sequence_extension::Mpeg2SequenceExtension::parse(&data[extension..]);
    assert!(parsed.unwrap().low_delay);
    // Without a trailing sequence_end_code the last picture completes at flush.
    if data.ends_with(&[0, 0, 1, 0xB7]) {
        data.truncate(data.len() - 4);
    }
    std::fs::write(&path, &data).unwrap();
    let pictures: Vec<usize> = data.windows(4).enumerate()
        .filter_map(|(i, w)| (w == [0, 0, 1, 0]).then_some(i)).collect();
    assert_eq!(pictures.len(), 3);
    let bounds = [0, pictures[1], pictures[2], data.len()];

    let mut dec = decoder();
    let (mut times, mut pixels, mut released) = (Vec::new(), Vec::new(), Vec::new());
    let mut drain = |dec: &mut Mpeg12Decoder| {
        let mut count = 0;
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(frame)) => {
                    times.push(frame.pts);
                    frame.planes.iter().for_each(|p| pixels.extend_from_slice(&p.data));
                    count += 1;
                }
                Ok(_) => panic!("non-video output"),
                Err(Error::NeedMore | Error::Eof) => return count,
                Err(err) => panic!("decode: {err}"),
            }
        }
    };
    // Packets carry DTS 0, 1, 2 and no PTS.
    for (dts, range) in bounds.windows(2).enumerate() {
        let mut packet = Packet::new(0, TimeBase::new(1, 25), data[range[0]..range[1]].to_vec());
        packet.dts = Some(dts as i64);
        dec.send_packet(&packet).unwrap();
        released.push(drain(&mut dec));
    }
    dec.flush().unwrap();
    released.push(drain(&mut dec));
    // A picture is complete when the next one starts and leaves then.
    assert_eq!(released, vec![0, 1, 1, 1]);
    assert_eq!(times, vec![Some(0), Some(1), Some(2)]);
    assert_eq!(pixels, ffmpeg_frames(&path));
}
