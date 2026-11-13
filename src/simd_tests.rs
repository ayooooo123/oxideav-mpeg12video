use super::*;

#[test]
fn pixels_match_scalar_rounding() {
    let src: Vec<u8> = (0..32 * 18).map(|i| ((i * 97 + i / 11) & 255) as u8).collect();
    for w in [8, 16] { for h in [8, 16] { for phase in 0..4 { for round in 0..2 { for avg in [false, true] {
        let mut dst = vec![113u8; 32 * 16];
        let mut expected = dst.clone();
        for y in 0..h { for x in 0..w {
            let at = y * 32 + x;
            let a = u32::from(src[at]);
            let v = match phase {
                0 => a,
                1 => (a + u32::from(src[at + 1]) + round) >> 1,
                2 => (a + u32::from(src[at + 32]) + round) >> 1,
                _ => (a + u32::from(src[at + 1]) + u32::from(src[at + 32]) + u32::from(src[at + 33]) + round + 1) >> 2,
            };
            expected[at] = if avg { ((113 + v + 1) >> 1) as u8 } else { v as u8 };
        }}
        pixels(&mut dst, 32, &src, 32, w, h, phase, round as u16, avg);
        assert_eq!(dst, expected);
    }}}}}
}

#[test]
fn column_transform_matches_wrapping_scalar() {
    let matrix = [
        [16383,22725,21407,19266,16383,12873,8867,4520],
        [16383,19266,8867,-4520,-16383,-22725,-21407,-12873],
        [16383,12873,-8867,-22725,-16383,4520,21407,19266],
        [16383,4520,-21407,-12873,16383,19266,-8867,-22725],
        [16383,-4520,-21407,12873,16383,-19266,-8867,22725],
        [16383,-12873,-8867,22725,-16383,-4520,21407,-19266],
        [16383,-19266,8867,4520,-16383,22725,-21407,12873],
        [16383,-22725,21407,-19266,16383,-12873,8867,-4520],
    ];
    let mut seed = 1u32;
    for _ in 0..128 {
        let block = core::array::from_fn(|_| { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 16) as i16 });
        let actual = columns(&block);
        for y in 0..8 { for x in 0..8 {
            let mut value = 16383i32 * 32;
            for k in 0..8 { value = value.wrapping_add((matrix[y][k] as i32).wrapping_mul(i32::from(block[k * 8 + x]))); }
            assert_eq!(actual[y][x], value >> 20);
        }}
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn independent_slice_rows_match_serial() {
    use crate::ff_decode::{decode_picture, Decoded, Pic, PICT_I};
    use crate::frame_assembly::FrameBuffer;
    use crate::sequence_extension::ChromaFormat;
    let pic = Pic { mpeg2: false, pict_type: PICT_I, f_code: [[1;2];2], full_pel: [false;2],
        intra_dc_precision: 0, frame_pred_frame_dct: true, concealment_motion_vectors: false,
        q_scale_type: false, intra_vlc_format: false, alternate_scan: false, top_field_first: false,
        progressive_sequence: true, height: 32, mb_width: 1, mb_height: 2,
        intra_matrix: [16;64], inter_matrix: [16;64], chroma_intra_matrix: [16;64], chroma_inter_matrix: [16;64], last: None, next: None };
    let mut packet = Vec::new();
    for row in 1..=2 {
        packet.extend_from_slice(&[0,0,1,row]);
        let mut bits = String::from("00010011");
        for _ in 0..4 { bits.push_str("10010"); }
        for _ in 0..2 { bits.push_str("0010"); }
        while bits.len() % 8 != 0 { bits.push('0'); }
        for b in bits.as_bytes().chunks_exact(8) { packet.push(b.iter().fold(0u8, |a, v| (a << 1) | (v - b'0'))); }
    }
    packet.extend_from_slice(&[0,0,1,0]);
    let decode = |threads| {
        let fresh = FrameBuffer::new(16,32,ChromaFormat::Yuv420);
        match decode_picture(&pic, &packet, fresh, &oxideav_core::ExecutionContext::with_threads(threads)).unwrap() {
            Decoded::Complete(frame) => frame,
            _ => panic!("complete independent rows expected"),
        }
    };
    let serial = decode(1);
    let parallel = decode(2);
    assert_eq!(serial.y.samples(), parallel.y.samples());
    assert_eq!(serial.cb.samples(), parallel.cb.samples());
    assert_eq!(serial.cr.samples(), parallel.cr.samples());
}
