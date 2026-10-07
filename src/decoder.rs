//! Packet-oriented MPEG-1/2 decoding. Sequence headers are validated during
//! send_packet; receive_frame incrementally reconstructs complete pictures and
//! returns display order. Flush only marks the compressed-input tail complete.
//! Two prediction anchors and a possible first field survive between packets.
//! Output timestamps stay in the packet time base; see `PresentationClock`.
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame, VideoPlane};
use crate::{frame_assembly::FrameBuffer, sequence_extension::ChromaFormat, streaming::StreamDecoder, Error as Mpeg12Error};
use crate::video_sequence::Stamp;

fn map_err(err: Mpeg12Error) -> Error {
    match err {
        Mpeg12Error::InvalidBitstream(detail) => Error::invalid(format!("mpeg12video: invalid bitstream: {detail}")),
        Mpeg12Error::ShortHeader => Error::invalid("mpeg12video: short header (unexpected end of elementary stream)"),
        Mpeg12Error::NotImplemented => Error::unsupported("mpeg12video: bitstream uses a not-yet-implemented syntax path"),
    }
}
/// Codec id string for MPEG-1 video (ISO/IEC 11172-2).
pub const MPEG1_CODEC_ID_STR: &str = "mpeg1video";
/// Codec id string for MPEG-2 video (ITU-T H.262 / ISO/IEC 13818-2).
pub const MPEG2_CODEC_ID_STR: &str = "mpeg2video";
/// Build a packet-oriented decoder. Bitstream geometry overrides container hints.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Mpeg12Decoder::new(params.codec_id.clone())))
}

#[derive(Debug)]
pub struct Mpeg12Decoder {
    codec_id: CodecId,
    stream: StreamDecoder,
    last_output: Option<(u32, u32, PixelFormat)>,
    clock: PresentationClock,
    failed: bool,
}
impl Mpeg12Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self { codec_id, stream: StreamDecoder::random_access(), last_output: None, clock: PresentationClock::default(), failed: false }
    }
    fn output_layout(&self) -> Option<(u32, u32, PixelFormat)> {
        self.last_output.or_else(|| self.stream.dimensions().map(|(w,h,c)| (w as u32,h as u32,pixel_format(c))))
    }
    fn fail(&mut self, err: Mpeg12Error) -> Error {
        // Damaged reference/quantizer state cannot leak into a later epoch.
        self.stream = StreamDecoder::random_access();
        self.failed = true;
        map_err(err)
    }
}
impl Decoder for Mpeg12Decoder {
    fn codec_id(&self) -> &CodecId { &self.codec_id }
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.failed { return Err(Error::invalid("mpeg12video: reset required after decode error")); }
        self.clock.set_time_base(packet.time_base);
        self.stream.push(&packet.data, Stamp { pts: packet.pts, dts: packet.dts }).map_err(|err| self.fail(err))
    }
    fn receive_frame(&mut self) -> Result<Frame> {
        if self.failed { return Err(Error::invalid("mpeg12video: reset required after decode error")); }
        let Some(released) = self.stream.next().map_err(|err| self.fail(err))? else {
            return Err(if self.stream.is_drained() { Error::Eof } else { Error::NeedMore });
        };
        let picture = &released.picture;
        let frame = &picture.decoded.frame;
        self.last_output = Some((frame.width as u32, frame.height as u32, pixel_format(frame.chroma_format)));
        let pts = self.clock.present(picture.stamp.pts, released.release_dts, picture.duration);
        let mut vf = match std::sync::Arc::try_unwrap(released.picture) {
            Ok(output) => owned_frame_buffer_to_video_frame(output.decoded.frame),
            Err(output) => frame_buffer_to_video_frame(&output.decoded.frame),
        };
        vf.pts = pts;
        Ok(Frame::Video(vf))
    }
    fn flush(&mut self) -> Result<()> {
        if self.failed { return Err(Error::invalid("mpeg12video: reset required after decode error")); }
        self.stream.finish();
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.stream = StreamDecoder::random_access();
        self.last_output = None;
        self.clock = PresentationClock::default();
        self.failed = false;
        Ok(())
    }
    fn output_pixel_format(&self) -> Option<PixelFormat> { self.output_layout().map(|(_,_,c)| c) }
    fn output_video_dimensions(&self) -> Option<(u32,u32)> { self.output_layout().map(|(w,h,_)| (w,h)) }
}

/// Display-order presentation times in the packet time base. Between a
/// picture's own PTS and the decode time that released it, the choice is
/// FFmpeg's `guess_correct_pts` (libavcodec/decode.c): the PTS unless PTS
/// values have been non-increasing more often than DTS values, as when a
/// demuxer labels coded-order times as PTS. Missing times continue exactly
/// from the previous frame's §6.3.10 duration; an untimed epoch starts at 0.
/// No picture index is mixed with container timestamps.
#[derive(Debug, Default)]
struct PresentationClock {
    time_base: Option<TimeBase>,
    last_pts: Option<i64>,
    last_dts: Option<i64>,
    faulty_pts: u64,
    faulty_dts: u64,
    /// End of the previous frame in ticks, exactly `numerator / denominator`.
    next: Option<(i128, i128)>,
    started: bool,
}
impl PresentationClock {
    fn set_time_base(&mut self, time_base: TimeBase) {
        if self.time_base != Some(time_base) {
            // An expectation in another unit cannot be carried over.
            self.next = None;
            self.time_base = Some(time_base);
        }
    }

    fn present(&mut self, pts: Option<i64>, release_dts: Option<i64>, duration: Option<(u64, u64)>) -> Option<i64> {
        let expected = self.next.and_then(|(n, d)| round_div(n, d));
        // A frame released without a decode time (end of input) follows the
        // previous one.
        let dts = release_dts.or(expected);
        if let Some(dts) = dts {
            self.faulty_dts += u64::from(self.last_dts.is_some_and(|last| dts <= last));
            self.last_dts = Some(dts);
        } else if pts.is_some() {
            self.last_dts = pts;
        }
        if let Some(pts) = pts {
            self.faulty_pts += u64::from(self.last_pts.is_some_and(|last| pts <= last));
            self.last_pts = Some(pts);
        } else if dts.is_some() {
            self.last_pts = dts;
        }
        let chosen = if pts.is_some() && (self.faulty_pts <= self.faulty_dts || dts.is_none()) { pts } else { dts };
        let chosen = chosen.or((!self.started).then_some(0));
        self.started = true;
        let step = duration.zip(self.time_base).and_then(|((num, den), base)| {
            let base = base.as_rational();
            (base.num > 0 && base.den > 0 && den > 0)
                .then(|| (i128::from(num) * i128::from(base.den), i128::from(den) * i128::from(base.num)))
        });
        self.next = match (chosen, step) {
            (Some(at), Some((num, den))) => Some(match self.next {
                // Continue an interpolated run exactly, not from its rounding.
                Some((n, d)) if expected == Some(at) => ((if d == den { n } else { n * den / d }) + num, den),
                _ => (i128::from(at) * den + num, den),
            }),
            _ => None,
        };
        chosen
    }
}

fn round_div(numerator: i128, denominator: i128) -> Option<i64> {
    i64::try_from((2 * numerator + denominator).div_euclid(2 * denominator)).ok()
}
fn pixel_format(chroma: ChromaFormat) -> PixelFormat {
    match chroma { ChromaFormat::Yuv420 => PixelFormat::Yuv420P, ChromaFormat::Yuv422 => PixelFormat::Yuv422P, ChromaFormat::Yuv444 => PixelFormat::Yuv444P }
}
/// Copy visible Y/Cb/Cr rectangles from a prediction reference. Width comes from
/// the validated FrameBuffer, never the stride of its padded macroblock grid.
pub fn frame_buffer_to_video_frame(frame: &FrameBuffer) -> VideoFrame {
    let (cw, ch) = frame.visible_chroma_dims();
    let plane = |plane: &crate::frame_assembly::Plane, w: usize, h: usize| VideoPlane {
        stride: w, data: plane.packed_rect(w, h),
    };
    VideoFrame { pts: None, planes: vec![
        plane(&frame.y, frame.width, frame.height), plane(&frame.cb, cw, ch), plane(&frame.cr, cw, ch),
    ] }
}
fn owned_frame_buffer_to_video_frame(frame: FrameBuffer) -> VideoFrame {
    let (cw, ch) = frame.visible_chroma_dims();
    VideoFrame { pts: None, planes: vec![
        VideoPlane { stride: frame.width, data: frame.y.into_packed_rect(frame.width, frame.height) },
        VideoPlane { stride: cw, data: frame.cb.into_packed_rect(cw, ch) },
        VideoPlane { stride: cw, data: frame.cr.into_packed_rect(cw, ch) },
    ] }
}
