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
        let pts = self.clock.present(picture.stamp.pts, released.release_dts, picture.duration)
            .map_err(|Unrepresentable| Error::invalid("mpeg12video: presentation time does not fit i64 ticks"))?;
        let frame = &picture.decoded.frame;
        self.last_output = Some((frame.width as u32, frame.height as u32, pixel_format(frame.chroma_format)));
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
/// No picture index is mixed with container timestamps. Interpolation uses
/// the time base in lowest terms with checked arithmetic; a frame needing a
/// time that does not fit `i64` ticks gets `Unrepresentable`.
#[derive(Debug, Default)]
struct PresentationClock {
    time_base: Option<TimeBase>,
    /// `time_base` in lowest terms, when positive.
    seconds_per_tick: Option<(u128, u128)>,
    last_pts: Option<i64>,
    last_dts: Option<i64>,
    faulty_pts: u64,
    faulty_dts: u64,
    /// When the previous frame ends.
    next: Next,
    started: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Unrepresentable;

#[derive(Debug, Default, Clone, Copy)]
enum Next {
    #[default]
    Unknown,
    At(Ticks),
    /// The previous frame ends beyond the exact tick range.
    Unrepresentable,
}

/// An exact time of `whole + rem / den` ticks in lowest terms, `rem < den`.
/// The whole part is wider than `i64`, so a long duration still adds
/// exactly; only a presented time must fit `i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ticks {
    whole: i128,
    rem: u128,
    den: u128,
}

impl Ticks {
    fn at(ticks: i64) -> Self {
        Self { whole: i128::from(ticks), rem: 0, den: 1 }
    }

    /// The exact sum over the common denominator, in lowest terms.
    fn plus(self, step: Ticks) -> Option<Ticks> {
        let g = gcd(self.den, step.den);
        let den = (self.den / g).checked_mul(step.den)?;
        let rem = self.rem.checked_mul(step.den / g)?.checked_add(step.rem.checked_mul(self.den / g)?)?;
        let carry = rem >= den;
        let rem = if carry { rem - den } else { rem };
        let g = gcd(rem, den);
        Some(Ticks {
            whole: self.whole.checked_add(step.whole)?.checked_add(i128::from(carry))?,
            rem: rem / g,
            den: den / g,
        })
    }

    /// The nearest tick, halves rounding up, when it fits `i64`.
    fn rounded(self) -> Option<i64> {
        i64::try_from(self.whole.checked_add(i128::from(2 * self.rem >= self.den))?).ok()
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A duration of `num / den` seconds in ticks of `seconds_per_tick`. The
/// factors are below 2^64 and 2^63, so the products fit `u128` and the whole
/// part fits `i128`.
fn duration_ticks((num, den): (u64, u64), (tick_num, tick_den): (u128, u128)) -> Option<Ticks> {
    let (n, d) = (u128::from(num) * tick_den, u128::from(den) * tick_num);
    let g = gcd(n, d);
    if g == 0 || d == 0 {
        return None;
    }
    let (n, d) = (n / g, d / g);
    Some(Ticks { whole: i128::try_from(n / d).ok()?, rem: n % d, den: d })
}

impl PresentationClock {
    fn set_time_base(&mut self, time_base: TimeBase) {
        if self.time_base != Some(time_base) {
            // An expectation in another unit cannot be carried over.
            self.next = Next::Unknown;
            self.time_base = Some(time_base);
            let base = time_base.as_rational();
            self.seconds_per_tick = (base.num > 0 && base.den > 0).then(|| {
                let (num, den) = (base.num as u128, base.den as u128);
                let g = gcd(num, den);
                (num / g, den / g)
            });
        }
    }

    fn present(&mut self, pts: Option<i64>, release_dts: Option<i64>, duration: Option<(u64, u64)>)
        -> std::result::Result<Option<i64>, Unrepresentable> {
        let (expected, overflowed) = match self.next {
            Next::At(end) => (end.rounded(), end.rounded().is_none()),
            Next::Unknown => (None, false),
            Next::Unrepresentable => (None, true),
        };
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
        if chosen.is_none() && overflowed {
            return Err(Unrepresentable);
        }
        self.started = true;
        self.next = match (chosen, duration, self.seconds_per_tick) {
            (Some(at), Some(duration), Some(tick)) => {
                // Continue an interpolated run exactly, not from its rounding.
                let start = match self.next {
                    Next::At(end) if expected == Some(at) => end,
                    _ => Ticks::at(at),
                };
                match duration_ticks(duration, tick).and_then(|step| start.plus(step)) {
                    Some(end) => Next::At(end),
                    None => Next::Unrepresentable,
                }
            }
            _ => Next::Unknown,
        };
        Ok(chosen)
    }
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
