//! Packet-oriented MPEG-1/2 decoding. Sequence headers are validated during
//! send_packet; receive_frame incrementally reconstructs complete pictures and
//! returns display order. Flush only marks the compressed-input tail complete.
//! Two prediction anchors and a possible first field survive between packets.
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, Result, VideoFrame, VideoPlane};
use crate::{frame_assembly::FrameBuffer, sequence_extension::ChromaFormat, streaming::StreamDecoder, Error as Mpeg12Error};

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
    next_pts: i64,
    failed: bool,
}
impl Mpeg12Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self { codec_id, stream: StreamDecoder::default(), last_output: None, next_pts: 0, failed: false }
    }
    fn output_layout(&self) -> Option<(u32, u32, PixelFormat)> {
        self.last_output.or_else(|| self.stream.dimensions().map(|(w,h,c)| (w as u32,h as u32,pixel_format(c))))
    }
    fn fail(&mut self, err: Mpeg12Error) -> Error {
        // Damaged reference/quantizer state cannot leak into a later epoch.
        self.stream = StreamDecoder::default();
        self.failed = true;
        map_err(err)
    }
}
impl Decoder for Mpeg12Decoder {
    fn codec_id(&self) -> &CodecId { &self.codec_id }
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.failed { return Err(Error::invalid("mpeg12video: reset required after decode error")); }
        self.stream.push(&packet.data, packet.pts).map_err(|err| self.fail(err))
    }
    fn receive_frame(&mut self) -> Result<Frame> {
        if self.failed { return Err(Error::invalid("mpeg12video: reset required after decode error")); }
        let Some(output) = self.stream.next().map_err(|err| self.fail(err))? else {
            return Err(if self.stream.is_drained() { Error::Eof } else { Error::NeedMore });
        };
        let frame = &output.decoded.frame;
        self.last_output = Some((frame.width as u32, frame.height as u32, pixel_format(frame.chroma_format)));
        let pts = output.pts.or(Some(self.next_pts));
        self.next_pts += 1;
        let mut vf = match std::sync::Arc::try_unwrap(output) {
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
        self.stream = StreamDecoder::default();
        self.last_output = None;
        self.next_pts = 0;
        self.failed = false;
        Ok(())
    }
    fn output_pixel_format(&self) -> Option<PixelFormat> { self.output_layout().map(|(_,_,c)| c) }
    fn output_video_dimensions(&self) -> Option<(u32,u32)> { self.output_layout().map(|(w,h,_)| (w,h)) }
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
