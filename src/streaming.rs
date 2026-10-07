//! Incremental elementary-stream framing. No decoded-output queue: reconstruction
//! advances on receive, retaining two shared anchors and at most one field.
use std::collections::VecDeque;
use crate::{Error, Result, sequence_extension::ChromaFormat};
use crate::video_sequence::{find_next_picture_boundary, PictureDecoder, Released, Stamp};

pub(crate) const MAX_BUFFERED_BYTES: usize = 32 * 1024 * 1024;
const MAX_TIMESTAMPS: usize = 4096;

#[derive(Debug, Default)]
pub(crate) struct StreamDecoder {
    buffer: Vec<u8>,
    cursor: usize,
    scan: usize,
    base: u64,
    stamps: VecDeque<(u64, Stamp)>,
    pictures: PictureDecoder,
    initial_geometry: Option<(usize, usize, ChromaFormat)>,
    eof: bool,
    drained: bool,
    seen_data: bool,
    #[cfg(test)]
    peak: usize,
}

impl StreamDecoder {
    /// Packet input that may begin at any picture (seek or mid-stream start).
    pub(crate) fn random_access() -> Self {
        Self { pictures: PictureDecoder::random_access(), ..Self::default() }
    }

    pub(crate) fn dimensions(&self) -> Option<(usize, usize, ChromaFormat)> {
        self.initial_geometry
    }

    fn compact(&mut self) {
        if self.cursor == 0 { return; }
        self.buffer.drain(..self.cursor);
        self.base += self.cursor as u64;
        self.scan = self.scan.saturating_sub(self.cursor);
        self.cursor = 0;
        // Keep the last timestamp preceding the unconsumed data: a PES can
        // carry a header before the picture to which its PTS applies.
        while self.stamps.len() > 1 && self.stamps[1].0 <= self.base {
            self.stamps.pop_front();
        }
    }

    pub(crate) fn push(&mut self, data: &[u8], stamp: Stamp) -> Result<()> {
        if self.eof { return Err(Error::InvalidBitstream("packet after flush; reset required")); }
        self.compact();
        if data.len() > MAX_BUFFERED_BYTES - self.buffer.len() {
            return Err(Error::InvalidBitstream("undrained MPEG input exceeds 32 MiB"));
        }
        if stamp != Stamp::default() {
            if self.stamps.len() == MAX_TIMESTAMPS {
                return Err(Error::InvalidBitstream("undrained MPEG timestamp limit"));
            }
            self.stamps.push_back((self.base + self.buffer.len() as u64, stamp));
        }
        self.seen_data |= !data.is_empty();
        self.buffer.extend_from_slice(data);
        #[cfg(test)] { self.peak = self.peak.max(self.buffer.len()); }
        // Publish real, validated geometry while send_packet is running.
        // Never decode pictures here: a single packet can contain many AUs.
        self.headers()?;
        self.compact();
        Ok(())
    }

    /// Move past complete sequence/GOP/user-data headers, leaving the cursor
    /// on a picture or sequence end. Partial prefixes keep at most three bytes.
    fn headers(&mut self) -> Result<()> {
        loop {
            let Some(rel) = find_next_picture_boundary(&self.buffer[self.cursor..]) else {
                self.cursor = self.buffer.len().saturating_sub(3).max(self.cursor);
                return Ok(());
            };
            self.cursor += rel;
            match self.buffer[self.cursor + 3] {
                0 => {
                    if self.pictures.dimensions().is_some() {
                        return Ok(());
                    }
                    if !self.pictures.is_random_access() {
                        return Err(Error::InvalidBitstream("picture before sequence header"));
                    }
                    // Random access: nothing decodes before a sequence header.
                    self.take_stamp();
                    self.cursor += 4;
                    self.scan = self.cursor;
                }
                0xB7 => return Ok(()),
                0xB3 => {
                    let after = match crate::sequence_extension::sequence_header_byte_length(&self.buffer[self.cursor..]) {
                        Ok(after) => after,
                        Err(Error::ShortHeader) if !self.eof => return Ok(()),
                        Err(err) => return Err(err),
                    };
                    let scan = self.scan.max(self.cursor + after);
                    let Some(next) = self.buffer[scan..].windows(4)
                        .position(|w| w[..3] == [0,0,1]).map(|p| scan + p) else {
                        if self.eof { return Err(Error::ShortHeader); }
                        self.scan = self.buffer.len().saturating_sub(3).max(self.cursor + after);
                        return Ok(());
                    };
                    self.scan = next;
                    match self.pictures.sequence(&self.buffer[self.cursor..]) {
                        Ok(()) => {
                            if self.initial_geometry.is_none() {
                                self.initial_geometry = self.pictures.dimensions();
                            }
                            self.cursor = if self.buffer[next + 3] == 0xB5 { next + 10 } else { next };
                            self.scan = self.cursor;
                        }
                        Err(Error::ShortHeader) if !self.eof => return Ok(()),
                        Err(err) => return Err(err),
                    }
                }
                0xB8 => {
                    // §6.2.2.6 / ISO/IEC 11172-2 §2.4.2.4: closed_gop follows
                    // the 25-bit time_code.
                    match self.buffer.get(self.cursor + 7) {
                        Some(&flags) => self.pictures.gop(flags & 0x40 != 0),
                        None if !self.eof => return Ok(()),
                        None => {}
                    }
                    self.cursor += 4;
                    self.scan = self.cursor;
                }
                _ => { self.cursor += 4; self.scan = self.cursor; }
            }
        }
    }

    pub(crate) fn finish(&mut self) { self.eof = true; }
    pub(crate) fn is_drained(&self) -> bool { self.drained }

    pub(crate) fn next(&mut self) -> Result<Option<Released>> {
        if self.drained { return Ok(None); }
        loop {
            self.headers()?;
            if self.buffer.len() - self.cursor < 4 {
                if !self.eof { return Ok(None); }
                if self.initial_geometry.is_none() && self.seen_data && !self.pictures.is_random_access() {
                    return Err(Error::InvalidBitstream("missing sequence header"));
                }
                self.drained = true;
                self.buffer.clear();
                self.cursor = 0;
                self.scan = 0;
                return Ok(self.pictures.end());
            }
            if self.buffer[self.cursor + 3] == 0xB7 {
                self.cursor += 4;
                self.scan = self.cursor;
                if let Some(frame) = self.pictures.end() { return Ok(Some(frame)); }
                continue;
            }
            // A sequence header without enough bytes is not a complete AU.
            if self.buffer[self.cursor + 3] != 0 { return Ok(None); }
            let scan = self.scan.max(self.cursor + 4);
            let boundary = find_next_picture_boundary(&self.buffer[scan..]).map(|p| scan + p);
            if boundary.is_none() && !self.eof {
                self.scan = self.buffer.len().saturating_sub(3).max(self.cursor + 4);
                return Ok(None);
            }
            let next = boundary.unwrap_or(self.buffer.len());
            let end = boundary.map_or(next, |p| p + 4);
            let stamp = self.take_stamp();
            // Include the boundary prefix: slice walkers use its 23 zero bits
            // as their terminator. At EOF their checked short-tail path applies.
            let result = self.pictures.picture(&self.buffer[self.cursor..end], stamp);
            self.cursor = next;
            self.scan = next;
            if let Some(frame) = result? { return Ok(Some(frame)); }
        }
    }

    /// Timestamps for the picture at the cursor: the last stamp of a packet
    /// beginning at or before it and after the previous picture. Later
    /// pictures in that packet get none.
    fn take_stamp(&mut self) -> Stamp {
        let position = self.base + self.cursor as u64;
        let mut stamp = Stamp::default();
        while self.stamps.front().is_some_and(|&(p, _)| p <= position) {
            stamp = self.stamps.pop_front().map_or(stamp, |(_, s)| s);
        }
        stamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leading_stuffing_is_discarded_and_packet_buffer_is_bounded() {
        let mut decoder = StreamDecoder::default();
        for _ in 0..4096 { decoder.push(&[0; 1024], Stamp::default()).unwrap(); }
        assert_eq!(decoder.buffer.len(), 3);
        assert!(decoder.peak <= 1027);
        // A picture without a following boundary cannot accumulate indefinitely.
        let data = include_bytes!("../tests/fixtures/conformance/mpeg2-100x62.m2v");
        decoder.push(data, Stamp::default()).unwrap();
        assert!(decoder.push(&vec![0; MAX_BUFFERED_BYTES], Stamp::default()).is_err());
        assert!(decoder.buffer.len() < MAX_BUFFERED_BYTES);
    }
}
