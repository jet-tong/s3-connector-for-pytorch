/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use bytes::Bytes;
use std::cmp;

/// Zero-copy segmented buffer that stores Bytes directly without PyBytes conversion.
/// Optimized for sequential access patterns with segment caching.
#[pyclass(
    name = "SegmentedBuffer",
    module = "s3torchconnectorclient._mountpoint_s3_client"
)]
pub struct SegmentedBuffer {
    segments: Vec<Bytes>,        // Direct Bytes storage (ref-counted, zero-copy)
    offsets: Vec<usize>,         // Segment start positions within buffer
    lengths: Vec<usize>,         // Segment lengths for fast access
    size: usize,                 // Total buffer size (sum of all segments)
    cursor: usize,               // Current read position
    last_seg_idx: usize,         // Cache for sequential access optimization
}

impl Default for SegmentedBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl SegmentedBuffer {
    #[new]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            offsets: Vec::new(),
            lengths: Vec::new(),
            size: 0,
            cursor: 0,
            last_seg_idx: 0,
        }
    }



    /// Read data into a pre-allocated buffer (Python buffer protocol)
    /// Handles memoryview objects from PyTorch DCP
    pub fn readinto(&mut self, _py: Python, buffer: &pyo3::Bound<'_, pyo3::PyAny>) -> PyResult<usize> {
        // Use PyO3's buffer protocol to extract writable buffer
        let buf = pyo3::buffer::PyBuffer::<u8>::get(buffer)?;
        if buf.readonly() {
            return Err(pyo3::exceptions::PyTypeError::new_err("writable buffer required"));
        }
        
        // Get mutable slice from buffer
        let buf_slice = unsafe {
            std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut u8, buf.len_bytes())
        };
        
        // Use existing optimized readinto implementation
        self.readinto_internal(buf_slice)
    }
    


    /// Read data and return as PyBytes (compatible with Python read() method)
    pub fn read(&mut self, py: Python, size: Option<usize>) -> PyResult<Py<PyBytes>> {
        let size = size.unwrap_or(self.size - self.cursor);
        if size == 0 {
            return Ok(PyBytes::new(py, &[]).into());
        }

        let mut buf = vec![0u8; size];
        let bytes_read = self.readinto_internal(&mut buf)?;
        
        if bytes_read == size {
            Ok(PyBytes::new(py, &buf).into())
        } else {
            Ok(PyBytes::new(py, &buf[..bytes_read]).into())
        }
    }

    /// Seek to position within buffer (compatible with Python seek() method)
    #[pyo3(signature = (offset, whence = 0))]
    pub fn seek(&mut self, offset: isize, whence: i32) -> PyResult<usize> {
        let new_pos = match whence {
            0 => offset as usize,  // SEEK_SET
            1 => (self.cursor as isize + offset) as usize,  // SEEK_CUR
            2 => (self.size as isize + offset) as usize,    // SEEK_END
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid whence value")),
        };

        self.cursor = new_pos;
        Ok(self.cursor)
    }

    /// Get current position in buffer
    pub fn tell(&self) -> usize {
        self.cursor
    }

    /// Get total buffer size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Close buffer and free resources
    pub fn close(&mut self) {
        self.segments.clear();
        self.offsets.clear();
        self.lengths.clear();
        self.size = 0;
        self.cursor = 0;
        self.last_seg_idx = 0;
    }
}

// Internal methods not exposed to Python
impl SegmentedBuffer {
    /// Append a Bytes segment to the buffer (zero-copy) - internal method
    pub(crate) fn append_bytes(&mut self, bytes: Bytes) -> PyResult<()> {
        let len = bytes.len();
        if len == 0 {
            return Ok(());
        }
        
        self.offsets.push(self.size);
        self.lengths.push(len);
        self.segments.push(bytes);
        self.size += len;
        Ok(())
    }
    
    /// Internal readinto implementation
    pub(crate) fn readinto_internal(&mut self, buffer: &mut [u8]) -> PyResult<usize> {
        let buf_bytes = buffer;
        let dest_len = buf_bytes.len();
        
        if dest_len == 0 || self.cursor >= self.size {
            return Ok(0);
        }

        let bytes_to_read = cmp::min(dest_len, self.size - self.cursor);
        let mut written = 0;
        let mut pos = self.cursor;

        // Find starting segment using cached index for O(1) sequential access
        let mut seg_idx = if self.last_seg_idx < self.offsets.len() && 
                             pos >= self.offsets[self.last_seg_idx] {
            self.last_seg_idx
        } else {
            0
        };

        // Find exact segment containing current position
        while seg_idx < self.offsets.len() && 
              pos >= self.offsets[seg_idx] + self.lengths[seg_idx] {
            seg_idx += 1;
        }

        // Fast copy loop across segments
        while written < bytes_to_read && seg_idx < self.segments.len() {
            let seg_start = self.offsets[seg_idx];
            let seg_len = self.lengths[seg_idx];
            let offset_in_seg = pos - seg_start;
            let available = seg_len - offset_in_seg;
            let copy_size = cmp::min(bytes_to_read - written, available);

            // Direct memcpy from Bytes to destination buffer
            let src = &self.segments[seg_idx][offset_in_seg..offset_in_seg + copy_size];
            buf_bytes[written..written + copy_size].copy_from_slice(src);

            written += copy_size;
            pos += copy_size;
            seg_idx += 1;
        }

        // Update cursor and cache segment index for next access
        self.cursor += written;
        self.last_seg_idx = seg_idx.saturating_sub(1);
        Ok(written)
    }
}