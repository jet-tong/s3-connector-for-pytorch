/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Bound, PyRef, PyRefMut, PyResult, Python};
use mountpoint_s3_client::types::GetBodyPart;
use bytes::Bytes;
use std::cmp;

use crate::exception::S3Exception;
use crate::mountpoint_s3_client_inner::MPGetObjectClosure;
use crate::segmented_buffer::SegmentedBuffer;

#[pyclass(
    name = "GetObjectStream",
    module = "s3torchconnectorclient._mountpoint_s3_client"
)]
pub struct GetObjectStream {
    next_part: MPGetObjectClosure,
    offset: u64,
    #[pyo3(get)]
    bucket: String,
    #[pyo3(get)]
    key: String,
    leftover: Option<Bytes>,     // Zero-copy slice of previous chunk
    leftover_pos: u64,           // Absolute position of leftover data
}

impl GetObjectStream {
    pub(crate) fn new(next_part: MPGetObjectClosure, bucket: String, key: String, start_offset: Option<u64>) -> Self {
        Self {
            next_part,
            offset: start_offset.unwrap_or(0),
            bucket,
            key,
            leftover: None,
            leftover_pos: 0,
        }
    }
}

#[pymethods]
impl GetObjectStream {
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyBytes>>> {
        let py = slf.py();

        let body_part = (slf.next_part)(py)?;
        match body_part {
            None => Ok(None),
            Some(GetBodyPart { offset, data }) => {
                if offset != slf.offset {
                    return Err(S3Exception::new_err(
                        "Data from S3 was returned out of order!",
                    ));
                }
                slf.offset += data.len() as u64;
                let data = PyBytes::new(py, data.as_ref());
                Ok(Some(data))
            }
        }
    }

    pub fn tell(slf: PyRef<'_, Self>) -> u64 {
        slf.offset
    }

    /// Zero-copy stream consumption into SegmentedBuffer for single range
    /// Handles leftover data from previous reads to maintain positioning
    pub fn consume_into_buffer(
        &mut self, 
        py: Python, 
        buffer: &mut SegmentedBuffer, 
        start: u64, 
        end: u64
    ) -> PyResult<usize> {
        let mut total_bytes = 0;
        let mut current_pos = self.leftover_pos;
        
        // 1. Handle leftover data first (if any overlaps with [start, end))
        if let Some(leftover_data) = self.leftover.take() {
            let leftover_len = leftover_data.len() as u64;
            let leftover_end = current_pos + leftover_len;
            
            if current_pos < end && leftover_end > start {
                let take_start = start.saturating_sub(current_pos) as usize;
                let take_end = cmp::min(leftover_len as usize, (end - current_pos) as usize);
                
                if take_end > take_start {
                    // ZERO-COPY: Use leftover slice
                    let leftover_slice = leftover_data.slice(take_start..take_end);
                    buffer.append_bytes(leftover_slice)?;
                    total_bytes += take_end - take_start;
                }
                
                // Store remaining leftover if any
                if (take_end as u64) < leftover_len {
                    self.leftover = Some(leftover_data.slice(take_end..));
                    self.leftover_pos = current_pos + take_end as u64;
                } else {
                    self.leftover = None;
                }
            } else {
                // Leftover doesn't overlap - keep it for later
                self.leftover = Some(leftover_data);
            }
            
            current_pos = leftover_end;
        }
        
        // 2. Process new chunks from stream
        while total_bytes < (end - start) as usize {
            let body_part = match (self.next_part)(py)? {
                Some(part) => part,
                None => break, // End of stream
            };
            
            let GetBodyPart { offset, data } = body_part;
            
            // Validate sequential order
            if offset != self.offset {
                return Err(S3Exception::new_err("Data from S3 was returned out of order!"));
            }
            
            let chunk_len = data.len() as u64;
            let chunk_end = current_pos + chunk_len;
            
            // Skip chunks that don't overlap with our range
            if current_pos >= end {
                // Store entire chunk as leftover
                self.leftover = Some(data);
                self.leftover_pos = current_pos;
                break;
            }
            
            // Process overlapping portion
            if current_pos < end && chunk_end > start {
                let take_start = start.saturating_sub(current_pos) as usize;
                let take_end = cmp::min(chunk_len as usize, (end - current_pos) as usize);
                
                if take_end > take_start {
                    // ZERO-COPY: Direct Bytes slice
                    let bytes_slice = data.slice(take_start..take_end);
                    buffer.append_bytes(bytes_slice)?;
                    total_bytes += take_end - take_start;
                }
                
                // Store leftover if chunk extends beyond our range
                if (take_end as u64) < chunk_len {
                    self.leftover = Some(data.slice(take_end..));
                    self.leftover_pos = current_pos + take_end as u64;
                }
            }
            
            // Update positions
            self.offset += chunk_len;
            current_pos += chunk_len;
            
            // Early exit if we've read everything we need
            if current_pos >= end {
                break;
            }
        }
        
        Ok(total_bytes)
    }

    /// Zero-copy batch processing for multiple ranges in single stream pass
    /// Optimal for processing multiple DCP items from same RangeGroup
    pub fn consume_multiple_ranges(
        &mut self, 
        py: Python, 
        ranges: Vec<(u64, u64)>
    ) -> PyResult<Vec<SegmentedBuffer>> {
        let mut buffers: Vec<SegmentedBuffer> = ranges.iter()
            .map(|_| SegmentedBuffer::new())
            .collect();
        let mut stream_pos = self.offset;
        
        // Sort ranges by start position for efficient processing
        let mut sorted_ranges: Vec<(usize, u64, u64)> = ranges.iter()
            .enumerate()
            .map(|(i, &(start, end))| (i, start, end))
            .collect();
        sorted_ranges.sort_by_key(|&(_, start, _)| start);
        
        while let Some(body_part) = (self.next_part)(py)? {
            let GetBodyPart { offset, data } = body_part;
            
            if offset != self.offset {
                return Err(S3Exception::new_err("Data from S3 was returned out of order!"));
            }
            
            let chunk_len = data.len() as u64;
            let chunk_end = stream_pos + chunk_len;
            
            // Process all ranges that overlap with this chunk
            for &(buffer_idx, range_start, range_end) in &sorted_ranges {
                if stream_pos < range_end && chunk_end > range_start {
                    let take_start = range_start.saturating_sub(stream_pos) as usize;
                    let take_end = cmp::min(chunk_len as usize, (range_end - stream_pos) as usize);
                    
                    if take_end > take_start {
                        // ZERO-COPY: Direct Bytes slice for each range
                        let bytes_slice = data.slice(take_start..take_end);
                        buffers[buffer_idx].append_bytes(bytes_slice)?;
                    }
                }
            }
            
            self.offset += chunk_len;
            stream_pos += chunk_len;
            
            // Early exit when all ranges processed
            if let Some(&(_, _, max_end)) = sorted_ranges.last() {
                if stream_pos >= max_end {
                    break;
                }
            }
        }
        
        Ok(buffers)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::types::IntoPyDict;
    use pyo3::{py_run, PyResult, Python};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use crate::mock_client::PyMockClient;
    use crate::mountpoint_s3_client::MountpointS3Client;

    #[test]
    fn test_get_object() -> PyResult<()> {
        let layer = tracing_subscriber::fmt::layer().with_ansi(true);
        let registry = tracing_subscriber::registry().with(layer);
        let _ = registry.try_init();

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let locals = [
                (
                    "MountpointS3Client",
                    py.get_type::<MountpointS3Client>(),
                ),
                (
                    "MockMountpointS3Client",
                    py.get_type::<PyMockClient>(),
                ),
            ];

            py_run!(
                py,
                *locals.into_py_dict(py).unwrap(),
                r#"
                mock_client = MockMountpointS3Client("us-east-1", "mock-bucket")
                client = mock_client.create_mocked_client()

                mock_client.add_object("key", b"data")
                stream = client.get_object("mock-bucket", "key")

                returned_data = b''.join(stream)
                assert returned_data == b"data"
                "#
            );
        });

        Ok(())
    }
}
