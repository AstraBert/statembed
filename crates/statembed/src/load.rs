//! Utilities for loading Safetensors model files.
//!
//! This module handles reading the binary Safetensors format, extracting tensor
//! metadata, and returning the raw tensor bytes. Files are memory-mapped, so the
//! tensor bytes are not copied: the operating system reads a page from disk the
//! first time it is used.

use std::{
    fs::File,
    ops::{Deref, Range},
    path::PathBuf,
    sync::Arc,
};

use crate::errors::LoadError;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

/// Supported data types for tensor elements.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Hash)]
pub enum DataType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,
}

impl DataType {
    /// Returns the size of each element in bytes.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_size(&self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::BOOL => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I8 => 1,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::F16 => 2,
        }
    }
}

/// The raw bytes of a tensor, borrowed from a memory-mapped Safetensors file.
///
/// It owns the mapping (through an `Arc`, so clones share it) and dereferences to the
/// bytes of the tensor only, without the file header. The file is unmapped when the
/// last clone is dropped.
#[derive(Debug, Clone)]
pub struct TensorBytes {
    mmap: Arc<Mmap>,
    range: Range<usize>,
}

impl Deref for TensorBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.mmap[self.range.clone()]
    }
}

/// Metadata describing a single tensor inside a Safetensors file.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq)]
pub struct TensorDetails {
    /// The element data type of the tensor.
    pub dtype: DataType,
    /// The shape of the tensor as `[rows, cols]`.
    pub shape: [u64; 2],
    /// Byte offsets `[start, end)` into the file where the tensor data lives.
    pub data_offsets: [u64; 2],
}

/// Parses the JSON header of a Safetensors file and returns the first
/// non-metadata tensor's details.
fn header_to_details(header: &[u8]) -> Result<TensorDetails, LoadError> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header)?;
    for (k, v) in map {
        if k == "__metadata__" {
            continue;
        }
        match serde_json::from_value::<TensorDetails>(v) {
            Ok(td) => return Ok(td),
            Err(_) => continue,
        }
    }

    Err(LoadError {
        cause: "Could not find tensor details for the current safetensors model".to_string(),
    })
}

/// Number of bytes at the start of a Safetensors file that give the size of the JSON header.
const HEADER_SIZE_BYTES: usize = size_of::<u64>();

/// Loads a Safetensors file using memory mapping.
///
/// Returns the tensor metadata and the tensor bytes, which are not copied: they are read
/// from disk when first used. Returns an error if the file is too short or if the header
/// or the tensor offsets do not fit in the file.
///
/// The file must not be modified or truncated by another process while it is mapped.
pub fn load_safetensors_file(
    path: impl Into<PathBuf>,
) -> Result<(TensorDetails, TensorBytes), LoadError> {
    let file = File::open(path.into())?;
    // SAFETY: the mapping is only read. Modifying the file while it is mapped is
    // not supported (see the function documentation).
    let mmap = unsafe { Mmap::map(&file)? };
    let file_len = mmap.len();

    let header_size_bytes = mmap.get(..HEADER_SIZE_BYTES).ok_or_else(|| LoadError {
        cause: format!(
            "File is too short ({file_len} bytes) to contain the size of the safetensors header"
        ),
    })?;
    let header_size = u64::from_le_bytes(header_size_bytes.try_into().map_err(|e| LoadError {
        cause: format!("Could not parse the first 8 bytes to u64 integer: {}", e),
    })?);
    let header_end = usize::try_from(header_size)
        .ok()
        .and_then(|size| size.checked_add(HEADER_SIZE_BYTES))
        .filter(|end| *end <= file_len)
        .ok_or_else(|| LoadError {
            cause: format!(
                "Safetensors header size ({header_size} bytes) does not fit in a file of {file_len} bytes"
            ),
        })?;
    let details = header_to_details(&mmap[HEADER_SIZE_BYTES..header_end])?;

    // offsets in the header are relative to the end of the header
    let absolute = |offset: u64| {
        usize::try_from(offset)
            .ok()
            .and_then(|offset| header_end.checked_add(offset))
    };
    let [start_offset, end_offset] = details.data_offsets;
    let (Some(start), Some(end)) = (absolute(start_offset), absolute(end_offset)) else {
        return Err(LoadError {
            cause: format!(
                "Tensor offsets [{start_offset}, {end_offset}) overflow the address space"
            ),
        });
    };
    if end > file_len || start > end {
        return Err(LoadError {
            cause: format!(
                "Tensor bytes [{start}, {end}) do not fit in a file of {file_len} bytes"
            ),
        });
    }
    Ok((
        details,
        TensorBytes {
            mmap: Arc::new(mmap),
            range: Range { start, end },
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_safetensors() {
        let (details, tensor) = load_safetensors_file("testfiles/model.safetensors")
            .expect("Should be able to load the safetensors model");
        // expected header: {"embeddings":{"dtype":"F32","shape":[29528,256],"data_offsets":[0,30236672]}}
        let expected_details = TensorDetails {
            data_offsets: [0, 30236672],
            shape: [29528, 256],
            dtype: DataType::F32,
        };
        assert_eq!(details, expected_details);
        assert_eq!(
            tensor.len() as u64,
            expected_details.data_offsets[1] - expected_details.data_offsets[0]
        );
    }

    /// Writes `bytes` to a file in the temp directory and returns its path.
    fn write_temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("statembed_{}_{name}", std::process::id()));
        std::fs::write(&path, bytes).expect("Should write the temp file");
        path
    }

    /// A safetensors file made of the header size, the header, and `data`.
    fn safetensors_bytes(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        bytes
    }

    #[test]
    fn test_load_safetensors_too_short_file_fails() {
        let path = write_temp_file("too_short", &[1, 2, 3]);
        let err = load_safetensors_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.cause.contains("too short"), "{}", err.cause);
    }

    #[test]
    fn test_load_safetensors_header_larger_than_file_fails() {
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend(b"{}");
        let path = write_temp_file("big_header", &bytes);
        let err = load_safetensors_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.cause.contains("header size"), "{}", err.cause);
    }

    #[test]
    fn test_load_safetensors_offsets_beyond_file_fails() {
        let header = r#"{"embeddings":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        // only 8 bytes of data for a tensor that claims 16
        let path = write_temp_file("short_data", &safetensors_bytes(header, &[0u8; 8]));
        let err = load_safetensors_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.cause.contains("do not fit"), "{}", err.cause);
    }

    #[test]
    fn test_load_safetensors_huge_offsets_fail_without_overflow() {
        let header = format!(
            r#"{{"embeddings":{{"dtype":"F32","shape":[2,2],"data_offsets":[0,{}]}}}}"#,
            u64::MAX
        );
        let path = write_temp_file("huge_offsets", &safetensors_bytes(&header, &[0u8; 16]));
        let err = load_safetensors_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.cause.contains("overflow"), "{}", err.cause);
    }
}
