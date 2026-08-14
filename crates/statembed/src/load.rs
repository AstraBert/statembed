//! Utilities for loading Safetensors model files.
//!
//! This module handles reading the binary Safetensors format, extracting tensor
//! metadata, and returning the raw tensor bytes. It supports both memory-mapped
//! and standard file I/O via feature flags.

use std::{fs::File, path::PathBuf};

use crate::errors::LoadError;
#[cfg(feature = "mmap")]
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
#[cfg(not(feature = "mmap"))]
use std::io::Read;

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

/// Loads a Safetensors file using memory mapping.
///
/// Returns the tensor metadata and a copy of the raw tensor bytes.
#[cfg(feature = "mmap")]
pub fn load_safetensors_file(
    path: impl Into<PathBuf>,
) -> Result<(TensorDetails, Vec<u8>), LoadError> {
    let file = File::open(path.into())?;
    let mmap = unsafe { Mmap::map(&file)? };
    let (header_size_bytes, rest) = mmap.split_at(size_of::<u64>());
    let header_size = u64::from_le_bytes(header_size_bytes.try_into().map_err(|e| LoadError {
        cause: format!("Could not parse the first 8 bytes to u64 integer: {}", e),
    })?);
    let (json_str, rest_tensor) = rest.split_at(header_size as usize);
    let details = header_to_details(json_str)?;
    let tensor =
        &rest_tensor[(details.data_offsets[0] as usize)..(details.data_offsets[1] as usize)];
    Ok((details, tensor.to_vec()))
}

/// Loads a Safetensors file using standard file I/O.
///
/// Returns the tensor metadata and a copy of the raw tensor bytes.
#[cfg(not(feature = "mmap"))]
pub fn load_safetensors_file(
    path: impl Into<PathBuf>,
) -> Result<(TensorDetails, Vec<u8>), LoadError> {
    let mut file = File::open(path.into())?;
    let mut content: Vec<u8> = vec![];
    file.read_to_end(&mut content)?;
    let (header_size_bytes, rest) = content.split_at(size_of::<u64>());
    let header_size = u64::from_le_bytes(header_size_bytes.try_into().map_err(|e| LoadError {
        cause: format!("Could not parse the first 8 bytes to u64 integer: {}", e),
    })?);
    let (json_str, rest_tensor) = rest.split_at(header_size as usize);
    let details = header_to_details(json_str)?;
    let tensor =
        &rest_tensor[(details.data_offsets[0] as usize)..(details.data_offsets[1] as usize)];
    Ok((details, tensor.to_vec()))
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
}
