use std::fs::File;

use crate::errors::LoadError;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct TensorDetails {
    pub dtype: DataType,
    pub shape: [u64; 2],
    pub data_offsets: [u64; 2],
}

fn header_to_details(header: &[u8]) -> Result<TensorDetails, LoadError> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header)?;
    for (k, v) in map {
        if k == "__metadata__" {
            continue;
        }
        let details: TensorDetails = serde_json::from_value(v)?;
        return Ok(details);
    }

    Err(LoadError {
        cause: "Could not find tensor details for the current safetensors model".to_string(),
    })
}

pub fn load_safetensors_file(path: &str) -> Result<(TensorDetails, Vec<u8>), LoadError> {
    let file = File::open(path)?;
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
