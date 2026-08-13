//! `statemebed`: Fast, lightweight static text embeddings.
//!
//! This library loads pre-trained static embedding models stored in the
//! Safetensors format and optionally tokenizes input text using Hugging Face
//! `tokenizers`. Embeddings are produced via mean-pooling over token-level
//! vectors and can optionally be L2-normalized.
//!
//! # Example
//! ```no_run
//! use statemebed::StaticEmbedding;
//!
//! let mut model = StaticEmbedding::from_dir("./my-model", Some(true)).unwrap();
//! let embedding = model.embed_text("hello world", None).unwrap();
//! ```

#[cfg(feature = "tokenizers")]
use crate::tokenize::tokenize;
#[cfg(feature = "tokenizers")]
use crate::{errors::TokenizationError, tokenize::load_tokenizer};
#[cfg(feature = "hf-hub")]
use hf_hub::{HFClient, RepoTypeModel};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "hf-hub")]
use std::sync::OnceLock;
#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

use crate::{
    errors::{EmbedError, InvalidModelOrPathError, LoadError},
    load::{DataType, TensorDetails, load_safetensors_file},
};

pub mod errors;
mod load;
#[cfg(feature = "tokenizers")]
mod tokenize;

/// Files that are downloaded when fetching a model from the Hugging Face Hub.
#[cfg(feature = "hf-hub")]
pub const DOWNLOAD_FILES: &[&str] = &["model.safetensors", "tokenizer.json"];
/// Global cache directory for models downloaded from the Hugging Face Hub.
#[cfg(feature = "hf-hub")]
pub static HF_CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Returns the global Hugging Face cache directory
/// for statembed (`~/.statembed`)
#[cfg(feature = "hf-hub")]
pub fn hf_cache_dir() -> &'static PathBuf {
    HF_CACHE_DIR.get_or_init(|| {
        dirs::home_dir()
            .expect("No home dir could be found for the current environment")
            .join(".statembed")
    })
}

/// Decodes a little-endian byte chunk into an `f32` value according to `dtype`.
fn decode(chunk: &[u8], dtype: DataType) -> Result<f32, EmbedError> {
    let fl = match dtype {
        DataType::BF16 => half::bf16::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to BF16 conversion error: {}", e),
        })?)
        .to_f32(),
        DataType::BOOL => {
            let val = unsafe { *chunk.get_unchecked(0) };
            val as f32
        }
        DataType::F32 => f32::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to F32 conversion error: {}", e),
        })?),
        DataType::F64 => f64::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to F64 conversion error: {}", e),
        })?) as f32,
        DataType::F16 => half::f16::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to F16 conversion error: {}", e),
        })?)
        .to_f32(),
        DataType::I16 => i16::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to I16 conversion error: {}", e),
        })?) as f32,
        DataType::I32 => i32::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to I32 conversion error: {}", e),
        })?) as f32,
        DataType::I64 => i64::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to I64 conversion error: {}", e),
        })?) as f32,
        DataType::I8 => i8::from_le_bytes(chunk.try_into().map_err(|e| EmbedError {
            cause: format!("Bytes to I8 conversion error: {}", e),
        })?) as f32,
        DataType::U8 => {
            let val = unsafe { *chunk.get_unchecked(0) };
            val as f32
        }
    };
    Ok(fl)
}

/// Performs mean-pooling over token embeddings extracted from a flat tensor buffer.
///
/// Token embeddings are cached in `token_map` to avoid re-decoding the same token.
fn sequential_mean_pooling(
    tensor: &[u8],
    tokens: Vec<u32>,
    dim: u64,
    dtype: DataType,
    dtype_size: usize,
    token_map: &mut HashMap<u32, Vec<f32>>,
) -> Result<Vec<f32>, EmbedError> {
    let dim_usize = dim as usize;
    let n = tokens.len() as f32;
    let mut mean_pooled: Vec<f32> = vec![0f32; dim_usize];

    for tok in tokens {
        if token_map.contains_key(&tok) {
            for (j, fl) in token_map[&tok].iter().enumerate() {
                unsafe {
                    *mean_pooled.get_unchecked_mut(j) += fl;
                }
            }
            continue;
        }
        let start = (tok as u64) * dim * (dtype_size as u64);
        let finish = start + (dim * (dtype_size as u64));
        let tok_repr = &tensor[(start as usize)..(finish as usize)];
        let mut decoded: Vec<f32> = Vec::with_capacity(dim_usize);
        for (j, tr) in tok_repr.chunks(dtype_size).enumerate() {
            let fl = decode(tr, dtype)?;
            decoded.push(fl);
            unsafe {
                *mean_pooled.get_unchecked_mut(j) += fl;
            }
        }
        token_map.insert(tok, decoded);
    }

    for x in &mut mean_pooled {
        *x /= n;
    }

    Ok(mean_pooled)
}

/// A static embedding model loaded from disk.
///
/// `StaticEmbedding` lazily loads the underlying tensor and tokenizer on first
/// use, then caches them for subsequent calls. It supports mean-pooled
/// embeddings with optional L2 normalization.
pub struct StaticEmbedding {
    /// Filesystem path to the model directory.
    pub base_path: PathBuf,
    /// Metadata about the loaded tensor (shape, dtype, offsets).
    pub tensor_details: Option<TensorDetails>,
    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,
    tensor: Option<Vec<u8>>,
    #[cfg(feature = "tokenizers")]
    tokenizer: Option<Tokenizer>,
    #[cfg(feature = "tokenizers")]
    median_token_length: Option<usize>,
    #[cfg(feature = "tokenizers")]
    unknown_token: Option<u32>,
    token_map: HashMap<u32, Vec<f32>>,
}

impl StaticEmbedding {
    /// Creates a `StaticEmbedding` from a local directory.
    ///
    /// The directory must contain at least `model.safetensors`. When the
    /// `tokenizers` feature is enabled, `tokenizer.json` is also required.
    ///
    /// # Arguments
    /// * `path` - Path to the model directory.
    /// * `normalize` - If `Some(true)`, output embeddings will be L2-normalized.
    pub fn from_dir<T: AsRef<Path>>(
        path: T,
        normalize: Option<bool>,
    ) -> Result<Self, InvalidModelOrPathError> {
        let p: &Path = path.as_ref();
        if !p.join("model.safetensors").exists() {
            return Err(InvalidModelOrPathError {
                model_or_path: p.to_string_lossy().to_string(),
                details: "Could not find `models.safetensors` in the specified directory"
                    .to_string(),
            });
        }

        #[cfg(feature = "tokenizers")]
        if !p.join("tokenizer.json").exists() {
            return Err(InvalidModelOrPathError {
                model_or_path: p.to_string_lossy().to_string(),
                details: "Could not find `tokenizer.json` in the specified directory".to_string(),
            });
        }

        Ok(Self {
            base_path: p.to_owned(),
            normalize: normalize.unwrap_or_default(),
            #[cfg(feature = "tokenizers")]
            tokenizer: None,
            tensor: None,
            tensor_details: None,
            #[cfg(feature = "tokenizers")]
            median_token_length: None,
            #[cfg(feature = "tokenizers")]
            unknown_token: None,
            token_map: HashMap::new(),
        })
    }

    /// Downloads a model from the Hugging Face Hub and returns a `StaticEmbedding`.
    ///
    /// # Arguments
    /// * `model_id` - Hugging Face model identifier in `owner/repo_name` format.
    /// * `normalize` - If `Some(true)`, output embeddings will be L2-normalized.
    /// * `force_download` - If `true`, re-downloads files even if they already exist locally.
    #[cfg(feature = "hf-hub")]
    pub async fn from_hf_hub(
        model_id: &str,
        normalize: Option<bool>,
        force_download: bool,
    ) -> Result<Self, InvalidModelOrPathError> {
        let client = HFClient::new().map_err(|e| InvalidModelOrPathError {
            model_or_path: model_id.to_string(),
            details: format!("Could not load HF client. Error: {}", e),
        })?;
        let split = model_id.split_once("/");
        if let Some((owner, name)) = split {
            let repo = client.repository(RepoTypeModel, owner, name);
            let base_path = hf_cache_dir().join(model_id.replace("/", "--"));
            for f in DOWNLOAD_FILES {
                // skip downloading if already there, unless we want to forcibly re-download
                if base_path.join(f).exists() && !force_download {
                    continue;
                }
                repo.download_file()
                    .filename(f.to_string())
                    .local_dir(&base_path)
                    .send()
                    .await
                    .map_err(|e| InvalidModelOrPathError {
                        model_or_path: model_id.to_string(),
                        details: format!("Could not download file {}. Error: {}", f, e),
                    })?;
            }
            return Ok(Self {
                base_path,
                normalize: normalize.unwrap_or_default(),
                #[cfg(feature = "tokenizers")]
                tokenizer: None,
                tensor: None,
                tensor_details: None,
                #[cfg(feature = "tokenizers")]
                median_token_length: None,
                #[cfg(feature = "tokenizers")]
                unknown_token: None,
                token_map: HashMap::new(),
            });
        }

        Err(InvalidModelOrPathError {
            model_or_path: model_id.to_string(),
            details: "Model ID should be reported as owner/repo_name".to_string(),
        })
    }

    /// Loads the `model.safetensors` tensor into memory.
    fn load_tensor(&mut self) -> Result<(), LoadError> {
        let (details, tensor) = load_safetensors_file(self.base_path.join("model.safetensors"))?;
        self.tensor = Some(tensor);
        self.tensor_details = Some(details);
        Ok(())
    }

    /// Loads the `tokenizer.json` and extracts vocabulary statistics.
    #[cfg(feature = "tokenizers")]
    fn load_tokenizer(&mut self) -> Result<(), TokenizationError> {
        use crate::tokenize::extract_tokenizer_details;

        let tokenizer = load_tokenizer(self.base_path.join("tokenizer.json"))?;
        let (median_length, unk_tok) = extract_tokenizer_details(&tokenizer)?;
        self.tokenizer = Some(tokenizer);
        self.median_token_length = Some(median_length);
        self.unknown_token = unk_tok;
        Ok(())
    }

    /// Generates an embedding for a pre-tokenized sequence of token IDs.
    ///
    /// The tensor is loaded lazily on first call. Embeddings are mean-pooled
    /// and optionally normalized.
    pub fn embed_tokens(&mut self, tokens: Vec<u32>) -> Result<Vec<f32>, EmbedError> {
        if self.tensor.is_none() {
            self.load_tensor()?;
        }
        if let Some(t) = self.tensor.as_deref() {
            let (dtype, dtype_size, dim) = {
                let td = self.tensor_details.unwrap();
                (td.dtype, td.dtype.to_size(), td.shape[1])
            };
            let mut mean_pooled: Vec<f32> =
                sequential_mean_pooling(t, tokens, dim, dtype, dtype_size, &mut self.token_map)?;

            if self.normalize {
                let norm = mean_pooled
                    .iter()
                    .map(|&v| v * v)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                for x in &mut mean_pooled {
                    *x /= norm;
                }
            }
            return Ok(mean_pooled);
        }
        Err(EmbedError {
            cause: "Tensor should be non-null at this point".to_string(),
        })
    }

    /// Truncates `text` to an approximate byte length based on the median token length.
    ///
    /// This is a fast, heuristic truncation that avoids running the full tokenizer.
    #[cfg(feature = "tokenizers")]
    fn truncate_input<'a>(&'a self, text: &'a str, max_token_length: usize) -> &'a str {
        text.char_indices()
            // median_token_length is always guaranteed to be non-null,
            // it just needs to be null for initialization.
            // For all the callsites of this methods, median_token_length
            // has already been assigned to a non-null value and thus can be unwrapped safely
            .nth(max_token_length.saturating_mul(self.median_token_length.unwrap()))
            .map_or(text, |(byte_idx, _)| &text[..byte_idx])
    }

    /// Tokenizes `input_text` and returns its embedding.
    ///
    /// The tokenizer and tensor are loaded lazily on first call. If
    /// `max_token_length` is provided, the input is heuristically truncated
    /// before tokenization.
    #[cfg(feature = "tokenizers")]
    pub fn embed_text(
        &mut self,
        input_text: &str,
        max_token_length: Option<usize>,
    ) -> Result<Vec<f32>, EmbedError> {
        if self.tokenizer.is_none() {
            self.load_tokenizer()?;
        }
        let text = match max_token_length {
            Some(m) => self.truncate_input(input_text, m),
            None => input_text,
        };
        if let Some(tk) = self.tokenizer.as_ref() {
            let tokens = tokenize(tk, text, self.unknown_token)?;
            let embedding = self.embed_tokens(tokens)?;
            return Ok(embedding);
        }
        Err(EmbedError {
            cause: "Tokenizer should be non-null at this point".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_directory() {
        let _ = StaticEmbedding::from_dir("testfiles/", None)
            .expect("Should be able to load model from testfiles");
    }

    #[test]
    fn test_load_tensor() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None)
            .expect("Should be able to load model from testfiles");
        model.load_tensor().expect("Should be able to load tensor");
        assert!(model.tensor.is_some());
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_load_tokenizer() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None)
            .expect("Should be able to load model from testfiles");
        model
            .load_tokenizer()
            .expect("Should be able to load tokenizer");
        assert!(model.tokenizer.is_some());
        assert!(model.unknown_token.is_some());
        assert!(model.median_token_length.is_some());
    }

    #[test]
    fn test_decode_succeeds_for_all_dtypes() {
        // BF16: 1.0 in bf16 le bytes
        let bf16_bytes = half::bf16::from_f32(1.0).to_le_bytes();
        assert_eq!(decode(&bf16_bytes, DataType::BF16).unwrap(), 1.0);

        // BOOL: single byte, non-zero -> 1.0
        let bool_bytes = [1u8];
        assert_eq!(decode(&bool_bytes, DataType::BOOL).unwrap(), 1.0);

        // F32
        let f32_bytes = 3.5f32.to_le_bytes();
        assert_eq!(decode(&f32_bytes, DataType::F32).unwrap(), 3.5);

        // F64 -> cast down to f32
        let f64_bytes = 2.25f64.to_le_bytes();
        assert_eq!(decode(&f64_bytes, DataType::F64).unwrap(), 2.25f32);

        // F16
        let f16_bytes = half::f16::from_f32(4.0).to_le_bytes();
        assert_eq!(decode(&f16_bytes, DataType::F16).unwrap(), 4.0);

        // I16
        let i16_bytes = (-42i16).to_le_bytes();
        assert_eq!(decode(&i16_bytes, DataType::I16).unwrap(), -42.0);

        // I32
        let i32_bytes = 12345i32.to_le_bytes();
        assert_eq!(decode(&i32_bytes, DataType::I32).unwrap(), 12345.0);

        // I64
        let i64_bytes = (-987654321i64).to_le_bytes();
        assert_eq!(decode(&i64_bytes, DataType::I64).unwrap(), -987654321.0);

        // I8
        let i8_bytes = (-7i8).to_le_bytes();
        assert_eq!(decode(&i8_bytes, DataType::I8).unwrap(), -7.0);

        // U8
        let u8_bytes = [200u8];
        assert_eq!(decode(&u8_bytes, DataType::U8).unwrap(), 200.0);
    }

    #[test]
    fn test_decode_fails_on_wrong_chunk_length() {
        // F32 expects exactly 4 bytes; give it 3.
        let bad_chunk = [0u8, 1u8, 2u8];
        let result = decode(&bad_chunk, DataType::F32);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.cause.contains("Bytes to F32 conversion error"),
            "unexpected error message: {}",
            err.cause
        );
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_token_map_populated() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None)
            .expect("Should be able to load model from testfiles");
        let _ = model.embed_text("hello there!", None);
        assert!(model.token_map.len() > 0);
        // token for 'hello'
        assert!(model.token_map.contains_key(&6598))
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_truncate_input_text() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None)
            .expect("Should be able to load model from testfiles");
        model
            .load_tokenizer()
            .expect("Should load tokenizer without problems");
        let original = "this is a very very long text";
        let truncated = model.truncate_input(original, 3);
        assert!(original.len() > truncated.len())
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_truncate_input_text_no_truncate() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None)
            .expect("Should be able to load model from testfiles");
        model
            .load_tokenizer()
            .expect("Should load tokenizer without problems");
        let original = "this is a very very long text";
        let truncated = model.truncate_input(original, 30);
        assert!(original.len() == truncated.len())
    }
}
