//! `statembed`: Fast, lightweight static text embeddings.
//!
//! This library loads pre-trained static embedding models stored in the
//! Safetensors format and tokenizes input text using Hugging Face
//! `tokenizers`. Embeddings are produced via mean-pooling over token-level
//! vectors and can optionally be L2-normalized.
//!
//! Texts are embedded in batches. With the `rayon` feature, the sequences of
//! each batch are pooled in parallel; without it, they are pooled one after the other.
//!
//! The unknown token of the tokenizer is removed before pooling. The model file is
//! memory-mapped, and it must not be modified while the model is loaded.
//!
//! # Example
//! ```ignore
//! use statembed::StaticEmbedding;
//!
//! let mut model = StaticEmbedding::from_dir("./my-model", Some(true), None, None).unwrap();
//! let embedding = model.embed_text("hello world").unwrap();
//! let embeddings = model.embed_texts(&["hello world", "goodbye world"], None).unwrap();
//! ```

use crate::errors::TokenizationError;
use crate::load::TensorBytes;
use crate::tokenize::{encode_batch, load_tokenizer};
#[cfg(feature = "hf-hub")]
use hf_hub::{HFClient, RepoTypeModel};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "hf-hub")]
use std::sync::OnceLock;
use tokenizers::TruncationParams;
use tokenizers::pipeline::PipelineTokenizer;
#[cfg(feature = "simd")]
use wide::f32x8;

use crate::{
    errors::{EmbedError, InvalidModelOrPathError, LoadError},
    load::{DataType, TensorDetails, load_safetensors_file},
};

pub mod errors;
mod load;
mod tokenize;

/// Default size (in bytes) of the decoded `f32` table (vocabulary x dimensions x 4)
/// below which the whole model is decoded at load time (256 MB).
pub const DEFAULT_EAGER_LOADING_THRESHOLD: u64 = 256 * 1024 * 1024;
/// Default number of texts that are tokenized and pooled together.
pub const DEFAULT_BATCH_SIZE: usize = 16;
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

/// Decodes one little-endian tensor row into `f32` values according to `dtype`.
///
/// The `dtype` is checked once per row, not once per number.
fn decode_row(row: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::F32 => row
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DataType::BF16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect(),
        DataType::F16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect(),
        DataType::F64 => row
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        DataType::I16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c) as f32)
            .collect(),
        DataType::I32 => row
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        DataType::I64 => row
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        DataType::I8 => row.iter().map(|&b| b as i8 as f32).collect(),
        DataType::U8 | DataType::BOOL => row.iter().map(|&b| b as f32).collect(),
    }
}

/// Decodes one little-endian tensor row into `f32` values according to `dtype`
/// and appends them to `original`, without allocating a new vector.
fn decode_row_into(row: &[u8], dtype: DataType, original: &mut Vec<f32>) {
    match dtype {
        DataType::F32 => original.extend(
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c)),
        ),
        DataType::BF16 => original.extend(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32()),
        ),
        DataType::F16 => original.extend(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32()),
        ),
        DataType::F64 => original.extend(
            row.as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c) as f32),
        ),
        DataType::I16 => original.extend(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes(*c) as f32),
        ),
        DataType::I32 => original.extend(
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as f32),
        ),
        DataType::I64 => original.extend(
            row.as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c) as f32),
        ),
        DataType::I8 => original.extend(row.iter().map(|&b| b as i8 as f32)),
        DataType::U8 | DataType::BOOL => original.extend(row.iter().map(|&b| b as f32)),
    }
}

/// SIMD elementwise accumulate: `acc[..] += src[..]`, scalar remainder tail.
#[cfg(feature = "simd")]
fn simd_add_into(acc: &mut [f32], src: &[f32]) {
    let len = acc.len();
    let mut i = 0;
    while i + 8 <= len {
        let a = f32x8::from(<[f32; 8]>::try_from(&acc[i..i + 8]).unwrap());
        let b = f32x8::from(<[f32; 8]>::try_from(&src[i..i + 8]).unwrap());
        acc[i..i + 8].copy_from_slice(&(a + b).to_array());
        i += 8;
    }
    for j in i..len {
        acc[j] += src[j];
    }
}

/// SIMD elementwise scalar divide: `v[..] /= n`, scalar remainder tail.
#[cfg(feature = "simd")]
fn simd_div_scalar(v: &mut [f32], n: f32) {
    let nv = f32x8::splat(n);
    let len = v.len();
    let mut i = 0;
    while i + 8 <= len {
        let a = f32x8::from(<[f32; 8]>::try_from(&v[i..i + 8]).unwrap());
        let result = (a / nv).to_array();
        v[i..i + 8].copy_from_slice(&result);
        i += 8;
    }
    for j in v.iter_mut().take(len).skip(i) {
        *j /= n;
    }
}

/// Elementwise accumulate: `acc[..] += src[..]`, SIMD when the feature is on.
fn add_into(acc: &mut [f32], src: &[f32]) {
    #[cfg(feature = "simd")]
    simd_add_into(acc, src);
    #[cfg(not(feature = "simd"))]
    for (a, s) in acc.iter_mut().zip(src) {
        *a += s;
    }
}

/// Elementwise scalar divide: `v[..] /= n`, SIMD when the feature is on.
fn div_scalar(v: &mut [f32], n: f32) {
    #[cfg(feature = "simd")]
    simd_div_scalar(v, n);
    #[cfg(not(feature = "simd"))]
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Mean-pools the rows of `table` (a flat `[vocab, dim]` buffer) selected by `ids`,
/// optionally L2-normalizing the result.
///
/// This function only reads from `table`, so it is safe to call from many threads.
/// An empty `ids` gives a zero vector.
fn pool_tokens(
    table: &[f32],
    dim: usize,
    vocab: usize,
    ids: &[u32],
    normalize: bool,
) -> Result<Vec<f32>, EmbedError> {
    let mut pooled = vec![0f32; dim];
    if ids.is_empty() {
        return Ok(pooled);
    }
    for &tok in ids {
        let t = tok as usize;
        if t >= vocab {
            return Err(EmbedError {
                cause: format!("Token id {tok} is out of range for a vocabulary of {vocab} tokens"),
            });
        }
        add_into(&mut pooled, &table[t * dim..(t + 1) * dim]);
    }
    div_scalar(&mut pooled, ids.len() as f32);

    if normalize {
        let norm = pooled.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
        for x in &mut pooled {
            *x /= norm;
        }
    }
    Ok(pooled)
}

/// Pools every sequence of the batch, one after the other.
#[cfg(not(feature = "rayon"))]
fn pool_batch(
    table: &[f32],
    dim: usize,
    vocab: usize,
    batch: &[Vec<u32>],
    normalize: bool,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    batch
        .iter()
        .map(|ids| pool_tokens(table, dim, vocab, ids, normalize))
        .collect()
}

/// Pools the sequences of the batch in parallel with `rayon`.
#[cfg(feature = "rayon")]
fn pool_batch(
    table: &[f32],
    dim: usize,
    vocab: usize,
    batch: &[Vec<u32>],
    normalize: bool,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    use rayon::prelude::*;

    batch
        .par_iter()
        .map(|ids| pool_tokens(table, dim, vocab, ids, normalize))
        .collect()
}

/// Whether `tokenizer.v2.json` in `dir` can be used instead of `tokenizer.json`.
///
/// It can if it is a file and it is not older than `tokenizer.json`, so that a changed
/// or re-downloaded `tokenizer.json` is not hidden by an outdated second file. If there is
/// no `tokenizer.json`, the second file is the only choice.
fn second_tokenizer_file_is_usable(dir: &Path) -> bool {
    let modified = |name: &str| {
        std::fs::metadata(dir.join(name))
            .ok()
            .filter(|m| m.is_file())
            .and_then(|m| m.modified().ok())
    };
    let Some(v2_modified) = modified("tokenizer.v2.json") else {
        return false;
    };
    match modified("tokenizer.json") {
        Some(original_modified) => v2_modified >= original_modified,
        None => true,
    }
}

/// A static embedding model loaded from disk.
///
/// `StaticEmbedding` lazily loads the underlying tensor and tokenizer on first
/// use, then caches them for subsequent calls. It supports mean-pooled
/// embeddings with optional L2 normalization.
///
/// The embedding matrix is kept as a flat `f32` table. Models smaller than the eager
/// loading threshold are decoded entirely at load time; larger ones are decoded
/// row by row, the first time each token is seen.
#[derive(Clone)]
pub struct StaticEmbedding {
    /// Filesystem path to the model directory.
    pub base_path: PathBuf,
    /// Metadata about the loaded tensor (shape, dtype, offsets).
    pub tensor_details: Option<TensorDetails>,
    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,
    /// Fallback truncation length for the tokenizer
    /// if none is found within the settings.
    /// Defaults to 512 if left to None.
    pub fallback_truncation_length: Option<usize>,
    /// Size of model vocabulary x embedding dimensions
    /// x 4 bytes at which eagerly decoding the model
    /// into a tokens cache is no longer ideal.
    /// Defaults to 256 MB.
    pub eager_loading_threshold: Option<u64>,
    /// Memory-mapped tensor bytes. Dropped (and unmapped) once the model is eagerly decoded.
    tensor: Option<TensorBytes>,
    tokenizer: Option<Arc<PipelineTokenizer>>,
    truncation: Option<TruncationParams>,
    /// ID of the unknown token, which is removed from the tokenized texts.
    unknown_token: Option<u32>,
    /// Flat `[vocab, dim]` table of decoded token vectors.
    tokens: Vec<f32>,
    /// One flag per token: whether its row in `tokens` is decoded (lazy loading only).
    decoded: Vec<bool>,
    /// Whether `tokens` was fully decoded at load time.
    eagerly_loaded: bool,
    /// Whether to load tokenizer from `tokenizer.v2.json`
    use_v2: bool,
}

impl StaticEmbedding {
    /// Creates a `StaticEmbedding` from a local directory.
    ///
    /// The directory must contain `model.safetensors` and `tokenizer.json`. When a
    /// `tokenizer.json` in the older v1 format is loaded, it is converted and the result is
    /// saved next to it as `tokenizer.v2.json`. Later models use that file for as long as it is
    /// not older than `tokenizer.json`. The original file is never changed. If a directory only
    /// has `tokenizer.v2.json`, that file is used.
    ///
    /// # Arguments
    /// * `path` - Path to the model directory.
    /// * `normalize` - If `Some(true)`, output embeddings will be L2-normalized.
    /// * `fallback_truncation_length` - If the tokenizer does not have a default truncation length set
    ///   use this as truncation length. If left to None, defaults to 512.
    /// * `eager_loading_threshold` - Size in bytes (vocabulary x dimensions x 4) of the decoded
    ///   model below which it is fully decoded at load time. Larger models are decoded on demand.
    ///   If left to None, defaults to 256 MB.
    pub fn from_dir<T: AsRef<Path>>(
        path: T,
        normalize: Option<bool>,
        fallback_truncation_length: Option<usize>,
        eager_loading_threshold: Option<u64>,
    ) -> Result<Self, InvalidModelOrPathError> {
        let p: &Path = path.as_ref();
        if !p.join("model.safetensors").exists() {
            return Err(InvalidModelOrPathError {
                model_or_path: p.to_string_lossy().to_string(),
                details: "Could not find `models.safetensors` in the specified directory"
                    .to_string(),
            });
        }

        let use_v2 = second_tokenizer_file_is_usable(p);

        if !p.join("tokenizer.json").exists() && !use_v2 {
            return Err(InvalidModelOrPathError {
                model_or_path: p.to_string_lossy().to_string(),
                details: "Could not find `tokenizer.json` or `tokenizer.v2.json` in the specified directory".to_string(),
            });
        }

        Ok(Self {
            base_path: p.to_owned(),
            normalize: normalize.unwrap_or_default(),
            tokenizer: None,
            tensor: None,
            tensor_details: None,
            truncation: None,
            unknown_token: None,
            tokens: Vec::new(),
            decoded: Vec::new(),
            fallback_truncation_length,
            eager_loading_threshold,
            eagerly_loaded: false,
            use_v2,
        })
    }

    /// Downloads a model from the Hugging Face Hub and returns a `StaticEmbedding`.
    ///
    /// # Arguments
    /// * `model_id` - Hugging Face model identifier in `owner/repo_name` format.
    /// * `normalize` - If `Some(true)`, output embeddings will be L2-normalized.
    /// * `force_download` - If `true`, re-downloads files even if they already exist locally.
    /// * `fallback_truncation_length` - If the tokenizer does not have a default truncation length set
    ///   use this as truncation length. If left to None, defaults to 512.
    /// * `eager_loading_threshold` - Size in bytes (vocabulary x dimensions x 4) of the decoded
    ///   model below which it is fully decoded at load time. Larger models are decoded on demand.
    ///   If left to None, defaults to 256 MB.
    #[cfg(feature = "hf-hub")]
    pub async fn from_hf_hub(
        model_id: &str,
        normalize: Option<bool>,
        force_download: bool,
        fallback_truncation_length: Option<usize>,
        eager_loading_threshold: Option<u64>,
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
                // a new `tokenizer.json` makes the converted second file outdated
                if *f == "tokenizer.json" {
                    let _ = std::fs::remove_file(base_path.join("tokenizer.v2.json"));
                }
            }
            let use_v2 = second_tokenizer_file_is_usable(&base_path);
            return Ok(Self {
                base_path,
                normalize: normalize.unwrap_or_default(),
                tokenizer: None,
                tensor: None,
                tensor_details: None,
                truncation: None,
                unknown_token: None,
                tokens: Vec::new(),
                decoded: Vec::new(),
                fallback_truncation_length,
                eager_loading_threshold,
                eagerly_loaded: false,
                use_v2,
            });
        }

        Err(InvalidModelOrPathError {
            model_or_path: model_id.to_string(),
            details: "Model ID should be reported as owner/repo_name".to_string(),
        })
    }

    /// Memory-maps the `model.safetensors` file and reads its header.
    ///
    /// The tensor bytes are not copied: the operating system reads them from disk when they
    /// are first used (for example by `eager_load_tokens`).
    ///
    /// This is called lazily by the embedding methods and by `init`. It is public so that
    /// each loading step can be run (and measured) on its own.
    pub fn load_tensor(&mut self) -> Result<(), LoadError> {
        let (details, tensor) = load_safetensors_file(self.base_path.join("model.safetensors"))?;
        self.tensor = Some(tensor);
        self.tensor_details = Some(details);
        Ok(())
    }

    /// Loads the tokenizer (`tokenizer.json`, or `tokenizer.v2.json` if it is usable), its
    /// truncation settings and the ID of its unknown token.
    ///
    /// This is called lazily by the embedding methods and by `init`. It is public so that
    /// each loading step can be run (and measured) on its own.
    pub fn load_tokenizer(&mut self) -> Result<(), TokenizationError> {
        let path = if self.use_v2 {
            self.base_path.join("tokenizer.v2.json")
        } else {
            self.base_path.join("tokenizer.json")
        };

        let (tokenizer, truncation, unknown_token) =
            load_tokenizer(path, self.fallback_truncation_length)?;
        self.tokenizer = Some(Arc::new(tokenizer));
        self.truncation = Some(truncation);
        self.unknown_token = unknown_token;
        Ok(())
    }

    /// Builds the `tokens` table from the loaded tensor.
    ///
    /// Below the eager loading threshold, every row is decoded to `f32` right away and
    /// the raw bytes are dropped. Above it, the table is allocated with zeros and rows
    /// are decoded on demand (see `decode_missing`), tracked by `decoded`.
    ///
    /// The tensor must be loaded first (see `load_tensor`), otherwise this does nothing.
    /// It is public so that each loading step can be run (and measured) on its own.
    pub fn eager_load_tokens(&mut self) -> Result<(), EmbedError> {
        let eager_loading_threshold = self
            .eager_loading_threshold
            .unwrap_or(DEFAULT_EAGER_LOADING_THRESHOLD);
        let (Some(tensor), Some(details)) = (self.tensor.as_ref(), self.tensor_details) else {
            return Ok(());
        };
        let (vocab, dim) = (details.shape[0] as usize, details.shape[1] as usize);
        let row_bytes = dim * details.dtype.to_size();
        let expected_bytes = vocab * row_bytes;
        if tensor.len() != expected_bytes {
            return Err(EmbedError {
                cause: format!(
                    "Tensor size mismatch: shape {:?} with dtype size {} expects {} bytes, found {}",
                    details.shape,
                    details.dtype.to_size(),
                    expected_bytes,
                    tensor.len()
                ),
            });
        }
        if details.shape[0] * details.shape[1] * 4 < eager_loading_threshold {
            let mut new_tokens: Vec<f32> = Vec::with_capacity(vocab * dim);
            for chunk in tensor.chunks_exact(row_bytes) {
                decode_row_into(chunk, details.dtype, &mut new_tokens);
            }
            self.tokens = new_tokens;
            self.eagerly_loaded = true;
            self.tensor = None;
        } else {
            // untouched zero pages do not use physical memory
            self.tokens = vec![0f32; vocab * dim];
            self.decoded = vec![false; vocab];
        }
        Ok(())
    }

    /// Decodes, once, the rows of every token in `batch` that is not in `tokens` yet.
    ///
    /// Only used when the model was not eagerly loaded. Token IDs that are out of range
    /// are skipped here and reported by `pool_tokens`.
    fn decode_missing(&mut self, batch: &[Vec<u32>]) -> Result<(), EmbedError> {
        let (Some(tensor), Some(details)) = (self.tensor.as_deref(), self.tensor_details) else {
            return Err(EmbedError {
                cause: "Tensor should be non-null at this point".to_string(),
            });
        };
        let (vocab, dim) = (details.shape[0] as usize, details.shape[1] as usize);
        let row_bytes = dim * details.dtype.to_size();
        for &tok in batch.iter().flatten() {
            let t = tok as usize;
            if t >= vocab || self.decoded[t] {
                continue;
            }
            let row = decode_row(&tensor[t * row_bytes..(t + 1) * row_bytes], details.dtype);
            self.tokens[t * dim..(t + 1) * dim].copy_from_slice(&row);
            self.decoded[t] = true;
        }
        Ok(())
    }

    /// Pre-load the model and the tokenizer, and decode the model if it is
    /// smaller than the eager loading threshold.
    /// They will be lazy-loaded on first request otherwise.
    ///
    /// The tensor and the tokenizer are loaded at the same time, on two threads.
    pub fn init(&mut self) -> Result<(), EmbedError> {
        let tok_path = if self.use_v2 {
            self.base_path.join("tokenizer.v2.json")
        } else {
            self.base_path.join("tokenizer.json")
        };
        let (tkr, tnsr) = std::thread::scope(|s| {
            let tensor_handle =
                s.spawn(|| load_safetensors_file(self.base_path.join("model.safetensors")));
            let tok_res = load_tokenizer(tok_path, self.fallback_truncation_length);

            (tok_res, tensor_handle.join())
        });
        let (tk, trunc, unk) = tkr?;
        let res = tnsr.map_err(|_| EmbedError {
            cause: "The tensor thread panicked".to_string(),
        })?;
        let (dets, bts) = res?;
        self.tensor_details = Some(dets);
        self.tensor = Some(bts);
        self.tokenizer = Some(Arc::new(tk));
        self.truncation = Some(trunc);
        self.unknown_token = unk;

        self.eager_load_tokens()
    }

    /// Generates an embedding for a pre-tokenized sequence of token IDs.
    ///
    /// The tensor is loaded lazily on first call if it has not already
    /// been initialized with a call to `init`. Embeddings are mean-pooled
    /// and optionally normalized. An empty sequence gives a zero vector, and
    /// a token ID outside the vocabulary gives an error. The IDs are used as given:
    /// the unknown token is only removed from texts that are tokenized by this crate.
    pub fn embed_tokens(&mut self, tokens: Vec<u32>) -> Result<Vec<f32>, EmbedError> {
        self.embed_token_batch(&[tokens])?
            .pop()
            .ok_or_else(|| EmbedError {
                cause: "Embedding batch should not be empty at this point".to_string(),
            })
    }

    /// Generates one embedding for each pre-tokenized sequence in `batch`.
    ///
    /// With the `rayon` feature, the sequences are pooled in parallel.
    /// Without it, they are pooled one after the other.
    pub fn embed_token_batch(&mut self, batch: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // an empty table means that neither the eager nor the lazy table was built yet
        if self.tokens.is_empty() {
            if self.tensor.is_none() {
                self.load_tensor()?;
            }
            self.eager_load_tokens()?;
        }
        let Some(details) = self.tensor_details else {
            return Err(EmbedError {
                cause: "Tensor details should be non-null at this point".to_string(),
            });
        };
        if !self.eagerly_loaded {
            self.decode_missing(batch)?;
        }
        pool_batch(
            &self.tokens,
            details.shape[1] as usize,
            details.shape[0] as usize,
            batch,
            self.normalize,
        )
    }

    /// Tokenizes `texts` in batches of `batch_size` and returns one embedding
    /// for each text, in the same order as the input.
    ///
    /// The tokenizer and tensor are loaded lazily on first call. Inputs are
    /// truncated but not padded: the mean-pooling reads real tokens only. The unknown
    /// token is removed from each text before pooling.
    pub fn embed_texts(
        &mut self,
        texts: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if self.tokenizer.is_none() {
            self.load_tokenizer().map_err(|e| EmbedError {
                cause: format!("Could not load tokenizer: {e}"),
            })?;
        }
        // clone the handles so that `self` can be borrowed mutably below
        let (Some(tk), Some(truncation)) = (self.tokenizer.clone(), self.truncation.clone()) else {
            return Err(EmbedError {
                cause: "Tokenizer should be non-null at this point".to_string(),
            });
        };
        let unknown_token = self.unknown_token;
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(batch_size) {
            let encodings = encode_batch(&tk, &truncation, batch).map_err(|e| EmbedError {
                cause: format!("Error while tokenizing the inputs: {e}"),
            })?;
            let ids: Vec<Vec<u32>> = encodings
                .iter()
                .map(|enc| {
                    enc.ids()
                        .iter()
                        .map(|t| t.id())
                        .filter(|id| Some(*id) != unknown_token)
                        .collect()
                })
                .collect();
            embeddings.extend(self.embed_token_batch(&ids)?);
        }
        Ok(embeddings)
    }

    /// Tokenizes `text` and returns its embedding.
    ///
    /// This is `embed_texts` for a single text.
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut result = self.embed_texts(&[text], Some(1))?;
        Ok(result.pop().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_directory() {
        let _ = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
    }

    #[test]
    fn test_load_tensor() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
        model.load_tensor().expect("Should be able to load tensor");
        assert!(model.tensor.is_some());
    }

    #[test]
    fn test_load_tokenizer() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
        model
            .load_tokenizer()
            .expect("Should be able to load tokenizer");
        assert!(model.tokenizer.is_some());
        assert!(model.truncation.is_some());
    }

    #[test]
    fn test_decode_row_succeeds_for_all_dtypes() {
        let bf16 = [half::bf16::from_f32(1.0), half::bf16::from_f32(-2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&bf16, DataType::BF16), vec![1.0, -2.0]);

        assert_eq!(decode_row(&[1u8, 0u8], DataType::BOOL), vec![1.0, 0.0]);

        let f32s = [3.5f32, -1.25f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&f32s, DataType::F32), vec![3.5, -1.25]);

        let f64s = [2.25f64, 8.0f64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&f64s, DataType::F64), vec![2.25, 8.0]);

        let f16s = [half::f16::from_f32(4.0), half::f16::from_f32(0.5)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&f16s, DataType::F16), vec![4.0, 0.5]);

        let i16s = [-42i16, 7i16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&i16s, DataType::I16), vec![-42.0, 7.0]);

        let i32s = [12345i32, -1i32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&i32s, DataType::I32), vec![12345.0, -1.0]);

        let i64s = [-987654321i64, 5i64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(decode_row(&i64s, DataType::I64), vec![-987654321.0, 5.0]);

        assert_eq!(
            decode_row(&[(-7i8) as u8, 3u8], DataType::I8),
            vec![-7.0, 3.0]
        );
        assert_eq!(decode_row(&[200u8, 1u8], DataType::U8), vec![200.0, 1.0]);
    }

    #[test]
    fn test_eager_loading_builds_full_table() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
        let _ = model
            .embed_text("hello there!")
            .expect("Should be able to embed text");
        let details = model.tensor_details.expect("Tensor details should be set");
        assert!(model.eagerly_loaded);
        assert!(model.tensor.is_none());
        assert_eq!(
            model.tokens.len() as u64,
            details.shape[0] * details.shape[1]
        );
    }

    #[test]
    fn test_lazy_loading_decodes_only_used_tokens() {
        // a threshold of 0 bytes forces lazy loading
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, Some(0))
            .expect("Should be able to load model from testfiles");
        let _ = model
            .embed_text("hello there!")
            .expect("Should be able to embed text");
        assert!(!model.eagerly_loaded);
        assert!(model.tensor.is_some());
        let n_decoded = model.decoded.iter().filter(|d| **d).count();
        assert!(n_decoded > 0);
        assert!(n_decoded < model.decoded.len());
        // token for 'hello'
        assert!(model.decoded[6598]);
    }

    #[test]
    fn test_lazy_and_eager_loading_give_same_embeddings() {
        let texts = ["hello there!", "this is a longer sentence to embed", "hi"];
        let mut eager = StaticEmbedding::from_dir("testfiles/", Some(true), None, None)
            .expect("Should be able to load model from testfiles");
        let mut lazy = StaticEmbedding::from_dir("testfiles/", Some(true), None, Some(0))
            .expect("Should be able to load model from testfiles");
        assert_eq!(
            eager.embed_texts(&texts, None).expect("Should embed"),
            lazy.embed_texts(&texts, None).expect("Should embed")
        );
    }

    #[test]
    fn test_embed_texts_matches_embed_text_in_order() {
        let texts = ["hello there!", "this is a longer sentence to embed", "hi"];
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
        // a batch size smaller than the input forces several batches
        let batched = model
            .embed_texts(&texts, Some(2))
            .expect("Should embed the batch");
        assert_eq!(batched.len(), texts.len());
        for (text, embedding) in texts.iter().zip(&batched) {
            assert_eq!(
                &model.embed_text(text).expect("Should embed text"),
                embedding
            );
        }
    }

    /// A model directory with the test tokenizer, whose `model.safetensors` is only a
    /// placeholder: it is enough to build a `StaticEmbedding` and to load its tokenizer.
    fn tokenizer_only_dir(name: &str) -> std::path::PathBuf {
        let dir = crate::tokenize::temp_dir(name);
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        std::fs::copy("testfiles/tokenizer.json", dir.join("tokenizer.json")).unwrap();
        dir
    }

    fn token_ids(model: &StaticEmbedding, text: &str) -> Vec<u32> {
        let (tk, truncation) = (
            model
                .tokenizer
                .as_ref()
                .expect("Tokenizer should be loaded"),
            model.truncation.as_ref().expect("Truncation should be set"),
        );
        encode_batch(tk, truncation, &[text]).expect("Should tokenize")[0]
            .ids()
            .iter()
            .map(|t| t.id())
            .collect()
    }

    #[test]
    fn test_second_tokenizer_file_is_used_by_later_models() {
        let dir = tokenizer_only_dir("second_file_used");

        let mut first = StaticEmbedding::from_dir(&dir, None, None, None)
            .expect("Should be able to load model from directory");
        assert!(!first.use_v2);
        first.load_tokenizer().expect("Should load the tokenizer");
        assert!(dir.join("tokenizer.v2.json").exists());

        let mut second = StaticEmbedding::from_dir(&dir, None, None, None)
            .expect("Should be able to load model from directory");
        assert!(second.use_v2);
        second.load_tokenizer().expect("Should load the tokenizer");

        let text = "hello \u{1F980} world, this is a test sentence";
        assert_eq!(first.unknown_token, second.unknown_token);
        assert_eq!(token_ids(&first, text), token_ids(&second, text));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_dir_works_with_only_the_second_tokenizer_file() {
        let dir = tokenizer_only_dir("second_file_only");
        StaticEmbedding::from_dir(&dir, None, None, None)
            .expect("Should be able to load model from directory")
            .load_tokenizer()
            .expect("Should load the tokenizer");
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();

        let mut model = StaticEmbedding::from_dir(&dir, None, None, None)
            .expect("The second tokenizer file should be enough");
        assert!(model.use_v2);
        model.load_tokenizer().expect("Should load the tokenizer");
        assert!(model.tokenizer.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_outdated_second_tokenizer_file_is_not_used() {
        let dir = tokenizer_only_dir("outdated_second_file");
        StaticEmbedding::from_dir(&dir, None, None, None)
            .expect("Should be able to load model from directory")
            .load_tokenizer()
            .expect("Should load the tokenizer");
        assert!(second_tokenizer_file_is_usable(&dir));

        // the second file is older than `tokenizer.json`, as if the latter was replaced
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(100);
        std::fs::File::options()
            .write(true)
            .open(dir.join("tokenizer.v2.json"))
            .unwrap()
            .set_modified(old)
            .unwrap();
        assert!(!second_tokenizer_file_is_usable(&dir));
        let mut model = StaticEmbedding::from_dir(&dir, None, None, None)
            .expect("Should be able to load model from directory");
        assert!(!model.use_v2);

        // loading from `tokenizer.json` writes a new second file, which is usable again
        model.load_tokenizer().expect("Should load the tokenizer");
        assert!(second_tokenizer_file_is_usable(&dir));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_second_tokenizer_file_is_usable_cases() {
        let dir = crate::tokenize::temp_dir("usable_cases");
        // no files at all
        assert!(!second_tokenizer_file_is_usable(&dir));
        // only the second file
        std::fs::write(dir.join("tokenizer.v2.json"), b"{}").unwrap();
        assert!(second_tokenizer_file_is_usable(&dir));
        // a directory with that name is not a file
        std::fs::remove_file(dir.join("tokenizer.v2.json")).unwrap();
        std::fs::create_dir(dir.join("tokenizer.v2.json")).unwrap();
        assert!(!second_tokenizer_file_is_usable(&dir));
        // only the original
        std::fs::remove_dir(dir.join("tokenizer.v2.json")).unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert!(!second_tokenizer_file_is_usable(&dir));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_dir_fails_without_any_tokenizer_file() {
        let dir = crate::tokenize::temp_dir("no_tokenizer");
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let err = StaticEmbedding::from_dir(&dir, None, None, None).err();
        std::fs::remove_dir_all(&dir).ok();
        assert!(err.is_some());
    }

    #[test]
    fn test_unknown_tokens_are_removed() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
        model
            .load_tokenizer()
            .expect("Should be able to load tokenizer");
        let unknown_token = model.unknown_token.expect("The model has an unknown token");

        // a crab emoji is not in the vocabulary
        let text = "hello \u{1F980} world";
        let all_ids = token_ids(&model, text);
        assert!(
            all_ids.contains(&unknown_token),
            "the test text should contain an unknown token: {all_ids:?}"
        );

        let known_ids: Vec<u32> = all_ids
            .into_iter()
            .filter(|id| *id != unknown_token)
            .collect();
        let expected = model.embed_tokens(known_ids).expect("Should embed tokens");
        assert_eq!(model.embed_text(text).expect("Should embed text"), expected);
    }

    #[test]
    fn test_embed_texts_empty_input() {
        let mut model = StaticEmbedding::from_dir("testfiles/", None, None, None)
            .expect("Should be able to load model from testfiles");
        assert!(model.embed_texts(&[], None).unwrap().is_empty());
    }

    #[test]
    fn test_pool_tokens_mean() {
        // vocab of 3 tokens, dim of 2
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = pool_tokens(&table, 2, 3, &[0, 2], false).unwrap();
        assert_eq!(pooled, vec![3.0, 4.0]);
    }

    #[test]
    fn test_pool_tokens_normalized() {
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = pool_tokens(&table, 2, 3, &[0, 2], true).unwrap();
        assert!((pooled[0] - 0.6).abs() < 1e-6);
        assert!((pooled[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_pool_tokens_empty_ids_gives_zeros() {
        let table = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            pool_tokens(&table, 2, 2, &[], true).unwrap(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn test_pool_tokens_out_of_range_id_fails() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let err = pool_tokens(&table, 2, 2, &[0, 2], false).unwrap_err();
        assert!(
            err.cause.contains("out of range"),
            "unexpected error message: {}",
            err.cause
        );
    }
}
