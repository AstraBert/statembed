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

#[cfg(feature = "hf-hub")]
pub const DOWNLOAD_FILES: &[&str] = &["model.safetensors", "tokenizer.json"];
#[cfg(feature = "hf-hub")]
pub static HF_CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

pub const PARALLELIZATION_THRESHOLD: u64 = 100_000;

#[cfg(feature = "hf-hub")]
pub fn hf_cache_dir() -> &'static PathBuf {
    HF_CACHE_DIR.get_or_init(|| {
        dirs::home_dir()
            .expect("No home dir could be found for the current environment")
            .join(".statembed")
    })
}

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

pub struct StaticEmbedding {
    pub base_path: PathBuf,
    pub tensor_details: Option<TensorDetails>,
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

        if !p.join("tokenizer.json").exists() {
            return Err(InvalidModelOrPathError {
                model_or_path: p.to_string_lossy().to_string(),
                details: "Could not find `tokenizer.json` in the specified directory".to_string(),
            });
        }

        Ok(Self {
            base_path: p.to_owned(),
            normalize: normalize.unwrap_or_default(),
            tokenizer: None,
            tensor: None,
            tensor_details: None,
            median_token_length: None,
            unknown_token: None,
            token_map: HashMap::new(),
        })
    }

    #[cfg(feature = "hf-hub")]
    pub async fn from_hf_hub(
        model_id: &str,
        normalize: Option<bool>,
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
                tokenizer: None,
                tensor: None,
                tensor_details: None,
                median_token_length: None,
                unknown_token: None,
                token_map: HashMap::new(),
            });
        }

        Err(InvalidModelOrPathError {
            model_or_path: model_id.to_string(),
            details: "Model ID should be reported as owner/repo_name".to_string(),
        })
    }

    fn load_tensor(&mut self) -> Result<(), LoadError> {
        let (details, tensor) = load_safetensors_file(self.base_path.join("model.safetensors"))?;
        self.tensor = Some(tensor);
        self.tensor_details = Some(details);
        Ok(())
    }

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

    #[cfg(feature = "tokenizers")]
    fn truncate_input<'a>(&'a self, text: &'a str, max_token_length: usize) -> &'a str {
        text.char_indices()
            .nth(max_token_length.saturating_mul(self.median_token_length.unwrap()))
            .map_or(text, |(byte_idx, _)| &text[..byte_idx])
    }

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
