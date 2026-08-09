#[cfg(feature = "tokenizers")]
use crate::tokenize::tokenize;
#[cfg(feature = "tokenizers")]
use crate::{errors::TokenizationError, tokenize::load_tokenizer};
#[cfg(feature = "hf-hub")]
use hf_hub::{HFClient, RepoTypeModel};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
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

#[cfg(feature = "hf-hub")]
pub fn hf_cache_dir() -> &'static PathBuf {
    HF_CACHE_DIR.get_or_init(|| {
        dirs::home_dir()
            .expect("No home dir could be found for the current environment")
            .join(".statembed")
    })
}

pub struct StaticEmbedding {
    pub base_path: PathBuf,
    pub tensor_details: Option<TensorDetails>,
    tensor: Option<Vec<u8>>,
    #[cfg(feature = "tokenizers")]
    tokenizer: Option<Tokenizer>,
}

impl StaticEmbedding {
    pub fn from_dir<T: AsRef<Path>>(path: T) -> Result<Self, InvalidModelOrPathError> {
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
            tokenizer: None,
            tensor: None,
            tensor_details: None,
        })
    }

    #[cfg(feature = "hf-hub")]
    pub async fn from_hf_hub(model_id: &str) -> Result<Self, InvalidModelOrPathError> {
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
                tokenizer: None,
                tensor: None,
                tensor_details: None,
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
        let tokenizer = load_tokenizer(self.base_path.join("tokenizer.json"))?;
        self.tokenizer = Some(tokenizer);
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
            #[cfg(not(feature = "rayon"))]
            {
                let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());

                for tok in tokens {
                    let start = (tok as u64) * dim * (dtype_size as u64);
                    let finish = start + (dim * (dtype_size as u64));
                    let tok_repr = &t[(start as usize)..(finish as usize)];
                    let mut tok_vec: Vec<f32> = Vec::with_capacity(dim as usize);
                    for tr in tok_repr.chunks(dtype_size) {
                        let fl = match dtype {
                            DataType::BF16 => {
                                half::bf16::from_le_bytes(tr.try_into().map_err(|e| {
                                    EmbedError {
                                        cause: format!("Bytes to BF16 conversion error: {}", e),
                                    }
                                })?)
                                .to_f32()
                            }
                            DataType::BOOL => {
                                let val = unsafe { *tr.get_unchecked(0) };
                                val as f32
                            }
                            DataType::F32 => {
                                f32::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to F32 conversion error: {}", e),
                                })?)
                            }
                            DataType::F64 => {
                                f64::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to F64 conversion error: {}", e),
                                })?) as f32
                            }
                            DataType::F16 => {
                                half::f16::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to F16 conversion error: {}", e),
                                })?)
                                .to_f32()
                            }
                            DataType::I16 => {
                                i16::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to I16 conversion error: {}", e),
                                })?) as f32
                            }
                            DataType::I32 => {
                                i32::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to I32 conversion error: {}", e),
                                })?) as f32
                            }
                            DataType::I64 => {
                                i64::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to I64 conversion error: {}", e),
                                })?) as f32
                            }
                            DataType::I8 => {
                                i8::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                    cause: format!("Bytes to I8 conversion error: {}", e),
                                })?) as f32
                            }
                            DataType::U8 => {
                                let val = unsafe { *tr.get_unchecked(0) };
                                val as f32
                            }
                        };
                        tok_vec.push(fl);
                    }
                    vectors.push(tok_vec);
                }

                let mut mean_pooled: Vec<f32> = Vec::with_capacity(dim as usize);
                for j in 0..(dim as usize) {
                    let mut summed: f32 = 0_f32;
                    for v in vectors.as_slice() {
                        let f = unsafe { *v.get_unchecked(j) };
                        summed += f;
                    }
                    mean_pooled.push(summed / vectors.len() as f32);
                }
                return Ok(mean_pooled);
            }
            #[cfg(feature = "rayon")]
            {
                let vectors = tokens
                    .par_iter()
                    .map(|tok| {
                        let start = (*tok as u64) * dim * (dtype_size as u64);
                        let finish = start + (dim * (dtype_size as u64));
                        let tok_repr = &t[(start as usize)..(finish as usize)];
                        let mut tok_vec: Vec<f32> = Vec::with_capacity(dim as usize);
                        for tr in tok_repr.chunks(dtype_size) {
                            let fl = match dtype {
                                DataType::BF16 => {
                                    half::bf16::from_le_bytes(tr.try_into().map_err(|e| {
                                        EmbedError {
                                            cause: format!("Bytes to BF16 conversion error: {}", e),
                                        }
                                    })?)
                                    .to_f32()
                                }
                                DataType::BOOL => {
                                    let val = unsafe { *tr.get_unchecked(0) };
                                    val as f32
                                }
                                DataType::F32 => {
                                    f32::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                        cause: format!("Bytes to F32 conversion error: {}", e),
                                    })?)
                                }
                                DataType::F64 => {
                                    f64::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                        cause: format!("Bytes to F64 conversion error: {}", e),
                                    })?) as f32
                                }
                                DataType::F16 => {
                                    half::f16::from_le_bytes(tr.try_into().map_err(|e| {
                                        EmbedError {
                                            cause: format!("Bytes to F16 conversion error: {}", e),
                                        }
                                    })?)
                                    .to_f32()
                                }
                                DataType::I16 => {
                                    i16::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                        cause: format!("Bytes to I16 conversion error: {}", e),
                                    })?) as f32
                                }
                                DataType::I32 => {
                                    i32::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                        cause: format!("Bytes to I32 conversion error: {}", e),
                                    })?) as f32
                                }
                                DataType::I64 => {
                                    i64::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                        cause: format!("Bytes to F16 conversion error: {}", e),
                                    })?) as f32
                                }
                                DataType::I8 => {
                                    i8::from_le_bytes(tr.try_into().map_err(|e| EmbedError {
                                        cause: format!("Bytes to I8 conversion error: {}", e),
                                    })?) as f32
                                }
                                DataType::U8 => {
                                    let val = unsafe { *tr.get_unchecked(0) };
                                    val as f32
                                }
                            };
                            tok_vec.push(fl);
                        }
                        Ok(tok_vec)
                    })
                    .collect::<Result<Vec<Vec<f32>>, EmbedError>>()?;
                let mean_pooled: Vec<f32> = (0..(dim as usize))
                    .into_par_iter()
                    .map(|j| {
                        let mut summed: f32 = 0_f32;
                        for v in vectors.as_slice() {
                            let f = unsafe { *v.get_unchecked(j) };
                            summed += f;
                        }
                        summed / vectors.len() as f32
                    })
                    .collect();
                return Ok(mean_pooled);
            }
        }
        Err(EmbedError {
            cause: "Tensor should be non-null at this point".to_string(),
        })
    }

    #[cfg(feature = "tokenizers")]
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if self.tokenizer.is_none() {
            self.load_tokenizer()?;
        }
        if let Some(tk) = self.tokenizer.as_ref() {
            let tokens = tokenize(tk, text)?;
            let embedding = self.embed_tokens(tokens)?;
            return Ok(embedding);
        }
        Err(EmbedError {
            cause: "Tokenizer should be non-null at this point".to_string(),
        })
    }
}
