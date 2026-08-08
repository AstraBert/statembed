#[cfg(feature = "tokenizers")]
use crate::{errors::TokenizationError, tokenize::load_tokenizer};
#[cfg(feature = "hf-hub")]
use hf_hub::{HFClient, RepoTypeModel};
use std::path::{Path, PathBuf};
#[cfg(feature = "hf-hub")]
use std::sync::OnceLock;
#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

use crate::{
    errors::{InvalidModelOrPathError, LoadError},
    load::{TensorDetails, load_safetensors_file},
};

pub mod errors;
pub mod load;
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
}
