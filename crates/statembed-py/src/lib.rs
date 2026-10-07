use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

/// A Python module implemented in Rust.
#[pymodule]
mod statembed_py {
    use pyo3::{
        exceptions::{PyRuntimeError, PyValueError},
        prelude::*,
    };
    use pyo3_stub_gen::derive::*;
    use statembed::StaticEmbedding as CoreEmbedding;

    #[gen_stub_pyclass]
    #[pyclass(from_py_object)]
    #[derive(Clone)]
    struct StaticEmbedding {
        core: CoreEmbedding,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StaticEmbedding {
        #[new]
        #[pyo3(signature = (model_dir, normalize = true, fallback_truncation_length = None, eager_loading_threshold = None))]
        /// Creates a `StaticEmbedding` from a local directory.
        ///
        /// The directory must contain `model.safetensors` and `tokenizer.json`. When a
        /// `tokenizer.json` in the older v1 format is loaded, it is converted and the result is
        /// saved next to it as `tokenizer.v2.json`, which later models use. The original file
        /// is never changed.
        ///
        /// # Arguments
        /// * `model_dir` - Path to the model directory.
        /// * `normalize` - If `True`, output embeddings will be L2-normalized.
        /// * `fallback_truncation_length` - If the tokenizer does not have a default truncation
        ///   length set, use this as truncation length. If left to `None`, defaults to 512.
        /// * `eager_loading_threshold` - Size in bytes (vocabulary x dimensions x 4) of the decoded
        ///   model below which it is fully decoded at load time. Larger models are decoded on
        ///   demand. If left to `None`, defaults to 256 MB.
        fn new(
            model_dir: String,
            normalize: bool,
            fallback_truncation_length: Option<usize>,
            eager_loading_threshold: Option<u64>,
        ) -> PyResult<Self> {
            Ok(Self {
                core: CoreEmbedding::from_dir(
                    &model_dir,
                    Some(normalize),
                    fallback_truncation_length,
                    eager_loading_threshold,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            })
        }

        /// Tokenizes `text` and returns its embedding.
        ///
        /// The tokenizer and the model are loaded lazily on first call. The unknown token is
        /// removed from the text before pooling.
        fn embed_text(&mut self, text: &str) -> PyResult<Vec<f32>> {
            self.core
                .embed_text(text)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        #[pyo3(signature = (texts, batch_size = None))]
        /// Tokenizes `texts` in batches of `batch_size` (defaults to 16) and returns one
        /// embedding for each text, in the same order as the input.
        fn embed_texts(
            &mut self,
            texts: Vec<String>,
            batch_size: Option<usize>,
        ) -> PyResult<Vec<Vec<f32>>> {
            self.core
                .embed_texts(
                    texts
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<&str>>()
                        .as_slice(),
                    batch_size,
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        /// Generates one embedding for each sequence of pre-tokenized token IDs.
        fn embed_tokens_batch(&mut self, tokens: Vec<Vec<u32>>) -> PyResult<Vec<Vec<f32>>> {
            self.core
                .embed_token_batch(&tokens)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        /// Generates an embedding for a pre-tokenized sequence of token IDs (use whatever tokenization
        /// libary you prefer to generate tokens).
        ///
        /// The tensor is loaded lazily on first call. Embeddings are mean-pooled
        /// and optionally normalized. The IDs are used as given: the unknown token is only
        /// removed from texts that are tokenized by `embed_text` and `embed_texts`.
        fn embed_tokens(&mut self, tokens: Vec<u32>) -> PyResult<Vec<f32>> {
            self.core
                .embed_tokens(tokens)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
    }
}

define_stub_info_gatherer!(stub_info);
