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
        #[pyo3(signature = (model_dir, normalize = true))]
        /// Creates a `StaticEmbedding` from a local directory.
        ///
        /// The directory must contain at least `model.safetensors`. When the
        /// `tokenizers` feature is enabled, `tokenizer.json` is also required.
        ///
        /// # Arguments
        /// * `model_dir` - Path to the model directory.
        /// * `normalize` - If `True`, output embeddings will be L2-normalized.
        fn new(model_dir: String, normalize: bool) -> PyResult<Self> {
            Ok(Self {
                core: CoreEmbedding::from_dir(&model_dir, Some(normalize))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            })
        }

        /// Generates an embedding for a pre-tokenized sequence of token IDs (use whatever tokenization
        /// libary you prefer to generate tokens).
        ///
        /// The tensor is loaded lazily on first call. Embeddings are mean-pooled
        /// and optionally normalized.
        fn embed_tokens(&mut self, tokens: Vec<u32>) -> PyResult<Vec<f32>> {
            self.core
                .embed_tokens(tokens)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
    }
}

define_stub_info_gatherer!(stub_info);
