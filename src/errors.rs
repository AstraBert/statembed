//! Error types for the `statemebed` library.
//!
//! This module defines the various error types that can occur when loading
//! models, tokenizing text, or generating embeddings.

use std::{fmt::Display, io};

/// An error that occurred while loading a model or tensor file.
#[derive(Debug)]
pub struct LoadError {
    /// Human-readable description of what went wrong.
    pub cause: String,
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(value: io::Error) -> Self {
        Self {
            cause: format!("IO Error: {}. Details: {}", value.kind(), value),
        }
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(value: serde_json::Error) -> Self {
        Self {
            cause: format!("SerDe Error: {}", value),
        }
    }
}

impl Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cause)
    }
}

/// An error that occurred during text tokenization or tokenizer loading.
#[derive(Debug)]
#[cfg(feature = "tokenizers")]
pub struct TokenizationError {
    /// Human-readable description of what went wrong.
    pub cause: String,
}

#[cfg(feature = "tokenizers")]
impl From<serde_json::Error> for TokenizationError {
    fn from(value: serde_json::Error) -> Self {
        Self {
            cause: format!("SerDe Error: {}", value),
        }
    }
}

#[cfg(feature = "tokenizers")]
impl std::error::Error for TokenizationError {}

#[cfg(feature = "tokenizers")]
impl Display for TokenizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cause)
    }
}

/// An error indicating an invalid model identifier or filesystem path.
#[derive(Debug)]
pub struct InvalidModelOrPathError {
    /// The model ID or path that was rejected.
    pub model_or_path: String,
    /// Additional details about why the model/path was invalid.
    pub details: String,
}

impl Display for InvalidModelOrPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid model or path to load the model: {}.\n{}",
            self.model_or_path, self.details
        )
    }
}

impl std::error::Error for InvalidModelOrPathError {}

/// An error that occurred while generating an embedding.
#[derive(Debug, Clone)]
pub struct EmbedError {
    /// Human-readable description of what went wrong.
    pub cause: String,
}

impl From<LoadError> for EmbedError {
    fn from(value: LoadError) -> Self {
        Self { cause: value.cause }
    }
}

#[cfg(feature = "tokenizers")]
impl From<TokenizationError> for EmbedError {
    fn from(value: TokenizationError) -> Self {
        Self { cause: value.cause }
    }
}

impl Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cause)
    }
}

impl std::error::Error for EmbedError {}
