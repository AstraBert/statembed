use std::{fmt::Display, io};

#[derive(Debug)]
pub struct LoadError {
    pub cause: String,
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(value: io::Error) -> Self {
        Self {
            cause: format!(
                "IO Error: {}. Details: {}",
                value.kind().to_string(),
                value.to_string()
            ),
        }
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(value: serde_json::Error) -> Self {
        Self {
            cause: format!("SerDe Error: {}", value.to_string()),
        }
    }
}

impl Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cause)
    }
}

#[derive(Debug)]
#[cfg(feature = "tokenizers")]
pub struct TokenizationError {
    pub cause: String,
}

#[cfg(feature = "tokenizers")]
impl From<serde_json::Error> for TokenizationError {
    fn from(value: serde_json::Error) -> Self {
        Self {
            cause: format!("SerDe Error: {}", value.to_string()),
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

#[derive(Debug)]
pub struct InvalidModelOrPathError {
    pub model_or_path: String,
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

#[derive(Debug, Clone)]
pub struct EmbedError {
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
