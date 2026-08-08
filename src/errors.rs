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
