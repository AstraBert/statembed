use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::errors::TokenizationError;

pub fn load_tokenizer(path: impl Into<PathBuf>) -> Result<Tokenizer, TokenizationError> {
    let tokenizer = Tokenizer::from_file(path.into()).map_err(|e| TokenizationError {
        cause: e.to_string(),
    })?;
    Ok(tokenizer)
}

pub fn tokenize(tk: &Tokenizer, text: &str) -> Result<Vec<u32>, TokenizationError> {
    let encoding = tk.encode(text, false).map_err(|e| TokenizationError {
        cause: format!("Error while encoding text: {}", e.to_string()),
    })?;
    let ids = encoding.get_ids();
    Ok(ids.to_vec())
}
