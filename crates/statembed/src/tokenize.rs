//! Tokenization utilities for the `statembed` library.
//!
//! This module provides helpers for loading `tokie` from JSON files,
//! encoding text, and extracting vocabulary statistics such as median token length.

use std::path::PathBuf;

use tokie::Tokenizer;

use crate::errors::TokenizationError;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct TokenizerSpec {
    model: Option<ModelSpec>,
}

#[derive(Deserialize)]
struct ModelSpec {
    unk_token: Option<String>,
    // any other fields (vocab, merges, etc.) are simply skipped by serde,
    // not allocated, since we don't declare them here.
}

/// Loads a `Tokenizer` from a JSON file on disk.
///
/// Returns also the ID of the `unk_token` if one is defined in the model config.
///
/// # Arguments
/// * `path` - Path to the `tokenizer.json` file.
pub fn load_tokenizer(
    path: impl Into<PathBuf>,
) -> Result<(Tokenizer, Option<u32>), TokenizationError> {
    let p = path.into();

    let file = File::open(&p)?;
    let reader = BufReader::new(file);
    let spec: TokenizerSpec = serde_json::from_reader(reader)?;

    let tokenizer = Tokenizer::from_json(&p).map_err(|e| TokenizationError {
        cause: e.to_string(),
    })?;

    let unk_token_id = if let Some(tok) = spec.model.and_then(|m| m.unk_token) {
        let id = tokenizer
            .token_to_id(&tok)
            .ok_or_else(|| TokenizationError {
                cause: format!("unk_token '{tok}' not found in vocabulary"),
            })?;
        Some(id)
    } else {
        None
    };

    Ok((tokenizer, unk_token_id))
}

/// Encodes a string into a vector of token IDs.
///
/// # Arguments
/// * `tk` - The tokenizer to use.
/// * `text` - The input text to encode.
/// * `unknown_token` - If provided, token IDs matching this value are filtered out.
pub fn tokenize(
    tk: &Tokenizer,
    text: &str,
    unknown_token: Option<u32>,
) -> Result<Vec<u32>, TokenizationError> {
    let encoding = tk.encode(text, false);
    let mut ids = encoding.ids;
    if let Some(unk_id) = unknown_token {
        ids.retain(|&id| id != unk_id);
    }
    Ok(ids)
}

/// Extracts useful statistics from a tokenizer's vocabulary.
///
/// Returns `median_token_length`, i.e. the median length (in bytes) of all vocabulary tokens.
pub fn extract_tokenizer_details(tk: &Tokenizer) -> usize {
    let mut lens: Vec<usize> = tk.get_vocab().keys().map(|tk| tk.len()).collect();
    lens.sort_unstable();

    lens.get(lens.len() / 2).copied().unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_load_tokenizer() {
        let _ = load_tokenizer("testfiles/tokenizer.json").expect("Should not fail");
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_extract_tokenizer_details() {
        let (expected_median_len, expected_unk_token) = (6, Some(1));
        let (tok, unk_token) =
            load_tokenizer("testfiles/tokenizer.json").expect("Should load tokenizer");
        let median_len = extract_tokenizer_details(&tok);
        assert_eq!(expected_median_len, median_len);
        assert_eq!(expected_unk_token, unk_token);
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_tokenize() {
        let (tok, _) = load_tokenizer("testfiles/tokenizer.json").expect("Should load tokenizer");
        let tokens = tokenize(&tok, "hello", Some(1)).expect("Should tokenize successfully");
        assert_eq!(tokens[0], 6598);
    }
}
