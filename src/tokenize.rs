use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::errors::TokenizationError;

pub fn load_tokenizer(path: impl Into<PathBuf>) -> Result<Tokenizer, TokenizationError> {
    let tokenizer = Tokenizer::from_file(path.into()).map_err(|e| TokenizationError {
        cause: e.to_string(),
    })?;
    Ok(tokenizer)
}

pub fn tokenize(
    tk: &Tokenizer,
    text: &str,
    unknown_token: Option<u32>,
) -> Result<Vec<u32>, TokenizationError> {
    let encoding = tk.encode(text, false).map_err(|e| TokenizationError {
        cause: format!("Error while encoding text: {}", e.to_string()),
    })?;
    let mut ids = encoding.get_ids().to_vec();
    if let Some(unk_id) = unknown_token {
        ids.retain(|&id| id != unk_id);
    }
    Ok(ids.to_vec())
}

pub fn extract_tokenizer_details(
    tk: &Tokenizer,
) -> Result<(usize, Option<u32>), TokenizationError> {
    let mut lens: Vec<usize> = tk.get_vocab(false).keys().map(|tk| tk.len()).collect();
    lens.sort_unstable();
    let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

    let spec: serde_json::Value = serde_json::to_value(tk)?;
    let unk_token = spec
        .get("model")
        .and_then(|m| m.get("unk_token"))
        .and_then(serde_json::Value::as_str);
    let unk_token_id = if let Some(tok) = unk_token {
        let id = tk.token_to_id(tok).ok_or_else(|| TokenizationError {
            cause: "unk_token '{tok}' not found in vocabulary".to_string(),
        })?;
        Some(id)
    } else {
        None
    };

    Ok((median_token_length, unk_token_id))
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
        let tok = load_tokenizer("testfiles/tokenizer.json").expect("Should load tokenizer");
        let (median_len, unk_token) =
            extract_tokenizer_details(&tok).expect("Should be able to extract tokenizer details");
        assert_eq!(expected_median_len, median_len);
        assert_eq!(expected_unk_token, unk_token);
    }

    #[test]
    #[cfg(feature = "tokenizers")]
    fn test_tokenize() {
        let tok = load_tokenizer("testfiles/tokenizer.json").expect("Should load tokenizer");
        let tokens = tokenize(&tok, "hello", Some(1)).expect("Should tokenize successfully");
        assert_eq!(tokens[0], 6598);
    }
}
