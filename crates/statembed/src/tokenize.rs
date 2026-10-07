//! Tokenizer loading and batch encoding of texts.

use std::{fs, path::PathBuf};

use serde::Deserialize;
use tokenizers::{
    TruncationParams, canonicalize_str, from_json,
    pipeline::{EncodeHandle, EncodeOptions, Encoding, PipelineTokenizer},
};

use crate::errors::TokenizationError;

const DEFAULT_MAX_LENGTH: usize = 512;

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

/// Returns the ID of the `unk_token` set in the model config of a `tokenizer.json`,
/// or `None` if the model has no unknown token.
///
/// The unknown token must be one of the tokenizer's added tokens.
fn unknown_token_id(
    content: &str,
    tokenizer: &PipelineTokenizer,
) -> Result<Option<u32>, TokenizationError> {
    let spec: TokenizerSpec = serde_json::from_str(content)?;
    let Some(token) = spec.model.and_then(|m| m.unk_token) else {
        return Ok(None);
    };
    tokenizer
        .get_added_vocabulary()
        .token_to_id(&token)
        .map(Some)
        .ok_or_else(|| TokenizationError {
            cause: format!("unk_token '{token}' not found in vocabulary"),
        })
}

/// Loads a tokenizer from a `tokenizer.json` file and, unless the file already
/// configures its own, sets truncation at `fallback_truncation_length`
/// (default 512).
///
/// Also returns the ID of the unknown token, if the model defines one.
///
/// A tokenizer in the older v1 format is converted in memory. The converted JSON is then
/// saved as `tokenizer.v2.json` next to `path`, so that later loads can skip the conversion.
/// A failed write does not stop the load, and the original file is never changed.
pub fn load_tokenizer(
    path: impl Into<PathBuf>,
    fallback_truncation_length: Option<usize>,
) -> Result<(PipelineTokenizer, TruncationParams, Option<u32>), TokenizationError> {
    let p = path.into();

    let tok_content = fs::read_to_string(&p)?;

    let res = from_json(&tok_content);

    let tokenizer = match res {
        Ok(t) => t,
        Err(e) => {
            if e.to_string()
                .contains("tokenizer version '1.0' is not `2.0`")
            {
                let converted = canonicalize_str(&tok_content)?;

                let _ = fs::write(p.with_file_name("tokenizer.v2.json"), &converted);

                from_json(&converted).map_err(|e| TokenizationError {
                    cause: format!("Error while loading tokenizer from converted JSON: {e}"),
                })?
            } else {
                return Err(TokenizationError {
                    cause: format!("Unable to load tokenizer: {e}"),
                });
            }
        }
    };

    let truncation_params = tokenizer
        .get_truncation()
        .unwrap_or(&TruncationParams {
            max_length: fallback_truncation_length.unwrap_or(DEFAULT_MAX_LENGTH),
            ..Default::default()
        })
        .to_owned();

    let unknown_token = unknown_token_id(&tok_content, &tokenizer)?;

    Ok((tokenizer, truncation_params, unknown_token))
}

/// Encodes each text, truncated but unpadded.
pub fn encode_batch(
    tk: &PipelineTokenizer,
    truncation: &TruncationParams,
    texts: &[&str],
) -> Result<Vec<Encoding>, TokenizationError> {
    let handle: EncodeHandle = tk.encode(
        texts,
        &EncodeOptions {
            add_special_tokens: false,
            encode_special_tokens: true,
            padding: tokenizers::pipeline::Override::Off,
            truncation: tokenizers::pipeline::Override::With(truncation.to_owned()),
        },
    );
    let encodings: Vec<Encoding> = handle.wait().map_err(|e| TokenizationError {
        cause: format!("Error while getting tokenization results: {e}"),
    })?;
    Ok(encodings)
}

/// A fresh, empty directory in the temp directory, for tests.
#[cfg(test)]
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("statembed_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Should create the temp directory");
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tokenizer() {
        let _ = load_tokenizer("testfiles/tokenizer.json", None).expect("Should not fail");
    }

    #[test]
    fn encode_batch_truncates_to_default_max_length() {
        let (tk, tr, _) =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        let long_document = "word ".repeat(3000);

        let encodings = encode_batch(&tk, &tr, &["what is rust?", long_document.as_str()])
            .expect("should encode");

        assert!(encodings[0].len() < DEFAULT_MAX_LENGTH);
        assert_eq!(encodings[1].len(), DEFAULT_MAX_LENGTH);
    }

    #[test]
    fn load_tokenizer_finds_unknown_token_id() {
        let (_, _, unknown_token) =
            load_tokenizer("testfiles/tokenizer.json", None).expect("tokenizer should load");
        // [UNK]
        assert_eq!(unknown_token, Some(1));
    }

    fn token_ids(tk: &PipelineTokenizer, tr: &TruncationParams, text: &str) -> Vec<u32> {
        encode_batch(tk, tr, &[text]).expect("should encode")[0]
            .ids()
            .iter()
            .map(|t| t.id())
            .collect()
    }

    #[test]
    fn load_tokenizer_writes_second_file_for_v1_tokenizer() {
        let dir = temp_dir("writes_v2");
        fs::copy("testfiles/tokenizer.json", dir.join("tokenizer.json")).unwrap();
        assert!(!dir.join("tokenizer.v2.json").exists());

        let (tk, tr, unknown_token) =
            load_tokenizer(dir.join("tokenizer.json"), None).expect("tokenizer should load");

        assert!(dir.join("tokenizer.v2.json").exists());
        // the original is not modified
        assert_eq!(
            fs::read(dir.join("tokenizer.json")).unwrap(),
            fs::read("testfiles/tokenizer.json").unwrap()
        );

        // the second file loads on its own and gives the same result
        let (tk_v2, tr_v2, unknown_token_v2) =
            load_tokenizer(dir.join("tokenizer.v2.json"), None).expect("v2 should load");
        assert_eq!(unknown_token, unknown_token_v2);
        let text = "hello \u{1F980} world, this is a test sentence";
        assert_eq!(token_ids(&tk, &tr, text), token_ids(&tk_v2, &tr_v2, text));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_tokenizer_does_not_write_second_file_for_v2_tokenizer() {
        let dir = temp_dir("no_write_for_v2");
        let v1 = fs::read_to_string("testfiles/tokenizer.json").unwrap();
        fs::write(
            dir.join("tokenizer.json"),
            canonicalize_str(&v1).expect("should convert"),
        )
        .unwrap();

        load_tokenizer(dir.join("tokenizer.json"), None).expect("tokenizer should load");

        assert!(!dir.join("tokenizer.v2.json").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_tokenizer_succeeds_when_second_file_cannot_be_written() {
        let dir = temp_dir("write_fails");
        fs::copy("testfiles/tokenizer.json", dir.join("tokenizer.json")).unwrap();
        // a directory with the same name makes writing the file fail
        fs::create_dir(dir.join("tokenizer.v2.json")).unwrap();

        let (_, _, unknown_token) = load_tokenizer(dir.join("tokenizer.json"), None)
            .expect("a failed write should not stop the load");

        assert_eq!(unknown_token, Some(1));
        fs::remove_dir_all(&dir).ok();
    }
}
