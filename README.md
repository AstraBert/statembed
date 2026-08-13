# statembed

Fast, lightweight static text embeddings for Rust.

`statembed` loads pre-trained static embedding models stored in the [Safetensors](https://github.com/huggingface/safetensors) format and optionally tokenizes input text using Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers). Embeddings are produced via mean-pooling over token-level vectors and can optionally be L2-normalized.

## Features

| Feature | Default | Description |
|---------|---------|--------------|
| `tokenizers` | ✅ | Enables text tokenization via `embed_text()`. Requires `tokenizer.json` alongside the model. |
| `hf-hub` | ❌ | Enables downloading models directly from the Hugging Face Hub. |
| `mmap` | ❌ | Uses memory-mapped I/O instead of reading the full file into memory. |



## Quick Start

Add `statembed` to your `Cargo.toml`:

```toml
[dependencies]
statembed = "0.1"
```

### Load from a local directory

```rust
use statembed::StaticEmbedding;

let mut model = StaticEmbedding::from_dir("./my-model", Some(true))?;
let embedding = model.embed_text("hello world", None)?;
```

The directory must contain:

- `model.safetensors` — the embedding tensor
- `tokenizer.json` — the tokenizer (required when the `tokenizers` feature is enabled)

### Download from Hugging Face Hub

Enable the `hf-hub` feature:

```toml
[dependencies]
statembed = { version = "0.1", features = ["hf-hub"] }
```

```rust
use statembed::StaticEmbedding;

let mut model = StaticEmbedding::from_hf_hub("minishlab/potion-base-8M", Some(true), false).await?;
let embedding = model.embed_text("hello world", None)?;
```

Models are cached under `~/.statembed/`.

### Embed pre-tokenized IDs

If you already have token IDs, skip the tokenizer entirely:

```rust
let tokens = vec![6598, 2088]; // "hello world"
let embedding = model.embed_tokens(tokens)?;
```



## How It Works

1. **Lazy loading** — The tensor and tokenizer are loaded from disk on the first embedding call, then cached in memory for subsequent calls.
2. **Mean pooling** — Each token ID maps to a row in the embedding matrix. The rows for all tokens in the input are averaged to produce a single fixed-length vector.
3. **Optional normalization** — When `normalize` is `true`, the resulting vector is L2-normalized (divided by its Euclidean norm).
4. **Token caching** — Decoded token vectors are cached in a `HashMap` so repeated tokens across calls (e.g., common words) don't need to be re-decoded.



## Benchmarks

Run benchmarks with Criterion:

```bash
cargo bench
```

This compares `statembed` against `model2vec-rs` on short, medium, long, and extra-long inputs.

## License

MIT
