# statembed

Fast, lightweight static text embeddings for Rust.

`statembed` loads pre-trained static embedding models stored in the [Safetensors](https://github.com/huggingface/safetensors) format and tokenizes input text using Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers). Embeddings are produced via mean-pooling over token-level vectors and can optionally be L2-normalized.

## Features

| Feature | Default | Description |
|---------|---------|--------------|
| `simd` | ✅ | Uses SIMD instructions (via `wide`) to accumulate and divide token vectors. |
| `hf-hub` | ❌ | Enables downloading models directly from the Hugging Face Hub. |
| `rayon` | ❌ | Pools the sequences of each batch in parallel. Without it, they are pooled one after the other. |



## Quick Start

Add `statembed` to your `Cargo.toml`:

```toml
[dependencies]
statembed = "0.1"
```

### Load from a local directory

```rust
use statembed::StaticEmbedding;

let mut model = StaticEmbedding::from_dir("./my-model", Some(true), None, None)?;
let embedding = model.embed_text("hello world")?;
```

The arguments after the path are `normalize`, `fallback_truncation_length` (used when the tokenizer sets no truncation, defaults to 512) and `eager_loading_threshold` (see [How It Works](#how-it-works)).

The directory must contain:

- `model.safetensors` — the embedding tensor
- `tokenizer.json` — the tokenizer (or `tokenizer.v2.json`, see [How It Works](#how-it-works))

### Embed a batch of texts

```rust
let embeddings = model.embed_texts(&["hello world", "goodbye world"], None)?;
```

The second argument is the batch size (defaults to 16). Embeddings are returned in the same order as the texts.

### Download from Hugging Face Hub

Enable the `hf-hub` feature:

```toml
[dependencies]
statembed = { version = "0.1", features = ["hf-hub"] }
```

```rust
use statembed::StaticEmbedding;

let mut model = StaticEmbedding::from_hf_hub("minishlab/potion-base-8M", Some(true), false, None, None).await?;
let embedding = model.embed_text("hello world")?;
```

Models are cached under `~/.statembed/`.

### Embed pre-tokenized IDs

If you already have token IDs, skip the tokenizer entirely:

```rust
let tokens = vec![6598, 2088]; // "hello world"
let embedding = model.embed_tokens(tokens)?;
```

The IDs are used as given. The unknown token is only removed from texts that `statembed` tokenizes itself.

### Load the model in advance

The model and the tokenizer are loaded on the first embedding call. To load them earlier, call `init()`. It loads the tensor and the tokenizer at the same time, on two threads:

```rust
model.init()?;
```



## How It Works

1. **Lazy loading** — The tensor and tokenizer are loaded on the first embedding call (or on `init()`), then kept in memory for subsequent calls. `init()` loads both at the same time, on two threads.
2. **Mean pooling** — Each token ID maps to a row in the embedding matrix. The rows for all tokens in the input are averaged to produce a single fixed-length vector.
3. **Optional normalization** — When `normalize` is `true`, the resulting vector is L2-normalized (divided by its Euclidean norm).
4. **Decoded table** — The embedding matrix is decoded to a flat `f32` table. If the table (vocabulary × dimensions × 4 bytes) is smaller than the eager loading threshold (256 MB by default), the whole matrix is decoded once at load time. Otherwise, rows are decoded the first time their token is seen.
5. **Batching** — Texts are tokenized and pooled in batches, without padding. With the `rayon` feature, the sequences of a batch are pooled in parallel.
6. **Memory mapping** — `model.safetensors` is memory-mapped, so its bytes are not copied: the operating system reads them from disk when they are first used. The file must not be modified while the model is loaded.
7. **Unknown tokens** — The unknown token of the tokenizer (`unk_token`) is removed from each text before pooling, as in `model2vec-rs`.
8. **Converted tokenizers** — A `tokenizer.json` in the older v1 format is converted when it is loaded. The result is saved next to it as `tokenizer.v2.json`, so that later loads can skip the conversion. That file is used for as long as it is not older than `tokenizer.json`, and the original is never changed. If the directory cannot be written, the model still loads. If you do not want the extra file, make the directory read-only.



## Benchmarks

Benchmarks use Criterion and live in `benches/`:

| Benchmark | What it measures |
|-----------|------------------|
| `embed_benchmark` | `statembed` on single texts and on batches of 16, 64 and 256 texts. |
| `m2vec_benchmark` | `model2vec-rs` on the same inputs. |
| `init_benchmark` | Eager and lazy loading: `init()`, `init()` plus the first batch, and a warm batch. |
| `components_benchmark` | Each loading step on its own: file read, tensor load, table decode, tokenizer load and its steps. |

```bash
cargo bench -p statembed --bench embed_benchmark
cargo bench -p statembed --features rayon --bench embed_benchmark   # parallel pooling
```

`init_benchmark` and `components_benchmark` also use the 32M and 128M models in `testfiles/init_benches/32M` and `testfiles/init_benches/128M`. Those files are too large for the repository and are ignored by git. A model that is not on disk is skipped.

To run everything, with and without `rayon`, and print one comparison table (including `model2vec-rs`), use the script from the repository root:

```bash
scripts/run_benches.sh [--quick] [--skip-m2vec] [--summary-only]
```

## License

MIT
