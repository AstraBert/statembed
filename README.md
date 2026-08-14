# statembed

Fast, lightweight static text embeddings.

`statembed` loads pre-trained static embedding models stored in the [Safetensors](https://github.com/huggingface/safetensors) format and produces fixed-length text embeddings via mean-pooling over token-level vectors, with optional L2 normalization. There's no neural network forward pass at inference time (just a lookup-and-average over an embedding matrix) which makes it extremely fast compared to transformer-based embedding models, at some cost to embedding quality.

This repository is a Cargo workspace with two crates:

| Crate | Language | Description |
|-------|----------|--------------|
| [`statembed`](crates/statembed) | Rust | Core library: model loading, tokenization, pooling, normalization. |
| [`statembed-py`](crates/statembed-py) | Python (via [PyO3](https://pyo3.rs)) | Python bindings exposing embedding from pre-tokenized input. |

## How it works

1. **Lazy loading** — the model tensor (and tokenizer, in Rust) are loaded from disk on the first embedding call, then cached in memory.
2. **Mean pooling** — each token ID maps to a row in the embedding matrix; the rows for all tokens in the input are averaged into a single fixed-length vector.
3. **Optional normalization** — when enabled, the resulting vector is L2-normalized.
4. **Token caching** — decoded token vectors are cached so repeated tokens across calls don't need to be re-decoded.

## Rust: `statembed`

```toml
[dependencies]
statembed = "0.1"
```

```rust
use statembed::StaticEmbedding;

let mut model = StaticEmbedding::from_dir("./my-model", Some(true))?;
let embedding = model.embed_text("hello world", None)?;
```

The model directory must contain `model.safetensors`, plus `tokenizer.json` if using the (default) `tokenizers` feature to embed raw text. Models can also be downloaded directly from the Hugging Face Hub with the `hf-hub` feature. See [`crates/statembed/README.md`](crates/statembed/README.md) for the full feature list (`tokenizers`, `hf-hub`, `mmap`, `simd`) and API details.

## Python: `statembed-py`

```bash
uv add statembed-py
```

Python bindings embed pre-tokenized input only, so you need to pair them with a tokenizer library such as [`tokenizers`](https://pypi.org/project/tokenizers/):

```python
from tokenizers import Tokenizer
from statembed_py import StaticEmbedding

model = StaticEmbedding(model_dir="./my-model")  # must contain model.safetensors
tokenizer = Tokenizer.from_file("./my-model/tokenizer.json")

tokens = tokenizer.encode("hello world").ids
embedding = model.embed_tokens(tokens)
```

See [`crates/statembed-py/README.md`](crates/statembed-py/README.md) for installation options, building from source, and the full API.

## Development

This is a Cargo workspace (`resolver = "3"`); run standard Cargo commands from the repo root:

```bash
cargo build
cargo test
cargo bench -p statembed   # compares against model2vec-rs
```

## License

MIT
