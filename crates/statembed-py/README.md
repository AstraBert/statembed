# statembed-py

Fast, lightweight static text embeddings.

> _Python bindings for the [`statembed`](../statembed/README.md) Rust crate, built with [PyO3](https://pyo3.rs)._

`statembed-py` loads a static embedding model (a `model.safetensors` file and its `tokenizer.json`) and produces mean-pooled, optionally L2-normalized embeddings from texts. If you already tokenize your texts, you can also pass token IDs.

## Installation

```bash
# with uv
uv add statembed-py
# with pip
pip install statembed-py
```

### Building from source

Requires [Rust](https://rustup.rs) and [maturin](https://www.maturin.rs):

```bash
pip install maturin
maturin develop --release
```

## Usage

```python
from functools import lru_cache
from statembed_py import StaticEmbedding

@lru_cache(maxsize=1)
def get_embedding_model() -> StaticEmbedding:
    # the directory must contain model.safetensors and tokenizer.json/tokenizer.v2.json
    return StaticEmbedding(model_dir="./my-model")

embedding = get_embedding_model().embed_text("hello world")
embeddings = get_embedding_model().embed_texts(["hello world", "goodbye world"])
```

With token IDs from a tokenizer library of your choice, such as [`tokenizers`](https://pypi.org/project/tokenizers/):

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./my-model/tokenizer.json")
tokens = tokenizer.encode("hello world").ids
embedding = get_embedding_model().embed_tokens(tokens)
```

## API

### `StaticEmbedding(model_dir, normalize=True, fallback_truncation_length=None, eager_loading_threshold=None)`

Loads a model from a local directory containing `model.safetensors` and `tokenizer.json`. The model and the tokenizer are loaded lazily on the first embedding call.

- `model_dir: str` — path to the model directory.
- `normalize: bool` — if `True` (default), output embeddings are L2-normalized.
- `fallback_truncation_length: int | None` — truncation length used when the tokenizer sets none. Defaults to 512.
- `eager_loading_threshold: int | None` — size in bytes (vocabulary × dimensions × 4) of the decoded model below which it is fully decoded at load time. Larger models are decoded on demand. Defaults to 256 MB.

A `tokenizer.json` in the older v1 format is converted when it is loaded, and the result is saved next to it as `tokenizer.v2.json` so that later loads are faster. The original file is never changed. If the directory cannot be written, the model still loads.

### `embed_text(text: str) -> list[float]`

Tokenizes the text and returns its embedding. The unknown token is removed before pooling.

### `embed_texts(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]`

Embeds each text, in batches of `batch_size` (defaults to 16). The embeddings are returned in the same order as the texts.

### `embed_tokens(tokens: Sequence[int]) -> list[float]`

Mean-pools the embedding rows for the given token IDs into a single fixed-length vector, applying normalization if enabled. The IDs are used as given, and an ID outside the vocabulary raises an error.

### `embed_tokens_batch(tokens: Sequence[Sequence[int]]) -> list[list[float]]`

Same as `embed_tokens`, for several sequences.

Type stubs (`statembed_py.pyi`) are bundled for editor and type-checker support.

## Development

- `maturin develop` — build and install the extension into the active virtualenv.
- `cargo run --bin stub_gen` — regenerate `statembed_py.pyi` after changing the PyO3 bindings.

## License

MIT
