# statembed-py

Fast, lightweight static text embeddings.

> _Python bindings for the [`statembed`](../statembed/README.md) Rust crate, built with [PyO3](https://pyo3.rs)._

`statembed-py` loads a static embedding model (a `model.safetensors` file) and produces mean-pooled, optionally L2-normalized embeddings from token IDs. It does not tokenize text itself — pair it with a tokenizer library such as [`tokenizers`](https://pypi.org/project/tokenizers/).

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
from tokenizers import Tokenizer
from statembed_py import StaticEmbedding

@lru_cache(maxsize=1)
def get_embedding_model() -> StaticEmbedding:
    return StaticEmbedding(model_dir="./my-model")  # must contain model.safetensors

@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file("./my-model/tokenizer.json")

def embed(text: str) -> list[float]:
    tokens = get_tokenizer().encode(text).ids
    embedding = get_embedding_model().embed_tokens(tokens)
    return embedding
```

## API

### `StaticEmbedding(model_dir, normalize=True)`

Loads a model from a local directory containing `model.safetensors`. The tensor is loaded lazily on the first `embed_tokens` call.

- `model_dir: str` — path to the model directory.
- `normalize: bool` — if `True` (default), output embeddings are L2-normalized.

### `embed_tokens(tokens: Sequence[int]) -> list[float]`

Mean-pools the embedding rows for the given token IDs into a single fixed-length vector, applying normalization if enabled.

Type stubs (`statembed_py.pyi`) are bundled for editor and type-checker support.

## Development

- `maturin develop` — build and install the extension into the active virtualenv.
- `cargo run --bin stub_gen` — regenerate `statembed_py.pyi` after changing the PyO3 bindings.

## License

MIT
