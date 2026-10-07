//! Measures each loading step on its own, to find what dominates model loading.
//!
//! - `file_read`: reading the raw files (the I/O floor, usually served from the page cache)
//! - `load_tensor`: `StaticEmbedding::load_tensor`, i.e. memory-mapping the tensor file and
//!   reading its header (the bytes are read from disk later, when they are first used)
//! - `decode_table`: `StaticEmbedding::eager_load_tokens`, i.e. decoding every row to `f32`
//! - `load_tokenizer`: `StaticEmbedding::load_tokenizer`, the whole tokenizer loading
//! - `tokenizer_steps`: the steps that `load_tokenizer` runs for a v1 `tokenizer.json`
//!
//! The 8M model is `testfiles/`; the 32M and 128M models live in
//! `testfiles/init_benches/` (gitignored, they are too large). Models that are not on
//! disk are skipped.

use criterion::{BatchSize, Bencher, BenchmarkId, Criterion, criterion_group, criterion_main};
use statembed::StaticEmbedding;
use std::{fs, hint::black_box, path::Path};
use tokenizers::{convert::canonicalize_file, from_json, from_json_file};

const MODELS: &[(&str, &str)] = &[
    ("8M", "testfiles"),
    ("32M", "testfiles/init_benches/32M"),
    ("128M", "testfiles/init_benches/128M"),
];

/// The models that are present on disk.
fn available_models() -> Vec<(&'static str, &'static str)> {
    MODELS
        .iter()
        .copied()
        .filter(|(name, dir)| {
            let found = Path::new(dir).join("model.safetensors").exists();
            if !found {
                eprintln!("Skipping the {name} model: no model found in {dir}");
            }
            found
        })
        .collect()
}

/// Times only `routine`; its output is dropped outside of the measurement,
/// so that freeing a large model or tokenizer is not counted.
fn per_iteration<T>(b: &mut Bencher, mut routine: impl FnMut() -> T) {
    b.iter_batched(|| (), |()| routine(), BatchSize::PerIteration)
}

/// A model that eagerly decodes, whatever its size.
fn new_model(dir: &str) -> StaticEmbedding {
    StaticEmbedding::from_dir(dir, Some(true), None, Some(u64::MAX))
        .expect("Should load the model from directory")
}

fn bench_file_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_read");
    group.sample_size(10);
    for (name, dir) in available_models() {
        for file in ["model.safetensors", "tokenizer.json"] {
            let path = Path::new(dir).join(file);
            group.bench_function(BenchmarkId::new(file, name), |b| {
                per_iteration(b, || {
                    fs::read(black_box(&path)).expect("Should read the file")
                })
            });
        }
    }
    group.finish();
}

fn bench_load_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_tensor");
    group.sample_size(10);
    for (name, dir) in available_models() {
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_batched(
                || new_model(dir),
                |mut model| {
                    model.load_tensor().expect("Should load the tensor");
                    model
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn bench_decode_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_table");
    group.sample_size(10);
    for (name, dir) in available_models() {
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_batched(
                || {
                    let mut model = new_model(dir);
                    model.load_tensor().expect("Should load the tensor");
                    model
                },
                |mut model| {
                    model.eager_load_tokens().expect("Should decode the table");
                    model
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn bench_load_tokenizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_tokenizer");
    group.sample_size(10);
    for (name, dir) in available_models() {
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_batched(
                || new_model(dir),
                |mut model| {
                    model.load_tokenizer().expect("Should load the tokenizer");
                    model
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

/// The steps of `load_tokenizer` for a v1 `tokenizer.json`: the file is read and parsed
/// once to find out that it is not v2, again to convert it, and a third time to load it.
fn bench_tokenizer_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer_steps");
    group.sample_size(10);
    for (name, dir) in available_models() {
        let path = Path::new(dir).join("tokenizer.json");
        let converted = canonicalize_file(&path)
            .unwrap_or_else(|_| panic!("Should convert the {name} tokenizer"));

        group.bench_function(BenchmarkId::new("1_read_to_string", name), |b| {
            per_iteration(b, || {
                fs::read_to_string(black_box(&path)).expect("Should read the file")
            })
        });
        // for a v1 file this ends with an error, which is expected
        group.bench_function(BenchmarkId::new("2_parse_as_v2_attempt", name), |b| {
            per_iteration(b, || from_json_file(black_box(&path)).is_ok())
        });
        group.bench_function(BenchmarkId::new("3_convert_v1_to_v2", name), |b| {
            per_iteration(b, || canonicalize_file(black_box(&path)).is_ok())
        });
        group.bench_function(BenchmarkId::new("4_parse_converted", name), |b| {
            per_iteration(b, || {
                from_json(black_box(&converted))
                    .unwrap_or_else(|_| panic!("Should load the converted {name} tokenizer"))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_file_read,
    bench_load_tensor,
    bench_decode_table,
    bench_load_tokenizer,
    bench_tokenizer_steps
);
criterion_main!(benches);
