//! Compares eager and lazy loading on models of different sizes.
//!
//! The 8M model is `testfiles/`; the 32M and 128M models live in
//! `testfiles/init_benches/` (gitignored, they are too large). Models that are not on
//! disk are skipped. The loading mode is forced with the eager loading threshold, so
//! every model is measured in both modes regardless of its size.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use statembed::StaticEmbedding;
use std::{hint::black_box, path::Path};

const MODELS: &[(&str, &str)] = &[
    ("8M", "testfiles"),
    ("32M", "testfiles/init_benches/32M"),
    ("128M", "testfiles/init_benches/128M"),
];

/// Number of texts that are tokenized and pooled together.
const BATCH_SIZE: usize = 16;
/// Number of texts embedded in each iteration.
const N_TEXTS: usize = 64;

const TEXTS: &[&str] = &[
    "This is a short sentence that should be embedded fast.",
    "The quick brown fox jumps over the lazy dog while researchers analyze how embedding models capture semantic meaning.",
    "Although static embedding libraries typically generate fixed-length vector representations by averaging or pooling word-level embeddings without accounting for contextual nuances like polysemy, word order, or surrounding syntax, they remain computationally efficient and surprisingly effective for tasks such as document clustering, semantic search, and coarse-grained similarity comparisons across large text corpora in production systems.",
];

#[derive(Clone, Copy)]
enum Mode {
    Eager,
    Lazy,
}

impl Mode {
    const ALL: [Mode; 2] = [Mode::Eager, Mode::Lazy];

    fn name(self) -> &'static str {
        match self {
            Mode::Eager => "eager",
            Mode::Lazy => "lazy",
        }
    }

    /// A table is decoded eagerly when it is smaller than the threshold.
    fn threshold(self) -> Option<u64> {
        match self {
            Mode::Eager => Some(u64::MAX),
            Mode::Lazy => Some(0),
        }
    }
}

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

fn load(dir: &str, mode: Mode) -> StaticEmbedding {
    StaticEmbedding::from_dir(dir, Some(true), None, mode.threshold())
        .expect("Should load the model from directory")
}

fn make_texts(n: usize) -> Vec<&'static str> {
    TEXTS.iter().cycle().take(n).copied().collect()
}

/// Time to read the model and, in eager mode, to decode it.
fn bench_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("init");
    group.sample_size(10);
    for (name, dir) in available_models() {
        for mode in Mode::ALL {
            group.bench_with_input(BenchmarkId::new(mode.name(), name), &dir, |b, dir| {
                b.iter_batched(
                    || load(dir, mode),
                    |mut model| {
                        model.init().expect("Should initialize the model");
                        model
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
    group.finish();
}

/// Time from a freshly created model to the first embedded batch.
///
/// This is the case where lazy loading can win: it skips decoding rows that are never used.
fn bench_first_batch(c: &mut Criterion) {
    let texts = make_texts(N_TEXTS);
    let mut group = c.benchmark_group("init_and_first_batch");
    group.sample_size(10);
    group.throughput(Throughput::Elements(N_TEXTS as u64));
    for (name, dir) in available_models() {
        for mode in Mode::ALL {
            group.bench_with_input(BenchmarkId::new(mode.name(), name), &dir, |b, dir| {
                b.iter_batched(
                    || load(dir, mode),
                    |mut model| {
                        model.init().expect("Should initialize the model");
                        let embeddings = model
                            .embed_texts(black_box(&texts), black_box(Some(BATCH_SIZE)))
                            .expect("Should embed the texts");
                        (model, embeddings)
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
    group.finish();
}

/// Embedding speed once the model is loaded and the rows for these texts are decoded.
fn bench_warm_batch(c: &mut Criterion) {
    let texts = make_texts(N_TEXTS);
    let mut group = c.benchmark_group("warm_batch");
    group.throughput(Throughput::Elements(N_TEXTS as u64));
    for (name, dir) in available_models() {
        for mode in Mode::ALL {
            let mut model = load(dir, mode);
            model.init().expect("Should initialize the model");
            // decodes the rows that these texts need, in lazy mode
            model
                .embed_texts(&texts, Some(BATCH_SIZE))
                .expect("Should embed the texts");
            group.bench_function(BenchmarkId::new(mode.name(), name), |b| {
                b.iter(|| model.embed_texts(black_box(&texts), black_box(Some(BATCH_SIZE))))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_init, bench_first_batch, bench_warm_batch);
criterion_main!(benches);
