use criterion::{Criterion, criterion_group, criterion_main};
use statemebed::StaticEmbedding;
use std::hint::black_box;

const TO_EMBED: &[(&str, &str)] = &[
    (
        "This is a short sentence that should be embedded fast.",
        "short",
    ),
    (
        "The quick brown fox jumps over the lazy dog while researchers analyze how embedding models capture semantic meaning.",
        "medium",
    ),
    (
        "Although static embedding libraries typically generate fixed-length vector representations by averaging or pooling word-level embeddings without accounting for contextual nuances like polysemy, word order, or surrounding syntax, they remain computationally efficient and surprisingly effective for tasks such as document clustering, semantic search, and coarse-grained similarity comparisons across large text corpora in production systems.",
        "long",
    ),
];

#[cfg(feature = "tokenizers")]
fn bench_no_norm(c: &mut Criterion) {
    let mut model = StaticEmbedding::from_dir("testfiles/", Some(false))
        .expect("Should load the model from directory");
    let mut group = c.benchmark_group("statembed_no_norm");
    for i in TO_EMBED.iter() {
        group.bench_with_input(format!("bench no norm {}", i.1), i, |b, &n| {
            b.iter(|| model.embed_text(black_box(n.0), black_box(None)))
        });
    }
    group.finish();
}

#[cfg(feature = "tokenizers")]
fn bench_w_norm(c: &mut Criterion) {
    let mut model = StaticEmbedding::from_dir("testfiles/", Some(true))
        .expect("Should load the model from directory");
    let mut group = c.benchmark_group("statembed_w_norm");
    for i in TO_EMBED.iter() {
        group.bench_with_input(format!("bench w norm {}", i.1), i, |b, &n| {
            b.iter(|| model.embed_text(black_box(n.0), black_box(None)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_no_norm, bench_w_norm);
criterion_main!(benches);
