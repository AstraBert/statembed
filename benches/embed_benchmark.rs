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
    (
        r#"The Lighthouse Keeper's Ledger

        Mira had kept the log for eleven years, and in eleven years, nothing had ever been wrong with the light.

        That was the job, mostly. Not the romantic version people imagined — storms and shipwrecks and heroic vigils — but a ledger. Fuel levels. Lamp hours. Wind speed at dawn. She wrote in pencil because pen bled through the damp pages, and she wrote every day at six a.m. whether or not there was anything to say.

        On the morning it started, there was something to say, and she didn't know how to say it.

        The light had gone out at 3:14 a.m. Not flickered — gone out, clean, like someone had reached up and turned a switch. She'd been asleep and hadn't seen it happen, only the absence of the sweep across her bedroom ceiling that had lulled her to sleep for over a decade. She woke at 3:16, disoriented by the dark, and ran up the spiral stairs in her socks.

        By 3:19 the light was on again. Working. Warm. As if nothing had happened.

        She wrote it in the ledger anyway. *0314 — outage, cause unknown, duration approx 5 min. Investigate.*

        She investigated. She checked the bulb, the rotation gears, the backup generator that hadn't needed to kick in because apparently nothing had actually failed. Every system reported nominal. She called the mainland office, and a bored technician named Owen told her it was probably a power flicker on the grid side, nothing to worry about, these things happened sometimes.

        Except Mira knew the lighthouse ran its own generator specifically so grid flickers on the mainland couldn't touch it. That was the entire point of the redundancy. She didn't say this to Owen. She wrote it in the ledger instead, in the small neat hand she'd developed over eleven years of nobody reading these pages but her.

        The second outage came four nights later, at 3:14 a.m. again, down to the minute. This time she was awake, sitting up in bed with a cup of tea gone cold, waiting for it. She watched the beam die through her window. Five minutes of true dark over the water. Then it returned.

        She climbed the stairs anyway, though there was nothing to fix. She stood at the top with her hand on the cold glass and looked out at the black Atlantic, and for the first time in eleven years she felt something she didn't have a word for yet. Not fear exactly. More like the sensation of being looked back at.

        She began sleeping with the ledger beside her bed.

        The third time, she was ready. She sat at the base of the light with a thermos and a notebook — a second one, not the official log, because she didn't yet know what category this belonged in. At 3:13 she started the stopwatch on her phone. At 3:14:00, the light went out. She counted. At 3:19:00, exactly, it came back.

        Five minutes. Every time. To the second.

        That wasn't mechanical failure. Mechanical failure didn't check a clock.

        She spent the next week doing the thing she was best at: making a ledger. Not of fuel and wind now, but of the outages themselves — when, how long, what the sea looked like, what she'd been dreaming about beforehand, if she could remember. She was a methodical woman and she trusted method more than she trusted fear, so she let the method carry her through the part where she might otherwise have simply left.

        By the fourth outage she noticed something in her own data. Each blackout lasted exactly five minutes, but the *interval* between them was shrinking. Four nights, then three, then two. If the pattern held, the fifth outage would come the very next night. And the one after that, if the interval kept halving, would arrive before the previous one had even properly ended.

        She almost called Owen again. She got as far as dialing before she put the phone down, because she understood, with the particular clarity of someone who has spent eleven years writing down small true things, that this was not a problem the mainland office had a form for.

        That night she didn't wait in bed. She climbed to the lamp room at 3:00 and sat facing the light directly, notebook in her lap, pencil ready, the way she'd sit facing anything she needed to understand by watching it closely enough.

        At 3:14, the light went out.

        But this time, in the dark, she heard it — not a sound exactly, more a pressure, like the moment before a held breath breaks. And in the five minutes of blackness she was no longer sure the room around her was the same room. The air smelled of salt and something underneath the salt, something old. She did not move. She did not turn on her phone's flashlight, though her thumb hovered over the button the entire time. She only listened, the way she'd trained herself to notice small changes — a degree of wind, a flicker of fuel — for eleven quiet years.

        At 3:19, the light returned, and the room was the room again, and she was alone, and her hands were shaking too hard to write anything at all.

        She sat there until dawn. Then she went downstairs, opened the official ledger to today's page, and in her small neat hand wrote only: *0314 — outage, five minutes, as before.*

        She did not write what she'd heard. She did not write that the interval, by her own arithmetic, meant the next one would begin before this one had fully ended — that soon there would be no interval left at all, only dark.

        Some things, she'd decided, watching the beam sweep steady and bright across the water one more time, did not belong in a ledger anyone else might read. They belonged only to the person keeping watch.

        She closed the book and went to make coffee, because in four hours it would be six a.m., and there would be fuel levels to record, and lamp hours, and wind speed at dawn — and whatever else the night had left for her to carry into the light."#,
        "extra-long",
    ),
];

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
