use statemebed::StaticEmbedding;

fn main() {
    let _ = StaticEmbedding::from_dir("/Users/clee/.statembed/minishlab--potion-code-16M")
        .expect("Should download the model");
}
