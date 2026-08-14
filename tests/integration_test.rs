use model2vec_rs::model::StaticModel;
use statembed::StaticEmbedding;
use std::sync::OnceLock;

static M2V_MODEL_NO_NORM: OnceLock<StaticModel> = OnceLock::new();
static M2V_MODEL_W_NORM: OnceLock<StaticModel> = OnceLock::new();

fn get_m2v_model(norm: bool) -> &'static StaticModel {
    if norm {
        M2V_MODEL_W_NORM.get_or_init(|| {
            StaticModel::from_pretrained("minishlab/potion-base-8M", None, None, None)
                .expect("Should be able to download model")
        })
    } else {
        M2V_MODEL_NO_NORM.get_or_init(|| {
            StaticModel::from_pretrained("minishlab/potion-base-8M", None, Some(false), None)
                .expect("Should be able to download model")
        })
    }
}

#[test]
#[cfg(feature = "tokenizers")]
fn test_embedding_equality() {
    let fixture_sentence = "hello world! this is a long sentence that both embedding model should embed in the same way";
    let m2v_model = get_m2v_model(false);
    let mut st_model =
        StaticEmbedding::from_dir("testfiles/", Some(false)).expect("Should be able to load model");
    let m2v_encoded = m2v_model.encode_single(fixture_sentence);
    let st_encoded = st_model
        .embed_text(fixture_sentence, None)
        .expect("Should be able to embed text");
    assert_eq!(m2v_encoded, st_encoded);
}

#[test]
#[cfg(feature = "tokenizers")]
fn test_embedding_equality_w_norm() {
    let fixture_sentence = "hello world! this is a long sentence that both embedding model should embed in the same way";
    let m2v_model = get_m2v_model(true);
    let mut st_model =
        StaticEmbedding::from_dir("testfiles/", Some(true)).expect("Should be able to load model");
    let m2v_encoded = m2v_model.encode_single(fixture_sentence);
    let st_encoded = st_model
        .embed_text(fixture_sentence, None)
        .expect("Should be able to embed text");
    assert_eq!(m2v_encoded, st_encoded);
}

#[cfg(feature = "hf-hub")]
#[tokio::test]
async fn test_load_from_hf_hub() {
    use statembed::hf_cache_dir;

    let model = StaticEmbedding::from_hf_hub("erikkaum/lattice-retrieval", None, true)
        .await
        .expect("Should download the model successfully");
    assert_eq!(
        model.base_path,
        hf_cache_dir().join("erikkaum--lattice-retrieval")
    );
    assert!(model.base_path.join("model.safetensors").exists());
    assert!(model.base_path.join("tokenizer.json").exists());
}
