use model2vec_rs::model::StaticModel;
use statemebed::StaticEmbedding;

#[test]
#[cfg(feature = "tokenizers")]
fn test_embedding_equality() {
    let fixture_sentence = "hello world! this is a long sentence that both embedding model should embed in the same way";
    let m2v_model =
        StaticModel::from_pretrained("minishlab/potion-base-8M", None, Some(false), None)
            .expect("Should be able to download model");
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
    let m2v_model = StaticModel::from_pretrained("minishlab/potion-base-8M", None, None, None)
        .expect("Should be able to download model");
    let mut st_model =
        StaticEmbedding::from_dir("testfiles/", Some(true)).expect("Should be able to load model");
    let m2v_encoded = m2v_model.encode_single(fixture_sentence);
    let st_encoded = st_model
        .embed_text(fixture_sentence, None)
        .expect("Should be able to embed text");
    assert_eq!(m2v_encoded, st_encoded);
}
