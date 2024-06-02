use std::{env, io::Error, path::PathBuf, sync::{Arc, RwLock}};

use rust_bert::{deberta::{DebertaConfigResources, DebertaMergesResources, DebertaModelResources, DebertaVocabResources}, mobilebert::{MobileBertConfigResources, MobileBertModelResources, MobileBertVocabResources}, pipelines::{common::{ModelResource, ModelType}, pos_tagging::{POSConfig, POSModel}, token_classification::{LabelAggregationOption, TokenClassificationConfig}}, resources::{BufferResource, LocalResource}};
use rust_bert::resources::RemoteResource;
use tch::Device;

fn get_weights() -> Vec<u8> {
    let base_path = env::current_dir().expect("Failed to get current directory");
    let model_path = base_path.join("nlp/rust_model.ot");

    let file = std::fs::read(model_path).unwrap();
    return file
}

fn main() {
    
    let base_path = env::current_dir().expect("Failed to get current directory");
    println!("Base path: {:?}", base_path);

    // let model_path = base_path.join("nlp/model.ot");
    // let config_path = base_path.join("nlp/config.json");
    // let vocab_path = base_path.join("nlp/vocab.json");
    // let merges_path = base_path.join("nlp/merges.txt"); 
    
    // println!("Merges path: {:?}", merges_path); // Print do merges_path
    // println!("Model path: {:?}", model_path);
    // println!("Config path: {:?}", config_path);
    // println!("Vocab path: {:?}", vocab_path);

    // if !model_path.exists() {
    //     println!("Model file not found: {:?}", model_path);
    // }
    // if !config_path.exists() {
    //     println!("Config file not found: {:?}", config_path);
    // }
    // if !vocab_path.exists() {
    //     println!("Vocab file not found: {:?}", vocab_path);
    // }

    // if !merges_path.exists() {
    //     println!("Merges file not found: {:?}", merges_path);
    // }

 
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DebertaConfigResources::DEBERTA_BASE_MNLI,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DebertaVocabResources::DEBERTA_BASE_MNLI,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        DebertaMergesResources::DEBERTA_BASE_MNLI,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        DebertaModelResources::DEBERTA_BASE_MNLI,
    ));

    let tokens = TokenClassificationConfig {
        model_type: ModelType::Deberta,
        model_resource: ModelResource::Torch(model_resource),
        config_resource: config_resource,
        vocab_resource: vocab_resource,
        merges_resource: Some(merges_resource),
            ..Default::default()
        };
    let config = POSConfig::from(tokens);
    let pos_model = POSModel::new(config).unwrap();
    let input = ["loving"];
    let output = pos_model.predict(&input);
    println!("{:?}", output)


}
