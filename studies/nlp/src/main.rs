use rust_bert::{mobilebert::{MobileBertConfigResources, MobileBertModelResources, MobileBertVocabResources}, pipelines::{common::{ModelResource, ModelType}, pos_tagging::{POSConfig, POSModel}, token_classification::{LabelAggregationOption, TokenClassificationConfig}}};
use rust_bert::resources::RemoteResource;
use tch::Device;

fn main() {

    let tokens = TokenClassificationConfig{
        model_type: ModelType::MobileBert,
        model_resource: ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
            MobileBertModelResources::MOBILEBERT_ENGLISH_POS,
        ))),
        config_resource: Box::new(RemoteResource::from_pretrained(
            MobileBertConfigResources::MOBILEBERT_ENGLISH_POS,
        )),
        vocab_resource: Box::new(RemoteResource::from_pretrained(
            MobileBertVocabResources::MOBILEBERT_ENGLISH_POS,
        )),
        merges_resource: None,
        lower_case: true,
        strip_accents: Some(true),
        add_prefix_space: None,
        kind: None,
        label_aggregation_function: LabelAggregationOption::First,
        batch_size: 64,
        device: Device::cuda_if_available(),
    };
    let config = POSConfig::from(tokens);
    let pos_model = POSModel::new(config).unwrap();
    let input = ["loving"];
    let output = pos_model.predict(&input);
    println!("{:?}", output)
}
