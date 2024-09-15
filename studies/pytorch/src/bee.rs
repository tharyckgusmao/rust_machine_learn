use std::env;
use std::error::Error;
use std::path::PathBuf;

use anyhow::{ bail, Result };
use tch::nn::{ self, ModuleT, OptimizerConfig, VarStore };
use tch::vision::{ imagenet, resnet };
use tch::{ Device, Kind, Tensor };
pub fn bee_test() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.join("data/hymenoptera_data");
    let dataset = imagenet::load_from_dir(dataset_path)?;
    println!("{dataset:?}");

    let model_path = project_dir.join("data/bee.ot");
    println!("Caminho do modelo: {:?}", model_path);

    let device = Device::cuda_if_available();
    let mut vs = VarStore::new(device);
    vs.load(model_path.as_path()).map_err(|op| {
        format!("Erro ao carregar o modelo: {:?}", op);
        op
    })?;

    let net = resnet::resnet34_no_final_layer(&vs.root());
    let linear = nn::linear(vs.root(), 512, 2, Default::default());

    let net2: nn::Sequential = nn
        ::seq()
        .add_fn(move |xs| net.forward_t(xs, false))
        .add(linear);

    let predicted = net2.forward_t(&dataset.test_images, false);
    let probabilities = predicted.softmax(-1, tch::Kind::Float);
    probabilities.print();

    let class = predicted.argmax(-1, false);
    class.print();

    let test_accuracy = predicted.accuracy_for_logits(&dataset.test_labels);

    println!("Test Accuracy: {:.2}%", 100.0 * f64::try_from(test_accuracy)?);

    Ok(())
}

pub fn bee_train() -> Result<()> {
    tch::manual_seed(123);

    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.join("data/hymenoptera_data");
    let model_path = project_dir.join("data/resnet34.ot");

    // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    let dataset = imagenet::load_from_dir(dataset_path)?;
    println!("{dataset:?}");

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = resnet::resnet34_no_final_layer(&vs.root());
    vs.load(model_path)?;

    // Pre-compute the final activations.
    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));
    println!("Train images shape: {:?}", train_images.size());
    println!("Test images shape: {:?}", test_images.size());

    // Initialize the linear layer and optimizer
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    for epoch_idx in 1..6000 {
        let predicted = train_images.apply(&linear);
        let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
        sgd.backward_step(&loss);

        let test_accuracy = test_images.apply(&linear).accuracy_for_logits(&dataset.test_labels);
        println!(
            "Epoch {}: Train Loss = {:.4}, Test Accuracy = {:.2}%",
            epoch_idx,
            f64::try_from(loss)?,
            100.0 * f64::try_from(test_accuracy)?
        );
    }

    let save_model_path = project_dir.join("data/bee.ot");
    vs.save(save_model_path)?;
    Ok(())
}
