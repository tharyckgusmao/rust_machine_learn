extern crate csv;
extern crate tch;
use std::{ env, error::Error, ops::Sub, path::PathBuf };
use tch::{
    nn::{ self, adam, Adam, ModuleT, Optimizer, OptimizerConfig, VarStore },
    Reduction,
    Tensor,
};
use std::str::FromStr;

const BATCH_SIZE: usize = 100;
fn main() -> Result<(), Box<dyn Error>> {
    // Carregar dados de entrada (features)

    tch::manual_seed(123);

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let binding = project_dir.clone().join("./entradas.csv");
    let inputs_file = binding.to_str().unwrap();
    let mut rdr_inputs = csv::Reader::from_path(inputs_file)?;
    let mut inputs: Vec<Tensor> = Vec::new();
    for result in rdr_inputs.records() {
        let record = result?;
        let input: Tensor = Tensor::from_slice(
            &record
                .iter()
                .map(|value| f64::from_str(value).unwrap())
                .collect::<Vec<f64>>()
        );
        inputs.push(input);
    }

    // Carregar dados de saída (targets)
    let binding = project_dir.clone().join("./saidas.csv");
    let outputs_file = binding.to_str().unwrap();
    let mut rdr_outputs = csv::Reader::from_path(outputs_file)?;
    let mut outputs: Vec<Tensor> = Vec::new();
    for result in rdr_outputs.records() {
        let record = result?;
        let output: Tensor = Tensor::from_slice(
            &record
                .iter()
                .map(|value| f64::from_str(value).unwrap())
                .collect::<Vec<f64>>()
        );
        outputs.push(output);
    }

    // Converter para tensores
    let inputs_tensor = Tensor::stack(&inputs, 0);
    let outputs_tensor = Tensor::stack(&outputs, 0);

    let inputs_tensor = inputs_tensor.to_kind(tch::Kind::Float);
    let outputs_tensor = outputs_tensor.to_kind(tch::Kind::Float);

    let num_samples = inputs_tensor.size()[0];
    let num_train = ((num_samples as f64) * 0.8) as i64;
    let train_inputs = inputs_tensor.narrow(0, 0, num_train);
    let test_inputs = inputs_tensor.narrow(0, num_train, num_samples - num_train);

    let train_outputs = outputs_tensor.narrow(0, 0, num_train);
    let test_outputs = outputs_tensor.narrow(0, num_train, num_samples - num_train);

    // Definir modelo, otimizador e outros parâmetros de treinamento
    let vs = VarStore::new(tch::Device::cuda_if_available());
    let model = build_model(&vs.root());

    let config = Adam { beta1: 0.9, beta2: 0.999, wd: 0.0001, eps: 1e-8, amsgrad: false };

    let mut optimizer = config.build(&vs, 1e-3)?;

    // Treinamento do modelo
    let num_batches = (train_inputs.size()[0] as usize) / BATCH_SIZE;
    let epochs = 1000;
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * BATCH_SIZE;
            let end_idx = (batch_idx + 1) * BATCH_SIZE;

            let inputs_batch = train_inputs.narrow(0, start_idx as i64, BATCH_SIZE as i64);
            let outputs_batch = train_outputs.narrow(0, start_idx as i64, BATCH_SIZE as i64);

            optimizer.zero_grad();

            let predicted = model.forward_t(&inputs_batch, true);
            // let loss = predicted.copy().mse_loss(&outputs_batch, tch::Reduction::Mean);
            let loss = predicted.binary_cross_entropy::<Tensor>(
                &outputs_batch,
                None,
                Reduction::Mean
            );
            loss.backward();
            optimizer.step();
            epoch_loss += loss.double_value(&[]);
        }

        if epoch % 10 == 0 {
            println!("Epoch: {}, Loss: {:.4}", epoch, epoch_loss / (num_batches as f64));
        }
    }

    let test_accuracy = evaluate_model(&model, &inputs_tensor, &outputs_tensor)?;
    println!("Test Accuracy: {:.2}%", test_accuracy);

    // Calculando a matriz de confusão
    let (true_positives, false_positives, true_negatives, false_negatives) =
        calculate_confusion_matrix(&model, &inputs_tensor, &outputs_tensor)?;
    println!("True Positives: {}", true_positives);
    println!("False Positives: {}", false_positives);
    println!("True Negatives: {}", true_negatives);
    println!("False Negatives: {}", false_negatives);

    Ok(())
}

fn evaluate_model(
    model: &impl ModuleT,
    inputs: &Tensor,
    targets: &Tensor
) -> Result<f64, Box<dyn Error>> {
    let predicted = model.forward_t(inputs, false);
    let predicted_labels = predicted.gt(0.5); // Limiar de decisão para classificação binária
    let correct = predicted_labels
        .eq_tensor(targets)
        .to_kind(tch::Kind::Float)
        .sum(tch::Kind::Float);
    let accuracy = (correct.double_value(&[]) / (targets.size()[0] as f64)) * 100.0;

    Ok(accuracy)
}

fn build_model(vs: &nn::Path) -> impl ModuleT {
    nn::seq()
        .add(nn::linear(vs / "fc1", 30, 16, Default::default()))
        .add_fn(|xs: &Tensor| xs.relu())
        .add(nn::linear(vs / "fc2", 16, 16, Default::default()))
        .add_fn(|xs: &Tensor| xs.relu())
        .add(nn::linear(vs / "fc3", 16, 1, Default::default()))

        .add_fn(|xs| xs.sigmoid())
}

fn calculate_confusion_matrix(
    model: &impl ModuleT,
    inputs: &Tensor,
    targets: &Tensor
) -> Result<(i64, i64, i64, i64), Box<dyn Error>> {
    let predicted = model.forward_t(inputs, false);
    let predicted_labels = predicted.gt(0.5).to_kind(tch::Kind::Int);

    let true_positives = predicted_labels
        .eq_tensor(targets)
        .logical_and(&predicted_labels)
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    let false_positives = predicted_labels
        .ne_tensor(targets)
        .logical_and(&predicted_labels)
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    let true_negatives = predicted_labels
        .eq_tensor(targets)
        .logical_and(&predicted_labels.logical_not())
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    let false_negatives = predicted_labels
        .ne_tensor(targets)
        .logical_and(&predicted_labels.logical_not())
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    Ok((true_positives, false_positives, true_negatives, false_negatives))
}
