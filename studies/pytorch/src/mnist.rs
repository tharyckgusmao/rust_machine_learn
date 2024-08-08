extern crate csv;
extern crate tch;
use std::{ env, error::Error, ops::Sub, path::PathBuf };
use minifb::{ Key, Window, WindowOptions };
use ndarray::{ s, Array };
use polars::{
    datatypes::{ CategoricalOrdering, DataType, Float32Type },
    prelude::{ CsvReadOptions, CsvReader, IndexOrder },
};
use tch::{
    data::{ self, Iter2 },
    nn::{ self, adam, Adam, ModuleT, Optimizer, OptimizerConfig, VarStore },
    vision::{ self, dataset::Dataset, image },
    Device,
    Kind,
    Reduction,
    Tensor,
};
use std::str::FromStr;
use polars::prelude::SerReader;
use crate::utils::{ calculate_confusion_matrix, print_calculate_confusion_matrix };
use polars::prelude::IntoLazy;
use polars::prelude::col;
const BATCH_SIZE: usize = 100;

pub fn mnist_test() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let binding = project_dir.clone().join("data");
    let dataset = vision::mnist::load_dir(binding).unwrap();

    let binding = project_dir.clone().join("./binary-mnist.ot");
    let mode_file = binding.to_str().unwrap();

    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    let model = build_model(&vs.root(), false);
    vs.load(mode_file).unwrap();

    let image = dataset.train_images.select(0, 1);
    let image_buffer = image.copy();
    let image = image.view([1, 1, 28, 28]);

    let predicted = model.forward_t(&image, false);
    let predicted_softmax = predicted.softmax(-1, Kind::Float);
    let predicted_classes = predicted_softmax.argmax(1, false);
    let predicted_values = predicted_softmax
        .gather(1, &predicted_classes.unsqueeze(1), false)
        .squeeze();
    println!("Probabilidades das classes:\n{:?}", predicted_softmax);
    println!("Valor da previsão:\n{:?}", predicted_values);
    println!("Classe prevista:\n{:?}", predicted_classes);
    display_image(&image_buffer, 28, format!("clase prevista: {}", predicted_classes.get(0)));

    Ok(())
}

pub fn mnist_train() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let binding = project_dir.clone().join("data");
    let dataset = vision::mnist::load_dir(binding).unwrap();

    let vs = VarStore::new(tch::Device::cuda_if_available());
    let model = build_model(&vs.root(), true);

    let config = Adam { beta1: 0.9, beta2: 0.999, wd: 0.0001, eps: 1e-8, amsgrad: false };

    let mut optimizer = config.build(&vs, 1e-3)?;

    let epochs = 5;
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut running_accuracy = 0.0;
        let mut correct_predictions = 0;
        let mut total_samples = 0;
        for (bimages, blabels) in dataset.train_iter(256).to_device(vs.device()) {
            let bimages = bimages.view([-1, 1, 28, 28]);
            optimizer.zero_grad();

            let predicted = model.forward_t(&bimages, true);

            let loss = predicted.copy().cross_entropy_for_logits(&blabels);

            loss.backward();
            optimizer.step();
            epoch_loss += loss.double_value(&[]);

            // Cálculo da acurácia
            let predicted_softmax = predicted.softmax(-1, Kind::Float);

            let (_, top_class) = predicted_softmax.topk(1, -1, true, true);

            // Converta blabels e top_class para o mesmo tipo
            let top_class = top_class.view_as(&blabels);
            let correct = top_class.eq_tensor(&blabels);

            total_samples += blabels.size()[0] as usize;
            correct_predictions += correct
                .to_kind(Kind::Float)
                .sum(Kind::Float)
                .double_value(&[]) as usize;
        }
        running_accuracy = ((correct_predictions as f64) / (total_samples as f64)) * 100.0;

        println!("Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%", epoch, epoch_loss, running_accuracy);
    }
    let test_accuracy = evaluate(&model, &dataset, Device::cuda_if_available())?;
    println!("Test Accuracy: {:.2}%", test_accuracy);

    let binding = project_dir.clone().join("./binary-mnist.ot");
    let save_model = binding.to_str().unwrap();
    vs.save(save_model).unwrap();
    Ok(())
}
fn display_image(image_tensor: &Tensor, original_size: usize, title: String) {
    let mut buffer: Vec<u32> = Vec::with_capacity(original_size * original_size);

    let image_tensor = image_tensor * 255.0;

    let mut buffer = vec![0u32; original_size * original_size];
    let image_vec: Vec<f64> = image_tensor.iter::<f64>().unwrap().collect();
    for (i, &pixel) in image_vec.iter().enumerate() {
        let gray = pixel as u32;
        buffer[i] = (gray << 16) | (gray << 8) | gray;
    }
    let mut window = Window::new(&title, 500, 500, WindowOptions::default()).expect(
        "Falha ao criar a janela"
    );

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(&buffer, original_size, original_size)
            .expect("Falha ao atualizar o buffer");
    }
}
fn evaluate(
    model: &impl ModuleT,
    data_loader: &Dataset,
    device: Device
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut total_samples = 0;
    let mut correct_predictions = 0;

    for (bimages, targets) in data_loader.train_iter(256).to_device(device) {
        let bimages = bimages.view([-1, 1, 28, 28]);
        let predicted = model.forward_t(&bimages, false);

        let predicted_softmax = predicted.softmax(-1, Kind::Float);

        let (_, top_class) = predicted_softmax.topk(1, -1, true, true);

        let equals = top_class.eq_tensor(&targets.view_as(&top_class));
        let correct_in_batch = equals
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .double_value(&[]) as usize;

        total_samples += targets.size()[0] as usize;
        correct_predictions += correct_in_batch;
    }

    let accuracy = ((correct_predictions as f64) / (total_samples as f64)) * 100.0;
    Ok(accuracy)
}
fn build_model(vs: &nn::Path, train: bool) -> impl ModuleT {
    // output = (input - filter + 1) / stride
    // convolução 1: (28 - 3 + 1) / 1 = 26x26
    // pooling 1: 13x13
    // convolução 2: (13 - 3 + 1) / 1 = 11x11
    // pooling 2: 5x5
    // 5 * 5 * 32
    // 800 -> 128 -> 128 -> 10

    let conv1 = nn::conv2d(vs / "conv1", 1, 32, 3, Default::default());
    let bnorm1 = nn::batch_norm2d(vs / "bnorm1", 32, Default::default());
    let conv2 = nn::conv2d(vs / "conv2", 32, 32, 3, Default::default());
    let bnorm2 = nn::batch_norm2d(vs / "bnorm2", 32, Default::default());
    let linear1 = nn::linear(vs / "linear1", 32 * 5 * 5, 128, Default::default());
    let linear2 = nn::linear(vs / "linear2", 128, 128, Default::default());
    let output = nn::linear(vs / "output", 128, 10, Default::default());

    nn::seq()
        .add(conv1)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| bnorm1.forward_t(xs, train))
        .add_fn(|xs: &Tensor| xs.max_pool2d_default(2))

        .add(conv2)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| bnorm2.forward_t(xs, train))
        .add_fn(|xs: &Tensor| xs.max_pool2d_default(2))

        .add_fn(|xs: &Tensor| xs.view([-1, 32 * 5 * 5]))
        .add(linear1)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| xs.dropout(0.2, train))

        .add(linear2)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| xs.dropout(0.2, train))

        .add(output)
}
