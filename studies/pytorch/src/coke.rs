extern crate csv;
extern crate tch;
use std::{ collections::HashMap, env, error::Error, ops::Sub, path::PathBuf };
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
use crate::{ dataloader::DataLoader, imagedataset::ImageDataset };
use polars::prelude::IntoLazy;
use polars::prelude::col;
const BATCH_SIZE: usize = 100;
use crate::tch::IndexOp;
pub fn coke_test() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let binding = project_dir.clone().join("./data/dataset-cokeornot");

    let binding_train_coke = binding.clone().join("./test/coke/");
    let binding_train_other = binding.clone().join("./test/other/");

    let device = Device::cuda_if_available();

    let paths = vec![binding_train_coke.to_str().unwrap(), binding_train_other.to_str().unwrap()];
    let dataset = ImageDataset::new(paths, device, 90000);
    let data_loader = DataLoader::new(&dataset, 90000);

    let binding = project_dir.clone().join("./binary-coke.ot");
    let mode_file = binding.to_str().unwrap();

    let mut vs = nn::VarStore::new(device);

    let model = build_model(&vs.root(), false);
    vs.load(mode_file).unwrap();

    // let (image, label) = dataset.get(0);
    // let image_buffer = image.copy();

    // let predicted = model.forward_t(&image, false);
    // println!("Classe prevista:\n{:?}", predicted.print());

    // let predicted_labels = predicted.gt(0.5).to_kind(Kind::Int64); // Limiar de decisão para classificação binária

    // println!("Classe prevista:\n{:?}", predicted_labels.print());
    // println!("Label Correta:\n{:?}", label);

    // display_image(
    //     &image_buffer,
    //     64,
    //     format!("clase prevista: {} classe correta {}", predicted_labels.get(0), label)
    // );

    let test_accuracy = evaluate(&model, data_loader, device)?;
    println!("Test Accuracy: {:.2}%", test_accuracy);

    Ok(())
}
pub fn coke_train() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let binding = project_dir.clone().join("./data/dataset-cokeornot");

    let binding_train_coke = binding.clone().join("./trainning/coke/");
    let binding_train_other = binding.clone().join("./trainning/others/");

    let device = Device::cuda_if_available();
    let paths = vec![binding_train_coke.to_str().unwrap(), binding_train_other.to_str().unwrap()];
    let dataset = ImageDataset::new(paths, device, 9000);
    let mut data_loader = DataLoader::new(&dataset, 9000);

    let vs = VarStore::new(tch::Device::cuda_if_available());
    let model = build_model(&vs.root(), true);

    let config = Adam { beta1: 0.9, beta2: 0.999, wd: 0.0001, eps: 1e-8, amsgrad: false };
    let mut optimizer = config.build(&vs, 1e-3)?;

    let epochs = 40;
    for epoch in 1..=epochs {
        let mut running_loss = 0.0;
        let mut running_accuracy = 0.0;

        let mut total_batches = 0;

        while let Some((batch_images, batch_labels)) = data_loader.next_batch() {
            optimizer.zero_grad();

            let predicted = model.forward_t(&batch_images, true);
            let loss = predicted.binary_cross_entropy::<Tensor>(
                &batch_labels.to_kind(Kind::Float),
                None,
                Reduction::Mean
            );

            loss.backward();
            optimizer.step();

            running_loss += loss.double_value(&[]);

            let predicted_classes = predicted.gt(0.5).to_kind(Kind::Int64).to_device(device);
            let equals = predicted_classes.eq_tensor(&batch_labels);
            let accuracy = equals.to_kind(Kind::Float).mean(Kind::Float).double_value(&[]);

            running_accuracy += accuracy;
            total_batches += 1;

            print!(
                "\rÉPOCA {:3} - Loop {:3}: perda {:03.4} - precisão {:03.2}%",
                epoch,
                total_batches,
                loss.double_value(&[]),
                accuracy * 100.0
            );
        }

        data_loader.reset();
        println!(
            "\rÉPOCA {:3} FINALIZADA: perda {:.5} - precisão {:.5}%",
            epoch,
            running_loss / (total_batches as f64),
            (running_accuracy / (total_batches as f64)) * 100.0
        );
    }

    let binding = project_dir.clone().join("./binary-coke.ot");
    let save_model = binding.to_str().unwrap();
    vs.save(save_model)?;

    Ok(())
}
fn display_image(image_tensor: &Tensor, original_size: usize, title: String) {
    let image_tensor = image_tensor.view([-1]);
    let image_tensor = image_tensor * 255.0;
    let image_tensor = image_tensor.to_kind(Kind::Uint8);
    let image_vec = Vec::<f32>::try_from(image_tensor).expect("wrong type of tensor");

    let mut buffer = vec![0u32; image_vec.len()];

    for (i, chunk) in image_vec.chunks(3).enumerate() {
        let r = chunk[0] as u32;
        let g = chunk[1] as u32;
        let b = chunk[2] as u32;
        buffer[i] = (r << 16) | (g << 8) | b;
    }
    let mut window: Window = Window::new(&title, 500, 500, WindowOptions::default()).expect(
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
    mut data_loader: DataLoader<'_>,
    device: Device
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut total_samples = 0;
    let mut correct_predictions = 0;
    let mut all_predicted = Vec::new();
    let mut all_labels = Vec::new();

    while let Some((batch_images, batch_labels)) = data_loader.next_batch() {
        // Move as entradas e etiquetas para o dispositivo
        let batch_images = batch_images.to_device(device);
        let batch_labels = batch_labels.to_device(device);

        // Faz a previsão
        let predicted = model.forward_t(&batch_images, false);
        // Converte previsões para classes binárias
        let predicted_classes = predicted.gt(0.5).to_kind(Kind::Int64);

        // Armazena previsões e etiquetas para calcular a matriz de confusão
        all_predicted.extend(
            predicted_classes.to_kind(Kind::Int64).view([-1]).iter::<i64>().unwrap()
        );
        all_labels.extend(batch_labels.to_kind(Kind::Int64).view([-1]).iter::<i64>().unwrap());

        // Calcula o número de previsões corretas
        let correct = predicted_classes.eq_tensor(&batch_labels);
        correct_predictions += correct.sum(Kind::Int64).int64_value(&[]);
        // Conta o total de amostras
        total_samples += batch_labels.size()[0] as usize;
    }

    // Calcula a acurácia
    let accuracy = if total_samples > 0 {
        ((correct_predictions as f64) / (total_samples as f64)) * 100.0
    } else {
        0.0
    };

    let predicted_tensor = Tensor::from_slice(&all_predicted).view([-1]);
    let labels_tensor = Tensor::from_slice(&all_labels).view([-1]);
    let confusion_matrix = calculate_confusion_matrix(&predicted_tensor, &labels_tensor);
    print_calculate_confusion_matrix(&confusion_matrix, &["coke", "other"]);
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

    let conv1 = nn::conv2d(vs / "conv1", 3, 32, 3, Default::default());
    let bnorm1 = nn::batch_norm2d(vs / "bnorm1", 32, Default::default());
    let conv2 = nn::conv2d(vs / "conv2", 32, 32, 3, Default::default());
    let bnorm2 = nn::batch_norm2d(vs / "bnorm2", 32, Default::default());
    let linear1 = nn::linear(vs / "linear1", 14 * 14 * 32, 128, Default::default());
    let linear2 = nn::linear(vs / "linear2", 128, 128, Default::default());
    let linear3 = nn::linear(vs / "linear3", 128, 1, Default::default());

    nn::seq()
        .add(conv1)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| bnorm1.forward_t(xs, train))
        .add_fn(|xs: &Tensor| {
            if xs.size().len() != 4 {
                panic!("Tensor shape is not 4D before pooling, but {:?}", xs.size());
            }
            // println!("Shape before first pooling: {:?}", xs.size());
            xs.max_pool2d_default(2)
        })

        .add(conv2)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| bnorm2.forward_t(xs, train))
        .add_fn(|xs: &Tensor| {
            if xs.size().len() != 4 {
                panic!("Tensor shape is not 4D before pooling, but {:?}", xs.size());
            }
            // println!("Shape before second pooling: {:?}", xs.size());
            xs.max_pool2d_default(2)
        })

        // Aqui está o ponto onde transformamos o tensor para 2D
        .add_fn(|xs: &Tensor| xs.view([-1, 32 * 14 * 14])) // Ajustado para 32 * 14 * 14
        .add(linear1)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| xs.dropout(0.2, train))

        .add(linear2)
        .add_fn(|xs: &Tensor| xs.relu())
        .add_fn(move |xs: &Tensor| xs.dropout(0.2, train))
        .add(linear3)

        .add_fn(|xs: &Tensor| xs.sigmoid())
}

fn calculate_confusion_matrix(predicted: &Tensor, labels: &Tensor) -> HashMap<(i64, i64), i64> {
    let mut confusion_matrix = HashMap::new();
    let predicted = predicted.to_kind(Kind::Int64).to_device(Device::Cpu);
    let labels = labels.to_kind(Kind::Int64).to_device(Device::Cpu);

    let num_classes = 2;
    for i in 0..predicted.size()[0] {
        let p = predicted.get(i).int64_value(&[]);
        let l = labels.get(i).int64_value(&[]);
        let entry = confusion_matrix.entry((l, p)).or_insert(0);
        *entry += 1;
    }

    confusion_matrix
}

fn print_calculate_confusion_matrix(
    confusion_matrix: &HashMap<(i64, i64), i64>,
    class_names: &[&str]
) {
    let num_classes = class_names.len();

    print!("Confusion Matrix:\n         ");
    for i in 0..num_classes {
        print!("{:<10}", class_names[i]);
    }
    println!();

    for i in 0..num_classes {
        print!("{:<10}", class_names[i]);
        for j in 0..num_classes {
            let value = confusion_matrix.get(&(i as i64, j as i64)).unwrap_or(&0);
            print!("{:<10}", value);
        }
        println!();
    }
}
