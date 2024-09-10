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
    vision::{ self, dataset::Dataset, image, imagenet, resnet },
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
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let project_dir = PathBuf::from(manifest_dir);

    // let dataset_path = project_dir.join("data/dataset-cokeornot/test/other/others014.png");
    let dataset_path = project_dir.join("data/dataset-cokeornot/test/coke/coke0003.jpeg");

    let model_path = project_dir.join("./binary-coke.ot");
    println!("Caminho do modelo: {:?}", dataset_path);

    println!("Caminho do modelo: {:?}", model_path);

    // https://github.com/LaurentMazare/tch-rs/releases
    let dataset = imagenet::load_image_and_resize224(dataset_path)?;
    let image_with_batch_dim = dataset.unsqueeze(0);
    println!("Dataset carregado: {:?}", image_with_batch_dim);

    let device = Device::cuda_if_available();

    let mut vs = VarStore::new(device);
    let net = resnet::resnet18_no_final_layer(&vs.root());

    vs.load(model_path.as_path()).map_err(|op| {
        format!("Erro ao carregar o modelo: {:?}", op);
        return op;
    })?;

    let vs = nn::VarStore::new(tch::Device::Cpu);

    let linear = nn::linear(vs.root(), 512, 1, Default::default());
    let net2 = nn
        ::seq()
        .add_fn(move |xs| net.forward_t(xs, false))
        .add(linear)
        .add_fn(|xs: &Tensor| xs.sigmoid());
    let predicted = net2.forward_t(&image_with_batch_dim, false);

    println!("Classe prevista:\n{:?}", predicted.print());

    Ok(())
}

pub fn coke_transfer_train() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.join("data/dataset-cokeornot");
    let model_path = project_dir.join("data/resnet18.ot");
    println!("Caminho do dataset: {:?}", dataset_path);
    println!("Caminho do modelo: {:?}", model_path);

    // Carregar o dataset
    let dataset = imagenet::load_from_dir(dataset_path)?;
    println!("Dataset carregado: {:?}", dataset);

    let device = Device::cuda_if_available();

    // Usar o mesmo VarStore para carregar e treinar o modelo
    let mut vs = VarStore::new(device);
    let net = resnet::resnet18_no_final_layer(&vs.root());

    // Carregar o modelo salvo
    vs.load(model_path.as_path())?;

    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    // Configurar a camada linear e o otimizador
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    let net2 = nn
        ::seq()
        .add(linear)
        .add_fn(|xs: &Tensor| xs.sigmoid());

    // Treinamento
    for epoch_idx in 1..1001 {
        let predicted = train_images.apply(&net2);
        let loss = predicted.binary_cross_entropy::<Tensor>(
            &dataset.train_labels.to_kind(Kind::Float),
            None,
            Reduction::Mean
        );
        sgd.backward_step(&loss);

        let predicted_labels = predicted.ge(0.5).to_kind(tch::Kind::Int64);

        let correct_predictions = predicted_labels
            .eq_tensor(&dataset.test_labels)
            .sum(tch::Kind::Int64);
        let accuracy =
            (100.0 * correct_predictions.double_value(&[])) /
            (dataset.test_labels.size()[0] as f64);

        println!(
            "Epoch {} - Loss: {:.4} - Accuracy: {:.2}%",
            epoch_idx,
            loss.double_value(&[]),
            accuracy
        );
        // let test_accuracy = test_images.apply(&net2).accuracy_for_logits(&dataset.test_labels);
        // println!("{} {:.2}%", epoch_idx, 100.0 * f64::try_from(test_accuracy)?);
    }

    // Salvar o modelo treinado
    let save_model_path = project_dir.join("./binary-coke.ot");
    vs.save(save_model_path)?;

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
