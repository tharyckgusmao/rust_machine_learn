use crate::data::DataLoader;
use crate::model::{ self, Vgg16 };
use crate::utils::utils;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{ Autodiff, LibTorch };
use burn::module::{ AutodiffModule, Module };
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{ AdamConfig, GradientsParams, Optimizer };
use burn::prelude::Backend;
use burn::record::{ FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder };
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ Device, ElementConversion, Int, Tensor };
use burn::train::metric::AccuracyInput;
use burn_import::pytorch::PyTorchFileRecorder;
use ui::state::{ self, StateNN };
use std::env;
use std::fs::create_dir;
use std::path::{ Path, PathBuf };
use std::sync::mpsc::Sender;

pub fn train_model(
    mut dataloader_train: DataLoader,
    mut dataloader_val: DataLoader,
    save_dir: &str,
    tx: Sender<StateNN>,
    batch_size: u16,
    device: LibTorchDevice
) {
    if !Path::new(save_dir).is_dir() {
        create_dir(save_dir).unwrap();
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);
    let weights_path = project_dir.join("weights/vgg16.pth");

    println!("The device is {:?}", device);

    let config_optimizer = AdamConfig::new();
    let mut model: Vgg16<Autodiff<LibTorch>> = Vgg16::new_with(2, &device, weights_path);
    println!("model {}", model);

    let mut optim = config_optimizer.init();
    let mut scheduler = utils::Scheduler::new(5, 1e-3, 0.5);

    let total_batch_train = dataloader_train.len_batch();
    let total_batch_val = dataloader_val.len_batch();
    let n_epochs = 30;

    let mut state_nn = StateNN::default();
    state_nn.progress.max_batch = total_batch_train as u16;
    state_nn.progress.current_batch = 0;
    state_nn.progress.batch_size = batch_size;
    state_nn.progress.max_epoch = n_epochs;
    state_nn.progress.current_epoch = 0;
    state_nn.classes = dataloader_train.get_classes();

    tx.send(state_nn.clone());
    let mut best_acc = 0.0;
    let mut best_loss = std::f64::MAX;

    for e in 1..n_epochs {
        let mut epoch_acc_train = 0.0;
        let mut epoch_loss_train = 0.0;
        let mut running_samples = 0;

        // Loop de Treinamento
        for (i, (tensor, labels)) in (&mut dataloader_train).enumerate() {
            let output = model.forward(tensor);
            let loss = CrossEntropyLoss::new(None, &output.device()).forward(
                output.clone(),
                labels.clone()
            );

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            model = optim.step(scheduler.get_lr(), model, grads);

            let acc = accuracy(output.clone(), labels);
            epoch_acc_train += (acc as f64) * (output.clone().dims()[0] as f64);
            epoch_loss_train += (loss.clone().into_scalar() as f64) * (output.dims()[0] as f64);
            running_samples += output.dims()[0];

            println!(
                "[Train - Epoch {} - Batch {}] Loss {:.4} | Accuracy {:.2}%",
                e,
                i,
                loss.clone().into_scalar(),
                acc
            );

            let history = (
                format!("Epoch {} Train Batch {:.4}", e, i),
                format!("{}%", (epoch_acc_train / (running_samples as f64)) * 100.0),
            );
            state_nn.history.push(history);

            tx.send(state_nn.clone());
        }

        let final_acc_train = epoch_acc_train / (running_samples as f64);
        let final_loss_train = epoch_loss_train / (running_samples as f64);

        println!(
            "[Train - Epoch {}] Final Loss {:.4} | Final Accuracy {:.2}%",
            e,
            final_loss_train,
            final_acc_train
        );

        state_nn.train_progress.accuracy.push((e as f64, final_acc_train.into()));

        let history = (format!("-> Epoch {:.4} Train", e), format!("{}%", final_acc_train));
        state_nn.history.push(history);

        let history = (format!("-> Epoch {:.4} Loss", e), format!("{}%", final_loss_train));
        state_nn.history.push(history);

        // // Loop de Validação
        let mut epoch_acc_val = 0.0;
        let mut epoch_loss_val = 0.0;
        running_samples = 0;

        let model_valid = model.valid();
        for (i, (tensor, labels)) in (&mut dataloader_val).enumerate() {
            let output = model.forward(tensor);
            let loss = CrossEntropyLoss::new(None, &output.device()).forward(
                output.clone(),
                labels.clone()
            );

            let acc = accuracy(output.clone(), labels);
            epoch_acc_val += (acc as f64) * (output.clone().dims()[0] as f64);
            epoch_loss_val += (loss.clone().into_scalar() as f64) * (output.dims()[0] as f64);
            running_samples += output.dims()[0];

            println!(
                "[Validation - Epoch {} - Batch {}] Loss {:.4} | Accuracy {:.2}%",
                e,
                i,
                loss.clone().into_scalar(),
                acc
            );

            let history = (
                format!("Epoch {} Val Batch {:.4}", e, i),
                format!("{}%", (epoch_acc_val / (running_samples as f64)) * 100.0),
            );
            state_nn.history.push(history);

            tx.send(state_nn.clone());
        }

        let final_acc_val = epoch_acc_val / (running_samples as f64);
        let final_loss_val = epoch_loss_val / (running_samples as f64);

        println!(
            "[Validation - Epoch {}] Final Loss {:.4} | Final Accuracy {:.2}%",
            e,
            final_loss_val,
            final_acc_val
        );

        state_nn.val_progress.accuracy.push((e as f64, final_acc_val));

        let history = (format!("-> Epoch {:.4} Val", e), format!("{}%", final_acc_val));
        state_nn.history.push(history);

        let history = (format!("-> Epoch {:.4} Loss", e), format!("{}%", final_loss_val));
        state_nn.history.push(history);

        scheduler.step(epoch_loss_val);

        if epoch_loss_val < best_loss {
            best_loss = epoch_loss_val;
            best_acc = final_acc_val;
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            model.clone().save_file(Path::new(save_dir).join("best_model.ptk"), &recorder).unwrap();
        }

        tx.send(state_nn.clone());
    }
}

pub fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    // Obtem os índices das previsões mais prováveis
    let predictions = output.argmax(1).squeeze(1);

    // Calcula o número total de previsões
    let num_predictions = targets.shape().dims[0] as f32;

    // Compara as previsões com os valores reais e conta o número de acertos
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    // Calcula a acurácia como uma porcentagem
    (num_corrects.elem::<f32>() / num_predictions) * 100.0
}
