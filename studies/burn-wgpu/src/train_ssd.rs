use std::{ env, fs::create_dir, path::{ Path, PathBuf }, sync::mpsc::Sender };

use burn::{
    backend::{ libtorch::LibTorchDevice, Autodiff, LibTorch },
    module::{ AutodiffModule, Module },
    nn::loss::{ CrossEntropyLoss, MseLoss },
    optim::{ AdamConfig, GradientsParams, Optimizer },
    prelude::Backend,
    record::{ FullPrecisionSettings, NamedMpkFileRecorder },
    tensor::{ cast::ToElement, ElementConversion, Int, Tensor },
};
use ui::state::StateNN;

use crate::{ data::DataLoader, model_ssd::Rcnn, utils::utils };

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
    let weights_path = project_dir.join("weights/best_model.mpk");

    let config_optimizer = AdamConfig::new();
    let mut model: Rcnn<Autodiff<LibTorch>> = Rcnn::new(2, &device, Some(weights_path));
    println!("model {}", model);

    // let mut optim = config_optimizer.init();
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
    let mse_loss: MseLoss = MseLoss::new();

    for e in 1..=n_epochs {
        let mut epoch_loss_train = 0.0;
        let mut running_samples = 0;

        // Loop de Treinamento
        for (i, (tensor, labels, loc_targets)) in (&mut dataloader_train).enumerate() {
            // Forward pass
            let (loc_preds, conf_preds) = model.forward(tensor.clone());
            // println!("Loc preds size: {}", loc_preds.shape());
            // println!("Conf preds size: {}", conf_preds);
            // println!("Labels size: {}", labels);
            // println!("Loc targets size: {}", loc_targets.shape());
            let loss_class = CrossEntropyLoss::new(None, &conf_preds.device()).forward(
                conf_preds.clone(),
                labels
            );

            // Calcula a perda para a localização
            let loss_loc = mse_loss.forward(
                loc_preds.clone(),
                loc_targets.clone(),
                burn::nn::loss::Reduction::Mean
            );

            // Cálculo da perda total
            let total_loss = loss_class + loss_loc; // Combine as perdas conforme sua necessidade

            // Backward pass e atualização do modelo
            let grads = total_loss.backward(); // Calcula os gradientes
            let grads = GradientsParams::from_grads(grads, &model); // Cria os parâmetros dos gradientes
            model = config_optimizer.init().step(scheduler.get_lr(), model, grads); // Atualiza o modelo

            // Atualiza a perda da época
            epoch_loss_train +=
                (total_loss.clone().into_scalar() as f64) * (tensor.dims()[0] as f64);
            running_samples += tensor.dims()[0];

            println!(
                "[Train - Epoch {} - Batch {}] Loss {}",
                e,
                i,
                total_loss.clone().into_scalar()
            );

            let history = (
                format!("Epoch {} Train Batch {:.4}", e, i),
                format!("{:.4}", total_loss.clone().into_scalar()),
            );
            state_nn.history.push(history);
            tx.send(state_nn.clone());
        }

        let final_loss_train = epoch_loss_train / (running_samples as f64);
        println!("[Train - Epoch {}] Final Loss {:.4}", e, final_loss_train);

        state_nn.train_progress.loss.push((e as f64, final_loss_train));
        let history = (format!("-> Epoch {:.4} Train", e), format!("{:.4}", final_loss_train));
        state_nn.history.push(history);

        // Loop de Validação
        // let mut epoch_loss_val = 0.0;
        // running_samples = 0;

        // let model_valid = model.valid();
        // for (i, (tensor, labels, _boxes)) in (&mut dataloader_val).enumerate() {
        //     let (class_outputs, _box_outputs) = model_valid.forward(tensor.clone());

        //     // Calcula CrossEntropy para a validação
        //     let loss = CrossEntropyLoss::new(None, &class_outputs.device()).forward(
        //         class_outputs.clone(),
        //         labels.clone()
        //     );

        //     epoch_loss_val += (loss.clone().into_scalar() as f64) * (tensor.dims()[0] as f64);
        //     running_samples += tensor.dims()[0];

        //     println!(
        //         "[Validation - Epoch {} - Batch {}] Loss {:.4}",
        //         e,
        //         i,
        //         loss.clone().into_scalar()
        //     );

        //     let history = (
        //         format!("Epoch {} Val Batch {:.4}", e, i),
        //         format!("{:.4}", loss.clone().into_scalar()),
        //     );
        //     state_nn.history.push(history);
        //     tx.send(state_nn.clone());
        // }

        // let final_loss_val = epoch_loss_val / (running_samples as f64);
        // println!("[Validation - Epoch {}] Final Loss {:.4}", e, final_loss_val);

        // state_nn.val_progress.loss.push((e as f64, final_loss_val));
        // let history = (format!("-> Epoch {:.4} Val", e), format!("{:.4}", final_loss_val));
        // state_nn.history.push(history);

        // scheduler.step(final_loss_val);

        // if final_loss_val < best_loss {
        //     best_loss = final_loss_val;
        //     best_acc = final_loss_val;
        //     let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        //     model
        //         .clone()
        //         .save_file(Path::new(save_dir).join("best_model_ssd.ptk"), &recorder)
        //         .unwrap();
        // }
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(Path::new(save_dir).join("best_model_ssd.ptk"), &recorder).unwrap();
        tx.send(state_nn.clone());
    }
}
fn iou<B: Backend>(box1: Tensor<B, 2>, box2: Tensor<B, 2>) -> Tensor<B, 1> {
    let device = box1.device();

    let indices0 = Tensor::from_data([0], &device);
    let indices1 = Tensor::from_data([1], &device);
    let indices2 = Tensor::from_data([2], &device);
    let indices3 = Tensor::from_data([3], &device);

    let xa = box1.clone().select(1, indices0.clone());
    let ya = box1.clone().select(1, indices1.clone());
    let xb = box1.clone().select(1, indices2.clone());
    let yb = box1.clone().select(1, indices3.clone());

    let box2_xa = box2.clone().select(1, indices0);
    let box2_ya = box2.clone().select(1, indices1);
    let box2_xb = box2.clone().select(1, indices2);
    let box2_yb = box2.clone().select(1, indices3);

    let shape = xb.clone().shape();
    let zero_tensor = Tensor::zeros(shape, &device);

    let inter_width = xb.clone().sub(xa.clone()).max_pair(zero_tensor.clone());
    let inter_height = yb.clone().sub(ya.clone()).max_pair(zero_tensor);
    let inter_area = inter_width.mul(inter_height);

    let area1 = xb.clone().sub(xa).mul(yb.sub(ya.clone()));
    let area2 = box2_xb.sub(box2_xa).mul(box2_yb.sub(box2_ya.clone()));
    let union_area = area1.add(area2).sub(inter_area.clone());

    // Calculando a IOU e retornando como tensor
    let iou_result = inter_area.div(union_area).mean(); // ou .sum() dependendo do que você precisa
    iou_result.reshape([1])
}
pub fn accuracy_rnn<B: Backend>(
    class_output: Tensor<B, 2>,
    loc_output: Tensor<B, 2>,
    class_targets: Tensor<B, 1, Int>,
    loc_targets: Tensor<B, 2>
) -> f32 {
    // Obtem os índices das previsões mais prováveis para as classes
    let class_preds = class_output.argmax(1).squeeze(1);

    let num_samples = class_targets.shape().dims[0] as f32;

    // Compara as classes preditas com as classes reais
    let class_corrects = class_preds.equal(class_targets.clone()).int().sum().into_scalar();
    let class_corrects = class_corrects.elem::<f32>();

    // Calcula o IoU entre as predições e os targets para os bounding boxes
    let ious = iou(loc_output, loc_targets);
    println!("ious {}", ious);

    // Considera uma predição correta se a classe estiver correta e o IoU for maior que um threshold (ex: 0.5)
    let iou_threshold = 0.5;
    let box_corrects = ious.greater_elem(iou_threshold).int().sum().into_scalar();
    let box_corrects = box_corrects.elem::<f32>();
    // Calcula a acurácia final levando em conta classe e localização
    let accuracy = class_corrects + (box_corrects / num_samples) * 2.0 * 100.0;
    accuracy
}
