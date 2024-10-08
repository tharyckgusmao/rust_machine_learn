mod utils;

use crate::model;
use crate::{ data::DataLoader, model::vgg };
use num_traits::float::Float;
use ui::state::{ self, StateNN };
use std::env;
use std::fs::create_dir;
use std::path::{ Path, PathBuf };
use std::sync::mpsc::Sender;
use tch::{
    nn::{ self, Module, ModuleT, OptimizerConfig },
    vision::{ imagenet, resnet::{ resnet34, resnet34_no_final_layer }, vgg::vgg19 },
    Device,
};

//credits @ramintoosi https://ramintoosi.ir
/// This function trains the model with train and val data loaders
pub fn train_model(
    mut dataloader_train: DataLoader,
    mut dataloader_val: DataLoader,
    save_dir: &str,
    tx: Sender<StateNN>,
    batch_size: u16
) {
    if !Path::new(save_dir).is_dir() {
        create_dir(save_dir).unwrap();
    }

    let device = Device::cuda_if_available();
    println!("The device is {:?}", device);

    let mut vs = nn::VarStore::new(device);

    // let net = model::net(&vs.root(), 2,false);
    let vgg_net = vgg(&vs.root(), 2, false);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let weights_path = project_dir.join("weights/vgg19.ot");
    vs.load(weights_path).unwrap();
    vs.freeze();

    let net = nn
        ::seq_t()
        .add(vgg_net)
        .add(nn::linear(vs.root() / "fc", 4096, 2, Default::default()));

    let lr = 1e-3;

    let mut opt = nn::Adam::default().build(&vs, lr).unwrap();
    opt.set_weight_decay(1e-4);

    let mut scheduler = utils::Scheduler::new(&mut opt, 5, lr, 0.5);

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
    let mut best_loss: f64 = f64::infinity();

    for e in 1..n_epochs {
        let mut epoch_acc_train = 0.0;
        let mut epoch_loss_train = 0.0;
        let mut running_samples = 0;
        for (i, (images, labels)) in (&mut dataloader_train).enumerate() {
            scheduler.opt.zero_grad();
            let out = net.forward_t(&images.to_device(device), true).to_device(Device::Cpu);
            let acc = out.accuracy_for_logits(&labels);
            let loss = out.cross_entropy_for_logits(&labels);
            epoch_acc_train += f64::try_from(acc).unwrap() * (out.size()[0] as f64);
            epoch_loss_train += f64::try_from(&loss).unwrap() * (out.size()[0] as f64);
            scheduler.opt.backward_step(&loss);
            running_samples += out.size()[0];
            state_nn.progress.current_batch = i as u16;

            let history = (
                format!("Epoch {} Train Batch {:.4}", e, i),
                format!("{}%", (epoch_acc_train / (running_samples as f64)) * 100.0),
            );
            state_nn.history.push(history);

            tx.send(state_nn.clone());
        }

        let final_acc_train = (epoch_acc_train / (running_samples as f64)) * 100.0;
        let final_loss_train = epoch_loss_train / (running_samples as f64);

        state_nn.train_progress.accuracy.push((e as f64, final_acc_train));
        // state_nn.train_progress.loss.push((e as f64, final_loss_train));

        let history: (String, String) = (
            format!("-> Epoch {:.4} Train", e),
            format!("{}%", final_acc_train),
        );
        state_nn.history.push(history);

        let history: (String, String) = (
            format!("-> Epoch {:.4} Loss", e),
            format!("{}%", final_loss_train),
        );
        state_nn.history.push(history);

        let mut epoch_acc_val = 0.0;
        let mut epoch_loss_val = 0.0;

        running_samples = 0;
        for (i, (images, labels)) in (&mut dataloader_val).enumerate() {
            let out = net.forward_t(&images.to_device(device), false).to_device(Device::Cpu);
            let loss = out.cross_entropy_for_logits(&labels);
            let acc = out.accuracy_for_logits(&labels);
            epoch_acc_val += f64::try_from(acc).unwrap() * (out.size()[0] as f64);
            epoch_loss_val += f64::try_from(&loss).unwrap() * (out.size()[0] as f64);

            running_samples += out.size()[0];

            let history = (
                format!("Epoch {} Val Batch {:.4}", e, i),
                format!("{}%", (epoch_acc_val / (running_samples as f64)) * 100.0),
            );
            state_nn.history.push(history);
        }

        let final_acc_val = (epoch_acc_val / (running_samples as f64)) * 100.0;
        let final_loss_val = epoch_loss_val / (running_samples as f64);

        state_nn.val_progress.accuracy.push((e as f64, final_acc_val));
        // state_nn.val_progress.loss.push((e as f64, final_loss_val));

        let history: (String, String) = (
            format!("-> Epoch {:.4} Val", e),
            format!("{}%", final_acc_val),
        );
        state_nn.history.push(history);

        let history: (String, String) = (
            format!("-> Epoch {:.4} Loss", e),
            format!("{}%", final_loss_val),
        );
        state_nn.history.push(history);
        state_nn.progress.current_epoch = e;
        epoch_acc_train /= dataloader_train.len() as f64;
        epoch_loss_train /= dataloader_train.len() as f64;

        scheduler.step(epoch_loss_val);

        if epoch_loss_val < best_loss {
            best_loss = epoch_loss_val;
            best_acc = epoch_acc_val;

            vs.save(Path::new(save_dir).join("best_model.ot")).unwrap();
        }

        tx.send(state_nn.clone());
    }
    // println!("Best validation loss = {best_loss:.4}, accuracy={:.4}", best_acc * 100.0);
}
