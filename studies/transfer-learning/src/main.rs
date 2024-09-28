use std::{ env, path::PathBuf, sync::{ mpsc, Arc, Mutex }, thread };

use color_eyre::eyre::Result;
use model::vgg;
use tch::{ nn::{ self, Module, ModuleT }, Device };
use ui::{ editor::Editor, state::StateNN, ui::{ App, AppState } };

mod data;
mod model;
mod train;
mod inference;

fn main() -> Result<()> {
    color_eyre::install()?;
    let (tx, rx) = mpsc::channel();
    let state_nn = Arc::new(Mutex::new(StateNN::default()));

    let editor = Editor {
        rx,
        state_nn: Arc::clone(&state_nn),
    };

    thread::spawn(move || {
        editor.listen_and_update();
    });

    thread::spawn(move || {
        // println!("Loading Dataset / train");

        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let project_dir = PathBuf::from(manifest_dir);

        let dataset_path = project_dir.clone().join("data/train");
        let dataset_val = project_dir.join("data/val");

        let dataset_train = data::Dataset::new(dataset_path.clone());
        // println!("Loading Dataset / val");
        let dataset_val = data::Dataset::new(dataset_val.clone());
        // dataset_train.print();
        // dataset_val.print();
        let batch_size = 32;
        let dataloader_train = data::DataLoader::new(dataset_train, batch_size as i64, true, true);
        let mut dataloader_val = data::DataLoader::new(
            dataset_val,
            batch_size as i64,
            false,
            false
        );
        // println!("{}", dataloader_val.len_batch());

        let save_path = project_dir.join("weights");

        train::train_model(
            dataloader_train,
            dataloader_val,
            save_path.to_str().unwrap(),
            tx.clone(),
            batch_size
        );

        // let prediction = inference::inference("data/coke/coke004.jpg");
        // println!("{prediction:?}");
        // let device = Device::cuda_if_available();
        // let mut vs = nn::VarStore::new(device);
        // let net = model::net(&vs.root(), 2,false);
        //     let vgg_net = vgg(&vs.root(), 2, false);

        //     let net =
        //         nn::seq_t()
        //             .add(vgg_net)
        //             .add(nn::linear(vs.root() / "fc", 4096, 2, Default::default()));

        //     vs.load("weights/best_model.ot").unwrap();
        //     let mut epoch_acc_val = 0.0;
        //     let mut epoch_loss_val = 0.0;

        //    let mut running_samples = 0;
        //     for (i, (images, labels)) in (&mut dataloader_val).enumerate() {
        //         let out = net
        //             .forward_t(&images.to_device(device),false)
        //             .to_device(Device::Cpu);
        //         let loss = out.cross_entropy_for_logits(&labels);
        //         let acc = out.accuracy_for_logits(&labels);
        //         epoch_acc_val += f64::try_from(acc).unwrap() * (out.size()[0] as f64);
        //         epoch_loss_val += f64::try_from(&loss).unwrap() * (out.size()[0] as f64);
        //         running_samples += out.size()[0];

        //     }
        //     println!("{}",format!(
        //         "loss={:<7.4} - accuracy={:<7.4}",
        //         epoch_loss_val / (running_samples as f64),
        //         epoch_acc_val / (running_samples as f64) * 100.0
        //     ));
    });

    let terminal = ratatui::init();
    let app = App {
        state: AppState::default(),
        state_nn: Arc::clone(&state_nn),
        scroll_position: 0,
    };

    let app_result = app.run(terminal);
    ratatui::restore();
    app_result
}
