use std::{ env, path::PathBuf, sync::{ mpsc, Arc, Mutex }, thread, time::{ Duration, Instant } };

use burn::{
    backend::{ libtorch::LibTorchDevice, Autodiff, LibTorch },
    module::AutodiffModule,
    nn::loss::CrossEntropyLoss,
    serde::de,
    tensor::{ Int, Tensor },
};
use burn_wgpu::WgpuDevice;
use image::DynamicImage;
use minifb::{ Key, Window, WindowOptions };
use model::Vgg16;
use nokhwa::{
    pixel_format::{ RgbAFormat, RgbFormat },
    query,
    utils::{ ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType },
    CallbackCamera,
    Camera,
};
use train::accuracy;
use ui::{ editor::Editor, state::StateNN, ui::{ App, AppState } };
use utils::images::load_buffer_image_and_resize224;

mod data;
mod model;
mod train;
mod utils;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = if args.len() < 2 { None } else { Some(args[1].as_str()) };
    match model {
        Some("train") => train(),
        Some("eval") => eval(),
        Some("camera") => camera(),

        _ => println!("Invalid argument"),
    }
}

fn train() {
    let (tx, rx) = mpsc::channel();
    let state_nn = Arc::new(Mutex::new(StateNN::default()));

    let editor = Editor {
        rx,
        state_nn: Arc::clone(&state_nn),
    };

    thread::spawn(move || {
        editor.listen_and_update();
    });

    // println!("Loading Dataset / train");
    // let device = WgpuDevice::BestAvailable;
    let device = LibTorchDevice::Cpu;
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.clone().join("data/train");
    let dataset_val = project_dir.clone().join("data/val");

    let dataset_train = data::Dataset::new(dataset_path.clone());
    // println!("Loading Dataset / val");
    let dataset_val = data::Dataset::new(dataset_val.clone());
    dataset_train.print();
    dataset_val.print();
    let batch_size = 100;
    let dataloader_train = data::DataLoader::new(dataset_train, batch_size as i64, true, true);
    let mut dataloader_val = data::DataLoader::new(dataset_val, batch_size as i64, false, false);
    println!("{}", dataloader_val.len_batch());

    let save_path = project_dir.join("weights");

    train::train_model(
        dataloader_train,
        dataloader_val,
        save_path.to_str().unwrap(),
        tx.clone(),
        batch_size,
        device
    );
}

fn eval() {
    let device = LibTorchDevice::Cpu;
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.clone().join("data/train");
    let dataset_val = project_dir.clone().join("data/val");

    let dataset_train = data::Dataset::new(dataset_path.clone());
    // println!("Loading Dataset / val");
    let dataset_val = data::Dataset::new(dataset_val.clone());
    dataset_train.print();
    dataset_val.print();
    let batch_size = 100;
    let dataloader_train = data::DataLoader::new(dataset_train, batch_size as i64, true, true);
    let mut dataloader_val = data::DataLoader::new(dataset_val, batch_size as i64, false, false);
    println!("{}", dataloader_val.len_batch());

    let weights_path = project_dir.join("weights/best_model.mpk");

    let mut model: Vgg16<Autodiff<LibTorch>> = Vgg16::new_with_pretreinned(
        2,
        &device,
        weights_path
    );
    println!("model {}", model);

    let mut epoch_acc_val = 0.0;
    let mut epoch_loss_val = 0.0;

    let mut running_samples = 0;
    for (i, (tensor, labels)) in (&mut dataloader_val).enumerate() {
        let output = model.forward(tensor);
        let loss = CrossEntropyLoss::new(None, &output.device()).forward(
            output.clone(),
            labels.clone()
        );
        let predictions: Tensor<Autodiff<LibTorch>, 1, Int> = output.clone().argmax(1).squeeze(1);

        println!("{}", predictions);

        let acc = accuracy(output.clone(), labels);
        println!("{}", acc);
        epoch_acc_val += (acc as f64) * (output.clone().dims()[0] as f64);
        epoch_loss_val += (loss.clone().into_scalar() as f64) * (output.dims()[0] as f64);
        running_samples += output.dims()[0];
    }

    let final_acc_val = epoch_acc_val / (running_samples as f64);
    let final_loss_val = epoch_loss_val / (running_samples as f64);

    println!("[Validation] Final Loss{} | Final Accuracy {}%", final_loss_val, final_acc_val);
}

fn camera() {
    let device = LibTorchDevice::Cpu;
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let weights_path = project_dir.join("weights/best_model.mpk");

    let mut model: Vgg16<Autodiff<LibTorch>> = Vgg16::new_with_pretreinned(
        2,
        &device,
        weights_path
    );

    let width = 1280;
    let height = 720;
    let buffer_size = width * height; // Total de pixels
    let buffer = Arc::new(Mutex::new(vec![0u32; buffer_size]));

    let buffer_clone = Arc::clone(&buffer);
    thread::spawn(move || {
        let cameras = query(ApiBackend::Video4Linux).unwrap();
        cameras.iter().for_each(|cam| println!("{:?}", cam));

        let format = RequestedFormat::new::<RgbAFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate
        );
        let first_camera = cameras.first().unwrap();
        let mut threaded = CallbackCamera::new(
            first_camera.index().clone(),
            format,
            |buffer| {}
        ).unwrap();
        threaded.open_stream().unwrap();

        loop {
            if let Ok(frame) = threaded.poll_frame() {
                let image = frame.decode_image::<RgbAFormat>().unwrap();
                let pixel_data = image.clone().into_vec();
                let mut frame_buffer = buffer_clone.lock().unwrap();
                for (i, chunk) in pixel_data.chunks(4).enumerate() {
                    if i >= frame_buffer.len() {
                        break;
                    }
                    frame_buffer[i] =
                        ((chunk[0] as u32) << 16) |
                        ((chunk[1] as u32) << 8) |
                        (chunk[2] as u32) |
                        (0xff << 24);
                }

                let tensor = load_buffer_image_and_resize224::<Autodiff<LibTorch>>(
                    DynamicImage::ImageRgba8(image),
                    &device,
                    false
                );
                let tensor = Tensor::stack(vec![tensor], 0);
                let output = model.forward(tensor);

                let predictions = output.clone().argmax(1).into_data().to_vec::<i64>().unwrap();

                for value in predictions {
                    match value {
                        0 => println!("Não Coca"),
                        1 => println!("Coca"),
                        _ => println!("Valor inesperado: {}", value),
                    }
                }
            }

            thread::sleep(std::time::Duration::from_millis(8));
        }
    });

    let mut window = Window::new(
        "Camera Frame Viewer",
        width,
        height,
        WindowOptions::default()
    ).expect("Unable to create window");

    let frame_rate = Duration::from_millis(8);
    let mut last_time = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_buffer = buffer.lock().unwrap();

        if last_time.elapsed() >= frame_rate {
            window.update_with_buffer(&frame_buffer, width, height).unwrap();
            last_time = Instant::now();
        }
    }
}
