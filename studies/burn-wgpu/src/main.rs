use std::{ env, path::PathBuf, sync::{ mpsc, Arc, Mutex }, thread, time::{ Duration, Instant } };

use burn::{
    backend::{ libtorch::LibTorchDevice, Autodiff, LibTorch },
    module::AutodiffModule,
    nn::loss::{ CrossEntropyLoss, MseLoss },
    serde::de,
    tensor::{ Int, Tensor },
};
use burn_wgpu::WgpuDevice;
use image::DynamicImage;
use minifb::{ Key, Window, WindowOptions };
use model::Vgg16;
use model_ssd::Rcnn;
use nokhwa::{
    pixel_format::{ RgbAFormat, RgbFormat },
    query,
    utils::{ ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType },
    CallbackCamera,
    Camera,
};
use train_ssd::accuracy_rnn;
use ui::{ editor::Editor, state::StateNN, ui::{ App, AppState } };
use utils::{
    display::{ confusion_matrix, display_results, print_confusion_matrix },
    images::load_buffer_image_and_resize224,
};

mod data;
mod model;
mod train;
mod utils;
mod model_ssd;
mod train_ssd;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = if args.len() < 2 { None } else { Some(args[1].as_str()) };
    match model {
        Some("train") => train(),
        Some("train_ssd") => train_ssd(),
        Some("eval") => eval(),
        Some("eval_ssd") => eval_ssd(),

        Some("camera") => camera(),
        Some("camera2") => camera_rnn(),

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
    let batch_size = 30;
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

fn train_ssd() {
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
    let batch_size = 20;
    let dataloader_train = data::DataLoader::new(dataset_train, batch_size as i64, true, false);
    let mut dataloader_val = data::DataLoader::new(dataset_val, batch_size as i64, false, false);
    println!("{}", dataloader_val.len_batch());

    let save_path = project_dir.join("weights");

    train_ssd::train_model(
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

    let mut model: Vgg16<Autodiff<LibTorch>> = Vgg16::new_with_pretrained(2, &device, weights_path);
    println!("model {}", model);

    let mut epoch_acc_val = 0.0;
    let mut epoch_loss_val = 0.0;

    let mut running_samples = 0;
    let mut all_predictions = Vec::new();
    let mut all_labels = Vec::new();
    let args: Vec<String> = std::env::args().collect();
    let preview = args.get(2).map(|s| s.as_str());

    for (i, (tensor, labels, boxes)) in (&mut dataloader_val).enumerate() {
        let (loss, output, labels, accuracy) = model.forward_step(
            labels.clone(),
            tensor.clone(),
            &device
        );

        epoch_acc_val += (accuracy as f64) * (output.clone().dims()[0] as f64);
        epoch_loss_val += (loss.clone().into_scalar() as f64) * (output.dims()[0] as f64);
        running_samples += output.dims()[0];

        let predictions = output.clone().argmax(1).into_data().to_vec::<i64>().unwrap();

        all_predictions.extend(predictions.clone());

        let labels: Vec<i64> = labels.clone().into_data().to_vec::<i64>().unwrap();
        all_labels.extend(labels.clone());
        if let Some(arg) = preview {
            let images: Vec<Vec<i64>> = tensor
                .clone()
                .chunk(tensor.dims()[0], 0)
                .iter()
                .map(|t| {
                    let te: Tensor<Autodiff<LibTorch>, 3> = t.clone().squeeze(0);
                    let te = te.mul_scalar(255.0).int();
                    let te = te.reshape([-1]).into_data().to_vec::<i64>().unwrap();
                    return te;
                })
                .collect();
            display_results(images, labels, predictions, 1200, 700);
        }
    }

    let final_acc_val = epoch_acc_val / (running_samples as f64);
    let final_loss_val = epoch_loss_val / (running_samples as f64);

    println!("[Validation] Final Loss{} | Final Accuracy {}%", final_loss_val, final_acc_val);

    let num_classes = 2;
    let matrix = confusion_matrix(all_predictions, all_labels, num_classes);
    print_confusion_matrix(matrix);
}

fn eval_ssd() {
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
    let mut dataloader_train = data::DataLoader::new(dataset_train, batch_size as i64, true, true);
    let mut dataloader_val = data::DataLoader::new(dataset_val, batch_size as i64, false, false);
    println!("{}", dataloader_val.len_batch());

    let weights_path = project_dir.join("weights/best_model_ssd.mpk");

    let mut model: Rcnn<Autodiff<LibTorch>> = Rcnn::new_with_pretrained(2, &device, weights_path);
    println!("model {}", model);
    let mse_loss: MseLoss = MseLoss::new();

    let mut epoch_acc_val = 0.0;
    let mut epoch_loss_val = 0.0;

    let mut running_samples = 0;
    for (i, (tensor, labels, loc_targets)) in (&mut dataloader_train).enumerate() {
        let (loc_preds, conf_preds) = model.forward(tensor.clone());

        let loss_class = CrossEntropyLoss::new(None, &conf_preds.device()).forward(
            conf_preds.clone(),
            labels.clone()
        );

        let loss_loc = mse_loss.forward(
            loc_preds.clone(),
            loc_targets.clone(),
            burn::nn::loss::Reduction::Mean
        );

        let total_loss = loss_class + loss_loc;

        let acc = accuracy_rnn(
            conf_preds.clone(),
            loc_preds.clone(),
            labels.clone(),
            loc_targets.clone()
        );
        println!("{}", acc);
        epoch_acc_val += (acc as f64) * (conf_preds.clone().dims()[0] as f64);
        epoch_loss_val += (total_loss.clone().into_scalar() as f64) * (tensor.dims()[0] as f64);
        running_samples += conf_preds.dims()[0];
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

    let mut model: Vgg16<Autodiff<LibTorch>> = Vgg16::new_with_pretrained(2, &device, weights_path);

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

                // for y in 0..height {
                //     let start = y * width;
                //     let end = start + width;

                //     for x in 0..width / 2 {
                //         let left_idx = start + x;
                //         let right_idx = end - x - 1;

                //         // Troca os pixels nas posições `left_idx` e `right_idx`
                //         frame_buffer[left_idx] = ((pixel_data[right_idx * 4] as u32) << 16)
                //             | ((pixel_data[right_idx * 4 + 1] as u32) << 8)
                //             | (pixel_data[right_idx * 4 + 2] as u32)
                //             | (0xff << 24);

                //         frame_buffer[right_idx] = ((pixel_data[left_idx * 4] as u32) << 16)
                //             | ((pixel_data[left_idx * 4 + 1] as u32) << 8)
                //             | (pixel_data[left_idx * 4 + 2] as u32)
                //             | (0xff << 24);
                //     }
                // }
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
                let output = model.forward(tensor, true);

                let predictions = output.clone().argmax(1).into_data().to_vec::<i64>().unwrap();
                // let predictions = output
                //     .greater_elem(0.5)
                //     .int()
                //     .squeeze::<1>(1)
                //     .into_data()
                //     .to_vec::<i64>()
                //     .unwrap();

                for value in predictions {
                    println!("{}", value);
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
fn camera_rnn() {
    let device = LibTorchDevice::Cpu;
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let weights_path = project_dir.join("weights/best_model_ssd.mpk");

    let mut model: Rcnn<Autodiff<LibTorch>> = Rcnn::new_with_pretrained(2, &device, weights_path);

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

                // Proporção de redimensionamento
                let input_size = 224; // tamanho de entrada do modelo
                let scale_x = (input_size as f32) / (width as f32); // Proporção horizontal
                let scale_y = (input_size as f32) / (height as f32); // Proporção vertical

                let tensor = load_buffer_image_and_resize224::<Autodiff<LibTorch>>(
                    DynamicImage::ImageRgba8(image),
                    &device,
                    false
                );
                let tensor = Tensor::stack(vec![tensor], 0);
                let (box_output, cls_output) = model.forward(tensor);

                let predictions = cls_output.clone().argmax(1).into_data().to_vec::<i64>().unwrap();
                let boxes: Vec<(f32, f32, f32, f32)> = box_output
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .chunks(4)
                    .map(|chunk| (chunk[0], chunk[1], chunk[2], chunk[3])) // (x1, y1, x2, y2)
                    .collect();

                {
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

                    for (value, bbox) in predictions.iter().zip(boxes.iter()) {
                        match value {
                            0 => println!("Não Coca"),
                            1 => {
                                println!("Coca");
                                // Normalizar bounding box
                                let (x1, y1, x2, y2) = *bbox;

                                // Ajustar as coordenadas de acordo com a proporção de redimensionamento
                                let adjusted_bbox = (
                                    x1 / scale_x,
                                    y1 / scale_y,
                                    x2 / scale_x,
                                    y2 / scale_y,
                                );
                                draw_bounding_box(&mut frame_buffer, width, height, adjusted_bbox);
                            }
                            _ => println!("Valor inesperado: {}", value),
                        }
                    }
                } // O mutex é liberado aqui
            }

            thread::sleep(std::time::Duration::from_millis(8)); // Ajustar se necessário
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
        let frame_buffer = buffer.lock().unwrap(); // Acesso ao buffer é seguro aqui

        if last_time.elapsed() >= frame_rate {
            window.update_with_buffer(&frame_buffer, width, height).unwrap();
            last_time = Instant::now();
        }
    }
}

// Função para desenhar um bounding box
fn draw_bounding_box(
    buffer: &mut Vec<u32>,
    width: usize,
    height: usize,
    adjusted_bbox: (f32, f32, f32, f32) // Ajustado
) {
    let color = 0xffff00ff; // Cor do bounding box (amarelo)
    let (x1, y1, x2, y2) = adjusted_bbox;

    // Converter coordenadas ajustadas de volta para pixels
    let x1 = x1 as usize;
    let y1 = y1 as usize;
    let x2 = x2 as usize;
    let y2 = y2 as usize;

    // Desenhar as linhas do bounding box
    for x in x1..=x2 {
        if y1 < height {
            buffer[y1 * width + x] = color; // topo
        }
        if y2 < height {
            buffer[y2 * width + x] = color; // fundo
        }
    }
    for y in y1..=y2 {
        if x1 < width {
            buffer[y * width + x1] = color; // esquerda
        }
        if x2 < width {
            buffer[y * width + x2] = color; // direita
        }
    }
}
