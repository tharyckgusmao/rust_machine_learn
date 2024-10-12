use std::path::Path;
use burn::{ backend::Autodiff, prelude::Backend, tensor::{ Data, Device, Tensor, TensorData } };
// use burn_wgpu::{ Wgpu, WgpuDevice };
use rand::Rng;

use image::{ self, DynamicImage, GenericImageView };

pub fn load_image_and_resize224<B: Backend>(path: &String, device: &Device<B>) -> Tensor<B, 3> {
    println!("{:?}", path);
    let img = image::open(&path).ok().unwrap();
    let tensor = apply_transform(&img, device);
    return tensor;
}

pub fn apply_transform<B: Backend>(img: &DynamicImage, device: &Device<B>) -> Tensor<B, 3> {
    let mut img: DynamicImage = img.clone();

    let mut rng = rand::thread_rng();

    // Flip horizontal aleatório
    // if rng.gen_bool(0.5) {
    //     flip_horizontal_in_place(&mut img);
    // }

    // Aplicar uma escala aleatória
    let width = 224;
    let height = 224;
    let channels = 3;

    let scale = rng.gen_range(0.8..1.2);
    let (w, h) = img.dimensions();
    let new_w = ((w as f32) * scale) as u32;
    let new_h = ((h as f32) * scale) as u32;
    img = img.resize_exact(new_w, new_h, image::imageops::FilterType::Nearest);

    let img = img.resize_exact(width, height, image::imageops::FilterType::Nearest);
    let img = img.to_rgb8();
    let img_data = img.into_raw();
    let tensor_data: Vec<f64> = img_data
        .iter()
        .map(|&x| (x as f64) / 255.0)
        .collect();

    let tensor_data = TensorData::from(tensor_data.as_slice());
    let tensor: Tensor<B, 1> = Tensor::from_data(tensor_data, device);

    let tensor = tensor.reshape([height as usize, width as usize, channels as usize]);

    return tensor;
}
