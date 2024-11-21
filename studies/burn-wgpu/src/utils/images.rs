use std::path::Path;
use burn::{ backend::Autodiff, prelude::Backend, tensor::{ Data, Device, Tensor, TensorData } };
// use burn_wgpu::{ Wgpu, WgpuDevice };
use rand::Rng;

use image::{
    self,
    imageops::flip_horizontal_in_place,
    DynamicImage,
    GenericImage,
    GenericImageView,
    ImageBuffer,
    Rgb,
    Rgba,
};

pub fn load_image_and_resize224<B: Backend>(
    path: &String,
    device: &Device<B>,
    augmentation: bool
) -> (Tensor<B, 3>, (u32, u32)) {
    // println!("{:?}", path);
    let img = image::open(&path).ok().unwrap();
    let original_size = img.dimensions();
    let tensor = apply_transform(&img, device, augmentation);
    return (tensor, original_size);
}
pub fn load_buffer_image_and_resize224<B: Backend>(
    buffer: DynamicImage,
    device: &Device<B>,
    augmentation: bool
) -> Tensor<B, 3> {
    let tensor = apply_transform(&buffer, device, augmentation);
    return tensor;
}

pub fn apply_transform<B: Backend>(
    img: &DynamicImage,
    device: &Device<B>,
    augmentation: bool
) -> Tensor<B, 3> {
    let width = 224;
    let height = 224;
    let channels = 3;

    let mut img: DynamicImage = img.clone();

    if augmentation {
        let mut rng = rand::thread_rng();

        if rng.gen_bool(0.5) {
            flip_horizontal_in_place(&mut img);
        }
        let scale = rng.gen_range(0.8..1.2);

        let (w, h) = img.dimensions();

        let new_w = ((w as f32) * scale) as u32;
        let new_h = ((h as f32) * scale) as u32;
        img = img.resize_exact(new_w, new_h, image::imageops::FilterType::Nearest);

        //cutout

        let cutout_size = rng.gen_range(0.1..0.3);
        let cutout_w = ((w as f32) * cutout_size) as u32;
        let cutout_h = ((h as f32) * cutout_size) as u32;

        let max_x = new_w.saturating_sub(cutout_w);
        let max_y = new_h.saturating_sub(cutout_h);

        let x = rng.gen_range(0..=max_x);
        let y = rng.gen_range(0..=max_y);

        let cutout_color = Rgba([0, 0, 0, 255]);

        for i in x..(x + cutout_w).min(w) {
            for j in y..(y + cutout_h).min(h) {
                img.put_pixel(i, j, cutout_color);
            }
        }
    }

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
