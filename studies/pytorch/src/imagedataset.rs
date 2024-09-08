use std::fs;
use image::{ imageops::flip_horizontal_in_place, DynamicImage, GenericImageView };
use rayon::prelude::*;
use tch::{ Device, Kind, Tensor };
use rand::Rng;
pub struct ImageDataset {
    images: Vec<Tensor>,
    labels: Vec<i64>,
    device: Device,
}

impl ImageDataset {
    pub fn new(folder_paths: Vec<&str>, device: Device, limit: usize) -> Self {
        let mut images = Vec::new();
        let mut labels = Vec::new();

        for folder_path in folder_paths {
            let paths: Vec<_> = fs
                ::read_dir(folder_path)
                .expect("Falha ao ler o diretório")
                .take(limit)
                .collect();

            let results: Vec<_> = paths
                .par_iter()
                .filter_map(|path| {
                    let path = path.as_ref().ok()?.path();
                    if
                        path.extension().and_then(|ext| ext.to_str()) == Some("jpg") ||
                        path.extension().and_then(|ext| ext.to_str()) == Some("jpeg") ||
                        path.extension().and_then(|ext| ext.to_str()) == Some("webp") ||
                        path.extension().and_then(|ext| ext.to_str()) == Some("png")
                    {
                        let img = image::open(&path).ok()?;
                        let tensor = Self::apply_transform(&img, device);
                        let label = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .and_then(|filename| Self::extract_label_from_filename(filename))
                            .unwrap_or(1); // Padrão se não houver label
                        Some((tensor, label))
                    } else {
                        None
                    }
                })
                .collect();

            for (tensor, label) in results {
                images.push(tensor);
                labels.push(label);
            }
        }

        ImageDataset { images, labels, device }
    }

    fn extract_label_from_filename(filename: &str) -> Option<i64> {
        println!("{:?}", filename);
        if filename.contains("dog") || filename.contains("coke") {
            Some(0)
        } else if filename.contains("cat") || filename.contains("other") {
            Some(1)
        } else {
            Some(1)
        }
    }

    pub fn apply_transform(img: &DynamicImage, device: Device) -> Tensor {
        let mut img = img.clone();

        let mut rng = rand::thread_rng();

        // Flip horizontal aleatório
        if rng.gen_bool(0.5) {
            flip_horizontal_in_place(&mut img);
        }

        // Aplicar uma escala aleatória
        let scale = rng.gen_range(0.8..1.2);
        let (width, height) = img.dimensions();
        let new_width = ((width as f32) * scale) as u32;
        let new_height = ((height as f32) * scale) as u32;
        img = img.resize_exact(new_width, new_height, image::imageops::FilterType::Nearest);

        let img = img.resize_exact(64, 64, image::imageops::FilterType::Nearest);
        let img = img.to_rgb8();
        let img_data = img.into_raw();

        let tensor =
            Tensor::from_data_size(&img_data, &[1, 3, 64, 64], Kind::Uint8).to_kind(Kind::Float) /
            255.0;

        tensor.to_device(device)
    }

    // Atualiza uma imagem transformada no dataset
    pub fn update_image(&mut self, index: usize, new_image: Tensor) {
        self.images[index] = new_image;
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn get(&self, index: usize) -> (Tensor, i64) {
        (self.images[index].copy(), self.labels[index])
    }

    pub fn get_device(&self) -> Device {
        self.device
    }
}
