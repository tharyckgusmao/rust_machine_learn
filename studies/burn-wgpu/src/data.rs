use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{ Autodiff, LibTorch };
use burn::prelude::Backend;
use burn::tensor::{ Data, Device, Int, Tensor, TensorData };
use image::DynamicImage;
use rand::seq::SliceRandom;
use rand::{ random, thread_rng };
use std::collections::HashMap;
use std::{ fs::read_dir, path::Path };
use image::GenericImageView;

use crate::utils::images::load_image_and_resize224;

pub struct Dataset {
    root: String,
    image_path: Vec<(i64, String)>,
    class_to_idx: HashMap<String, i64>,
    total_size: usize,
}

impl Dataset {
    /// This function walks through the root folder and gathers images and creates a Dataset
    pub fn new<T: AsRef<Path>>(root: T) -> Dataset {
        let root = root.as_ref();

        let mut image_path: Vec<(i64, String)> = Vec::new();
        let mut class_to_idx: HashMap<String, i64> = HashMap::new();

        Self::get_images_and_classes(&root, &mut image_path, &mut class_to_idx);

        Dataset {
            root: root.to_str().unwrap().to_string(),
            total_size: image_path.len(),
            image_path,
            class_to_idx,
        }
    }

    /// In the input folder finds the classes and images
    fn get_images_and_classes(
        dir: &Path,
        image_path: &mut Vec<(i64, String)>,
        class_to_idx: &mut HashMap<String, i64>
    ) {
        for (class_id, root_class) in read_dir(&dir).unwrap().enumerate() {
            let root_class = root_class.unwrap().path().clone();
            if root_class.is_dir() {
                Self::get_images_in_folder(&root_class, image_path, class_id as i64);
                let class_name_str = root_class.file_name().unwrap().to_str().unwrap().to_string();
                class_to_idx.insert(class_name_str.clone(), class_id as i64);
            }
        }
    }

    /// Find images with specific extensions "jpg", "png", "jpeg"
    fn get_images_in_folder(dir: &Path, image_path: &mut Vec<(i64, String)>, class_idx: i64) {
        let valid_ext = vec!["jpg", "png", "jpeg", "webp"];

        for file_path in read_dir(&dir).unwrap() {
            let file_path = &file_path.unwrap().path().clone();
            if
                file_path.is_file() &&
                valid_ext.contains(
                    &file_path.extension().unwrap().to_str().unwrap().to_lowercase().as_str()
                )
            {
                image_path.push((class_idx, file_path.to_str().unwrap().to_string()));
            }
        }
    }

    /// A simple print function for our Dataset
    pub fn print(&self) {
        println!("DATASET ({})", self.root);
        println!("Classes: {:?}", self.class_to_idx);
        println!("Size: {}", self.total_size);
        println!("Sample of data\n{:?}", &self.image_path[1..3]);
    }

    fn get_item(
        &self,
        idx: usize,
        device: &LibTorchDevice,
        augmentation: bool
    ) -> (Tensor<Autodiff<LibTorch>, 3>, i64) {
        let tensor: Tensor<Autodiff<LibTorch>, 3> = load_image_and_resize224(
            &self.image_path[idx].1,
            device,
            augmentation
        );

        (tensor, self.image_path[idx].0.clone())
    }
}

// A struct for our data loader
pub struct DataLoader {
    dataset: Dataset,
    batch_size: i64,
    batch_index: i64,
    shuffle: bool,
    train: bool,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: i64, shuffle: bool, train: bool) -> DataLoader {
        DataLoader {
            dataset,
            batch_size,
            batch_index: 0,
            shuffle,
            train,
        }
    }

    fn shuffle_dataset(&mut self) {
        let mut rng = thread_rng();
        self.dataset.image_path.shuffle(&mut rng)
    }

    /// Total number of images in the dataset
    pub fn len(&self) -> usize {
        self.dataset.total_size
    }

    /// Number of batches based on the dataset size and batch size
    pub fn len_batch(&self) -> usize {
        (self.dataset.total_size + (self.batch_size as usize) - 1) / (self.batch_size as usize)
    }

    /// Get classes
    pub fn get_classes(&self) -> Vec<(String, String)> {
        self.dataset.class_to_idx
            .clone()
            .into_iter()
            .map(|(key, value)| (key, value.to_string()))
            .collect()
    }
}

/// Implement iterator for our DataLoader to get batches of images and labels
impl Iterator for DataLoader {
    type Item = (Tensor<Autodiff<LibTorch>, 4>, Tensor<Autodiff<LibTorch>, 1, Int>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = (self.batch_index * self.batch_size) as usize;
        let mut end = ((self.batch_index + 1) * self.batch_size) as usize;
        if start >= self.dataset.total_size {
            self.batch_index = 0;
            return None;
        }
        if end > self.dataset.total_size {
            end = self.dataset.total_size;
        }
        if self.batch_index == 0 && self.shuffle {
            self.shuffle_dataset();
        }
        let mut images: Vec<Tensor<Autodiff<LibTorch>, 3>> = vec![];
        let mut labels: Vec<Tensor<Autodiff<LibTorch>, 1, Int>> = vec![];

        let device = LibTorchDevice::default();

        for i in start..end {
            let (image_t, label) = self.dataset.get_item(i, &device, self.train);
            images.push(image_t);

            let tensor_data = TensorData::from([label].as_slice());
            let tensor = Tensor::from_data(tensor_data, &device);
            labels.push(tensor);
        }

        self.batch_index += 1;

        let batched_images = Tensor::stack(images, 0);
        let batched_labels = Tensor::cat(labels, 0);

        Some((batched_images, batched_labels))
    }
}
