use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{ Autodiff, LibTorch };
use burn::prelude::Backend;
use burn::tensor::{ Data, Device, Int, Shape, Tensor, TensorData };
use image::DynamicImage;
use rand::seq::SliceRandom;
use rand::{ random, thread_rng };
use std::collections::HashMap;
use std::{ fs::read_dir, path::Path };
use image::GenericImageView;
use csv::ReaderBuilder;
use serde_json::Value;
use crate::utils::images::load_image_and_resize224;

#[derive(Debug, Clone)]
pub struct BoundingBox {
    x: i64,
    y: i64,
    width: i64,
    height: i64,
}

#[derive(Debug)]
struct Annotation {
    filename: String,
    boxes: Vec<BoundingBox>,
}

pub struct Dataset {
    root: String,
    image_path: Vec<(i64, String)>,
    class_to_idx: HashMap<String, i64>,
    total_size: usize,
    annotations: HashMap<String, Vec<BoundingBox>>, // Adiciona um campo para as anotações
}

impl Dataset {
    /// This function walks through the root folder and gathers images and creates a Dataset
    pub fn new<T: AsRef<Path>>(root: T) -> Dataset {
        let annotation_file = root.as_ref().join("attributes.csv");
        let root = root.as_ref();

        let mut image_path: Vec<(i64, String)> = Vec::new();
        let mut class_to_idx: HashMap<String, i64> = HashMap::new();
        let annotations = Self::load_annotations_from_csv(annotation_file);
        Self::get_images_and_classes(&root, &mut image_path, &mut class_to_idx);

        Dataset {
            root: root.to_str().unwrap().to_string(),
            total_size: image_path.len(),
            image_path,
            class_to_idx,
            annotations,
        }
    }

    /// In the input folder finds the classes and images
    fn get_images_and_classes(
        dir: &Path,
        image_path: &mut Vec<(i64, String)>,
        class_to_idx: &mut HashMap<String, i64>
    ) {
        let mut class_id = 0;
        for root_class in read_dir(&dir).unwrap() {
            let root_class = root_class.unwrap().path();

            if root_class.is_dir() && !root_class.clone().to_str().unwrap().ends_with(".ignore") {
                Self::get_images_in_folder(&root_class, image_path, class_id as i64);
                let class_name_str = root_class.file_name().unwrap().to_str().unwrap().to_string();
                class_to_idx.insert(class_name_str.clone(), class_id as i64);
                class_id += 1;
            }
        }
    }

    /// Find images with specific extensions "jpg", "png", "jpeg"
    fn get_images_in_folder(dir: &Path, image_path: &mut Vec<(i64, String)>, class_idx: i64) {
        let valid_ext = vec!["jpg", "png", "jpeg", "webp"];

        for file_path in read_dir(&dir).unwrap() {
            let file_path = &file_path.unwrap().path();
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

    fn load_annotations_from_csv<T: AsRef<Path>>(
        file_path: T
    ) -> HashMap<String, Vec<BoundingBox>> {
        let mut annotations = HashMap::new();
        let mut reader = match ReaderBuilder::new().from_path(file_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to open CSV file: {}", e);
                return annotations;
            }
        };

        for result in reader.records() {
            let record = result.expect("Unable to read record");
            let filename = &record[0];
            let region_shape_attributes: Value = serde_json
                ::from_str(&record[5].replace("\"\"", "\""))
                .expect("Failed to parse JSON");

            if let Some(region_shape) = region_shape_attributes.as_object() {
                let x = region_shape.get("x").and_then(Value::as_i64).unwrap_or(0);
                let y = region_shape.get("y").and_then(Value::as_i64).unwrap_or(0);
                let width = region_shape.get("width").and_then(Value::as_i64).unwrap_or(0);
                let height = region_shape.get("height").and_then(Value::as_i64).unwrap_or(0);

                let bounding_box = BoundingBox { x, y, width, height };
                annotations.entry(filename.to_string()).or_insert_with(Vec::new).push(bounding_box);
            }
        }

        annotations
    }

    /// A simple print function for our Dataset
    pub fn print(&self) {
        println!("DATASET ({})", self.root);
        println!("Classes: {:?}", self.class_to_idx);
        println!("Size: {}", self.total_size);
        println!("Sample of data\n{:?}", &self.image_path[1..3]);
    }

    pub fn get_filename(&self, index: usize) -> String {
        let full_path = &self.image_path[index].1;

        let path = Path::new(full_path);

        match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => String::from(""),
        }
    }

    fn get_item(
        &self,
        idx: usize,
        device: &LibTorchDevice,
        augmentation: bool
    ) -> (Tensor<Autodiff<LibTorch>, 3>, i64, Tensor<Autodiff<LibTorch>, 2>) {
        let (tensor, (original_width, original_height)) = load_image_and_resize224(
            &self.image_path[idx].1,
            device,
            augmentation
        );

        let label = self.image_path[idx].0.clone();
        let bounding_boxes = self.annotations.get(&self.get_filename(idx));

        let boxes_tensor: Tensor<Autodiff<LibTorch>, 2> = if let Some(boxes) = bounding_boxes {
            let scale_x = 244.0 / (original_width as f32);
            let scale_y = 244.0 / (original_height as f32);

            let flattened_boxes: Vec<f32> = boxes
                .iter()
                .flat_map(|b| {
                    let adjusted_x = (b.x as f32) * scale_x;
                    let adjusted_y = (b.y as f32) * scale_y;
                    let adjusted_width = (b.width as f32) * scale_x;
                    let adjusted_height = (b.height as f32) * scale_y;

                    vec![adjusted_x, adjusted_y, adjusted_width, adjusted_height]
                })
                .collect();

            if flattened_boxes.len() != 4 {
                panic!("Expected 4 elements in flattened_boxes, got {}", flattened_boxes.len());
            }
            let tensor_data = TensorData::new(flattened_boxes, Shape::new([1, 4]));
            Tensor::from_data(tensor_data, device)
        } else {
            Tensor::empty(Shape::new([1, 4]), device)
        };
        (tensor, label, boxes_tensor)
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
        self.dataset.image_path.shuffle(&mut rng);
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
    type Item = (
        Tensor<Autodiff<LibTorch>, 4>,
        Tensor<Autodiff<LibTorch>, 1, Int>,
        Tensor<Autodiff<LibTorch>, 2>,
    );

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
        let mut bounding_boxes: Vec<Tensor<Autodiff<LibTorch>, 2>> = vec![];

        let device = LibTorchDevice::default();

        for i in start..end {
            let (image_t, label, boxes) = self.dataset.get_item(i, &device, self.train);
            images.push(image_t);

            let tensor_data = TensorData::from([label].as_slice());
            let tensor = Tensor::from_data(tensor_data, &device);
            labels.push(tensor);

            bounding_boxes.push(boxes);
        }
        self.batch_index += 1;

        let batched_images = Tensor::stack(images, 0);
        let batched_labels = Tensor::cat(labels, 0);
        let batched_boxes = Tensor::cat(bounding_boxes, 0);
        Some((batched_images, batched_labels, batched_boxes))
    }
}
