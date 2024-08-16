use crate::imagedataset::ImageDataset;
use tch::Tensor;

pub struct DataLoader<'a> {
    dataset: &'a ImageDataset,
    batch_size: usize,
    indices: Vec<usize>,
    current_index: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a ImageDataset, batch_size: usize) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        DataLoader {
            dataset,
            batch_size,
            indices,
            current_index: 0,
        }
    }
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_index >= self.indices.len() {
            return None;
        }

        let start = self.current_index;
        let end = (self.current_index + self.batch_size).min(self.indices.len());
        self.current_index = end;

        let batch_indices = &self.indices[start..end];
        let images: Vec<Tensor> = batch_indices
            .iter()
            .map(|&i| self.dataset.get(i).0)
            .collect();
        let labels: Vec<i64> = batch_indices
            .iter()
            .map(|&i| self.dataset.get(i).1)
            .collect();

        let image_tensor = Tensor::cat(&images, 0);
        let label_tensor = Tensor::from_slice(&labels).to_kind(tch::Kind::Int64).unsqueeze(1);

        Some((image_tensor, label_tensor))
    }
}
