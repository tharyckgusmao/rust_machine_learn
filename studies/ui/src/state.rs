use std::sync::{ Arc, Mutex };

#[derive(Debug, Default, Clone)]
pub struct NNProgress {
    pub accuracy: Vec<(f64, f64)>,
    pub loss: Vec<(f64, f64)>,
}
#[derive(Debug, Default, Clone)]
pub struct Progress {
    pub current_epoch: u16,
    pub max_epoch: u16,
    pub batch_size: u16,
    pub max_batch: u16,
    pub current_batch: u16,
}
#[derive(Debug, Default, Clone)]
pub struct StateNN {
    pub train_progress: NNProgress,
    pub val_progress: NNProgress,
    pub progress: Progress,
    pub classes: Vec<(String, String)>,
    pub history: Vec<(String, String)>,
}

pub type StateMutex = Arc<Mutex<StateNN>>;
