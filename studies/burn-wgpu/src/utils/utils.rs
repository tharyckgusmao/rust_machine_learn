use std::f64::INFINITY;

pub struct Scheduler {
    patience: i64,
    factor: f64,
    lr: f64,
    step: i64,
    last_val: f64,
}

impl Scheduler {
    pub fn new(mut patience: i64, lr: f64, mut factor: f64) -> Scheduler {
        if patience < 0 {
            patience = 5;
        }
        if factor < 0.0 {
            factor = 0.95;
        }

        Scheduler {
            patience,
            factor,
            lr,
            step: 0,
            last_val: INFINITY,
        }
    }
    /// Check the input value
    /// If we waited enough, decrease the lr
    pub fn step(&mut self, value: f64) {
        if value < self.last_val {
            self.last_val = value;
            self.step = 0;
        } else {
            self.step += 1;
            if self.step == self.patience {
                self.step = 0;
                self.lr = self.factor * self.lr;
            }
        }
    }

    pub fn get_lr(&self) -> f64 {
        self.lr
    }
}
