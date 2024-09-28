use std::{ sync::{ mpsc::{ Receiver, Sender }, Arc, Mutex } };
use crate::state::StateNN;

pub struct Editor {
    pub rx: Receiver<StateNN>,
    pub state_nn: Arc<Mutex<StateNN>>,
}

impl Editor {
    pub fn listen_and_update(&self) {
        loop {
            if let Ok(received_state) = self.rx.recv() {
                if let Ok(mut state_nn) = self.state_nn.lock() {
                    // println!("{:?}", state_nn);

                    *state_nn = received_state;
                } else {
                    println!("failed to lock state_nn");
                }
            }
        }
    }
}
