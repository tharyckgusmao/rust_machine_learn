use std::{ sync::{ mpsc, Arc, Mutex }, thread, time::Duration };

use ui::{ App, AppState, StateNN };
use color_eyre::Result;

pub mod ui;
fn main() -> Result<()> {
    color_eyre::install()?;
    let (tx, rx) = mpsc::channel();
    let terminal = ratatui::init();

    let state_nn = Arc::new(Mutex::new(StateNN::default()));

    let app = App {
        state: AppState::default(),
        state_nn: Arc::clone(&state_nn),
        rx,
    };
    let state_nn_clone = Arc::clone(&state_nn);

    thread::spawn(move || {
        for epoch in 0..10 {
            thread::sleep(Duration::from_secs(1));

            if let Ok(mut state_nn) = state_nn_clone.lock() {
                state_nn.progress.current_epoch = epoch;
                state_nn.progress.max_epoch = 10;
            }
        }
    });

    let app_result = app.run(terminal);

    ratatui::restore();

    app_result
}
