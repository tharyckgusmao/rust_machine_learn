use std::{ sync::{ mpsc, Arc, Mutex }, thread, time::Duration };

use editor::Editor;
use state::StateNN;
use ui::{ App, AppState };
use color_eyre::Result;

pub mod ui;
pub mod state;
pub mod editor;

fn example() -> Result<()> {
    color_eyre::install()?;

    let (tx, rx) = mpsc::channel();
    let state_nn = Arc::new(Mutex::new(StateNN::default()));

    let editor = Editor {
        rx,
        state_nn: Arc::clone(&state_nn),
    };

    thread::spawn(move || {
        editor.listen_and_update();
    });

    thread::spawn(move || {
        for epoch in 0..10 {
            thread::sleep(Duration::from_secs(1));
            let mut state_clone = StateNN::default();

            state_clone.progress.current_epoch = epoch;
            state_clone.progress.max_epoch = 10;

            tx.send(state_clone);
        }
    });

    let terminal = ratatui::init();
    let app = App {
        state: AppState::default(),
        state_nn: Arc::clone(&state_nn),
        scroll_position: 0,
    };

    let app_result = app.run(terminal);
    ratatui::restore();
    app_result
}
