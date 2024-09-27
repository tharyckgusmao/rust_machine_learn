use ratatui::{
    backend::CrosstermBackend,
    buffer::{ Buffer },
    layout::{ self, Alignment, Constraint, Direction, Layout, Rect },
    style::{ palette::tailwind, Color, Modifier, Style },
    symbols,
    text::{ Span, Text },
    widgets::{
        block::Title,
        Axis,
        Block,
        Borders,
        Cell,
        Chart,
        Dataset,
        Gauge,
        HighlightSpacing,
        LineGauge,
        Padding,
        Paragraph,
        Row,
        Table,
        Widget,
    },
    DefaultTerminal,
    Frame,
    Terminal,
};
use std::{
    error::Error,
    f64::consts::E,
    io,
    rc::Rc,
    sync::{ mpsc::{ self, Receiver }, Arc, Mutex },
    thread::{ self, sleep },
};
use crossterm::event::{ self, Event, KeyCode, KeyEventKind };
use crossterm::terminal::{ disable_raw_mode, enable_raw_mode };
use std::time::{ Duration, Instant };
use color_eyre::Result;
use ratatui::prelude::Stylize;

const CUSTOM_LABEL_COLOR: Color = tailwind::SLATE.c200;

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
}

#[derive(Debug)]
pub struct App {
    pub state: AppState,
    pub state_nn: Arc<Mutex<StateNN>>,
    pub rx: Receiver<StateNN>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    #[default]
    Running,
    Started,
    Quitting,
}

impl App {
    pub fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        while self.state != AppState::Quitting {
            terminal.draw(|frame| frame.render_widget(&self, frame.area()))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn handle_events(&mut self) -> Result<()> {
        let timeout = Duration::from_secs_f32(1.0 / 20.0);
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => self.quit(),
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn start(&mut self) {
        self.state = AppState::Started;
    }

    fn quit(&mut self) {
        self.state = AppState::Quitting;
    }
}

impl Widget for &App {
    #[allow(clippy::similar_names)]
    fn render(self, area: Rect, buf: &mut Buffer) {
        let body = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Max(1), Constraint::Fill(2), Constraint::Max(1)].as_ref())
            .split(area);

        let container = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Length(7), Constraint::Fill(1)].as_ref())
            .split(body[1]);

        let section_progress = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Fill(1)].as_ref())
            .split(container[0]);

        let section_info = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(20), Constraint::Fill(1)].as_ref())
            .split(container[1]);

        let section_axis = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(section_info[1]);

        render_header(body[0], buf);
        render_footer(body[2], buf);
        self.render_progress(section_progress, buf);

        self.render_axisx_graphs(section_axis, buf);
        self.render_table(section_info[0], buf);
    }
}

fn render_header(area: Rect, buf: &mut Buffer) {
    Paragraph::new("Trainning Neural Network")
        .bold()
        .alignment(Alignment::Left)
        .fg(CUSTOM_LABEL_COLOR)
        .render(area, buf);
}

fn render_footer(area: Rect, buf: &mut Buffer) {
    Paragraph::new("Press ENTER to start")
        .alignment(Alignment::Center)
        .fg(CUSTOM_LABEL_COLOR)
        .bold()
        .render(area, buf);
}

fn calculate_percent(current: u16, max: u16) -> u16 {
    if max == 0 { 0 } else { (current * 100) / max }
}

impl App {
    fn render_progress(&self, area: std::rc::Rc<[Rect]>, buf: &mut Buffer) {
        let state_nn = self.state_nn.lock().unwrap();

        LineGauge::default()
            .block(Block::default().borders(Borders::ALL).title("Epochs"))
            .filled_style(Style::default().fg(Color::Cyan))
            .ratio(
                (
                    calculate_percent(
                        state_nn.progress.current_epoch,
                        state_nn.progress.max_epoch
                    ) as f64
                ) / 100.0
            )
            .render(area[0], buf);
        Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Batch Progress"))
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(
                (
                    calculate_percent(
                        state_nn.progress.current_epoch,
                        state_nn.progress.max_epoch
                    ) as f64
                ) / 100.0
            )
            .render(area[1], buf);
    }
    fn render_axisx_graphs(&self, area: Rc<[Rect]>, buf: &mut Buffer) {
        let state_nn = self.state_nn.lock().unwrap();

        self.render_axisx(
            area[0],
            buf,
            "Accuracy",
            &state_nn.train_progress.accuracy,
            &state_nn.train_progress.loss
        );
        self.render_axisx(
            area[1],
            buf,
            "Loss",
            &state_nn.val_progress.accuracy,
            &state_nn.val_progress.loss
        );
    }
    fn render_axisx(
        &self,
        area: Rect,
        buf: &mut Buffer,
        title: &str,
        accuracy: &Vec<(f64, f64)>,
        loss: &Vec<(f64, f64)>
    ) {
        let accuracy_dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(ratatui::widgets::GraphType::Line)
            .data(accuracy);

        let loss_dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Yellow))
            .graph_type(ratatui::widgets::GraphType::Line)
            .data(loss);

        Chart::new(vec![accuracy_dataset, loss_dataset])
            .block(Block::default().borders(Borders::ALL).title(title))
            .x_axis(
                Axis::default()
                    .title("Epochs")
                    .bounds([0.0, accuracy.len() as f64])
                    .style(Style::default().fg(Color::Gray))
            )
            .y_axis(
                Axis::default()
                    .title("Value")
                    .bounds([0.0, 1.0])
                    .style(Style::default().fg(Color::Gray))
            )
            .render(area, buf);
    }

    fn render_table(&self, area: Rect, buf: &mut Buffer) {
        let state_nn = self.state_nn.lock().unwrap();

        let header_style = Style::default();
        let title = title_block("Classes and Labels");

        let header = Row::new(vec![Cell::from(Text::raw("Class")), Cell::from(Text::raw("Encode"))])
            .style(header_style)
            .height(1);

        let rows = state_nn.classes.iter().map(|(class, encode)| {
            vec![Cell::from(format!("{class}")), Cell::from(format!("{encode}"))]
                .into_iter()
                .collect::<Row>()
                .height(1)
        });
        Table::new(rows, [Constraint::Length(10), Constraint::Min(1)])
            .header(header)
            .block(title)
            .highlight_spacing(HighlightSpacing::Always)
            .render(area, buf);
    }
}

fn title_block(title: &str) -> Block {
    let title = Title::from(title).alignment(Alignment::Center);
    Block::new()
        .borders(Borders::NONE)
        .padding(Padding::vertical(1))
        .title(title)
        .borders(Borders::ALL)
        .fg(CUSTOM_LABEL_COLOR)
}
