use minifb::{ Key, Window, WindowOptions };
use image::{ DynamicImage, GenericImageView, RgbImage };
use std::time::Duration;

pub fn display_results(
    images: Vec<Vec<i64>>,
    labels: Vec<i64>,
    predictions: Vec<i64>,
    window_width: usize,
    window_height: usize
) {
    let resize = 100;
    let images_per_page = 50;
    let mut buffer: Vec<u32> = vec![0; window_width * window_height];
    let mut window = Window::new(
        "Resultados de Validação",
        window_width,
        window_height,
        WindowOptions::default()
    ).unwrap_or_else(|e| panic!("{}", e));

    let mut start_index = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0);

        let images_per_row = window_width / resize;
        let images_per_col = window_height / resize;

        for i in 0..images_per_page {
            let image_index = start_index + i;
            if image_index >= images.len() {
                break;
            }

            let image_rgb = image_to_rgb(&images[image_index]);
            let resized_image = resize_image(&image_rgb, 224, 224, resize, resize);

            let x_offset = (i % images_per_row) * resize;
            let y_offset = (i / images_per_row) * resize;

            if x_offset + resize <= window_width && y_offset + resize <= window_height {
                for y in 0..resize {
                    for x in 0..resize {
                        let pixel_index = (y_offset + y) * window_width + (x_offset + x);
                        buffer[pixel_index] = resized_image[y * resize + x];
                    }
                }

                draw_number(
                    &mut buffer,
                    window_width,
                    x_offset + 10,
                    y_offset,
                    labels[image_index] as u8,
                    0xff0000
                ); // vermelho
                draw_number(
                    &mut buffer,
                    window_width,
                    x_offset + 10,
                    y_offset + 10,
                    predictions[image_index] as u8,
                    0x00ff00
                ); // verde
            } else {
                println!("Imagem {} não cabe na janela", i + 1);
            }
        }

        window.update_with_buffer(&buffer, window_width, window_height).unwrap();

        if window.is_key_down(Key::Enter) {
            start_index += images_per_page;
            if start_index >= images.len() {
                start_index = 0;
            }
            std::thread::sleep(Duration::from_millis(300));
        }

        std::thread::sleep(Duration::from_millis(100));
    }
}

fn draw_number(
    buffer: &mut Vec<u32>,
    window_width: usize,
    x_offset: usize,
    y_offset: usize,
    number: u8,
    color: u32
) {
    let digit_patterns = [
        &[0b111, 0b101, 0b101, 0b101, 0b111], // 0
        &[0b010, 0b110, 0b010, 0b010, 0b111], // 1
        &[0b111, 0b001, 0b111, 0b100, 0b111], // 2
        &[0b111, 0b001, 0b111, 0b001, 0b111], // 3
        &[0b101, 0b101, 0b111, 0b001, 0b001], // 4
        &[0b111, 0b100, 0b111, 0b001, 0b111], // 5
        &[0b111, 0b100, 0b111, 0b101, 0b111], // 6
        &[0b111, 0b001, 0b001, 0b001, 0b001], // 7
        &[0b111, 0b101, 0b111, 0b101, 0b111], // 8
        &[0b111, 0b101, 0b111, 0b001, 0b111], // 9
    ];

    if let Some(pattern) = digit_patterns.get(number as usize) {
        for (y, row) in pattern.iter().enumerate() {
            for x in 0..3 {
                if ((row >> (2 - x)) & 1) == 1 {
                    let pixel_index = (y_offset + y) * window_width + (x_offset + x);
                    if pixel_index < buffer.len() {
                        buffer[pixel_index] = color;
                    }
                }
            }
        }
    }
}

fn draw_label(
    buffer: &mut Vec<u32>,
    window_width: usize,
    x_offset: usize,
    y_offset: usize,
    label: i64,
    prediction: i64
) {
    let color_label = 0xff0000; // Vermelho para o rótulo
    let color_prediction = 0x00ff00; // Verde para a previsão

    for y in 0..10 {
        for x in 0..30 {
            let index_label = (y_offset + y) * window_width + (x_offset + x);
            let index_pred = (y_offset + y) * window_width + (x_offset + x + 32);

            if index_label < buffer.len() {
                buffer[index_label] = color_label;
            }
            if index_pred < buffer.len() {
                buffer[index_pred] = color_prediction;
            }
        }
    }
}

fn resize_image(
    image: &Vec<u8>,
    original_width: usize,
    original_height: usize,
    new_width: usize,
    new_height: usize
) -> Vec<u32> {
    let image_rgb = RgbImage::from_vec(
        original_width as u32,
        original_height as u32,
        image.clone()
    ).unwrap();

    let img = DynamicImage::ImageRgb8(image_rgb);

    let resized_img = img.resize(
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::Triangle
    );

    let resized_pixels = resized_img.to_rgb8().into_raw();

    let mut resized_image = Vec::with_capacity(new_width * new_height);
    for chunk in resized_pixels.chunks(3) {
        let r = chunk[0] as u32;
        let g = chunk[1] as u32;
        let b = chunk[2] as u32;

        // Monta o pixel como `0xRRGGBB`
        let pixel = (r << 16) | (g << 8) | b;
        resized_image.push(pixel);
    }
    return resized_image;
}

fn image_to_rgb(image: &Vec<i64>) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(image.len());

    for chunk in image.chunks(3) {
        let r = chunk[0] as u8;
        let g = chunk[1] as u8;
        let b = chunk[2] as u8;

        buffer.push(r);
        buffer.push(g);
        buffer.push(b);
    }

    buffer
}

pub fn confusion_matrix(
    predictions: Vec<i64>,
    labels: Vec<i64>,
    num_classes: i64
) -> Vec<Vec<i64>> {
    let mut matrix = vec![vec![0; num_classes as usize]; num_classes as usize];

    for (pred, label) in predictions.iter().zip(labels.iter()) {
        matrix[*label as usize][*pred as usize] += 1;
    }

    matrix
}

pub fn print_confusion_matrix(matrix: Vec<Vec<i64>>) {
    println!("Matriz de Confusão:");
    for row in matrix {
        println!("{:?}", row);
    }
}
