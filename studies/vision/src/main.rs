extern crate opencv;

use image::{ GenericImageView, ImageBuffer, Luma, RgbImage };
use ndarray::{ s, Array2, Array3, ArrayView, ArrayViewMut, Axis };

use std::{ env, fs, path::PathBuf };

use opencv::{
    core::{ FileNode, FileStorage, Point, Rect, Scalar, Size, CV_8UC1 },
    face::LBPHFaceRecognizer,
    highgui,
    imgcodecs,
    imgproc::{ self, resize },
    objdetect::CascadeClassifier,
    prelude::*,
    tracking,
    types::{ self, VectorOfMat, VectorOfi32 },
    videoio,
    Error,
    Result,
};
use regex::Regex;

fn using_haarcascade() -> Result<(), Error> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);
    let image_path = project_dir.clone().join("sample.png");
    // Carregar a imagem de um arquivo
    let mut img: Mat = imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;

    // Converter a imagem para escala de cinza
    let mut grayscale_image = Mat::default();
    imgproc::cvt_color(&img, &mut grayscale_image, imgproc::COLOR_BGR2GRAY, 0)?;

    // Carregar o classificador Haar para detecção de faces
    let classifier_path = project_dir.clone().join("haarcascade_frontalface_alt.xml");

    let mut classifier = CascadeClassifier::new(classifier_path.to_str().unwrap())?;

    // Vetor para armazenar as faces detectadas
    let mut faces = types::VectorOfRect::new();

    // Detectar faces
    classifier.detect_multi_scale(
        &grayscale_image,
        &mut faces,
        1.4,
        3,
        0,
        Size::new(30, 30),
        Size::new(0, 0)
    )?;

    // Desenhar retângulos ao redor das faces detectadas
    for face in faces.iter() {
        imgproc::rectangle(
            &mut img,
            face,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0
        )?;
    }
    let mut resize_img = Mat::default();
    let width = 900;
    let height = (((img.rows() as f64) / (img.cols() as f64)) * (width as f64)) as i32;
    let size = Size::new(width, height);
    imgproc::resize(&img, &mut resize_img, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Exibir a imagem com as faces detectadas em uma janela
    highgui::named_window("Exibição da Imagem", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("Exibição da Imagem", &resize_img)?;

    loop {
        let key = highgui::wait_key(10)?;
        if key == ('q' as i32) {
            break;
        }
    }
    Ok(())
}

fn image_to_ndarray(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Array2<u8> {
    let (width, height) = img.dimensions();
    let mut array = Array2::<u8>::zeros((height as usize, width as usize));

    for (x, y, pixel) in img.enumerate_pixels() {
        array[(y as usize, x as usize)] = pixel[0];
    }

    array
}

fn image_to_mat(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> opencv::Result<Mat> {
    let (width, height) = img.dimensions();
    let mut mat = (unsafe { Mat::new_rows_cols(height as i32, width as i32, CV_8UC1) })?;
    let mat_data = unsafe { std::slice::from_raw_parts_mut(mat.data_mut(), mat.total() as usize) };
    mat_data.copy_from_slice(img.as_raw());
    Ok(mat)
}

fn train_face_yale_hist() -> Result<(), Error> {
    let re = Regex::new(r"subject(\d+)\.sleepy\.gif").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);
    let dir_path = project_dir.clone().join("yalefaces/treinamento");

    let mut ids = VectorOfi32::new();
    let mut images = VectorOfMat::new();

    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                if let Some(file_name) = entry.file_name().to_str() {
                    if let Some(captures) = re.captures(file_name) {
                        if let Some(number) = captures.get(1) {
                            let img = image::open(entry.path()).unwrap().to_luma8();
                            let bin_img = image_to_mat(&img)?;

                            ids.push(number.as_str().parse::<i32>().unwrap());
                            images.push(bin_img);
                        }
                    }
                }
            }
        }
    }

    println!("IDs: {:?}", ids);

    let path_classifier = project_dir.clone().join("yalefaces/classifierLBPH.yml");
    let mut file_storage_opencv = FileStorage::new_def(
        path_classifier.to_str().unwrap(),
        1
    ).unwrap();
    let min_neighbors = 3;
    let grid_x = 8;
    let grid_y = 8;
    let threshold = 40.0;
    let mut classifier = LBPHFaceRecognizer::create(1, 8, 8, 8, f64::MAX)?;
    classifier.train(&images, &ids)?;
    classifier.write_1(&mut file_storage_opencv)?;
    println!("Classifier trained and saved to {:?}", path_classifier);

    Ok(())
}

fn test_yale_hist() -> opencv::Result<()> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let classifier_path = project_dir.clone().join("yalefaces/classifierLBPH.yml");

    let mut classifier = LBPHFaceRecognizer::create_def()?;
    let classifier_path_str = classifier_path.to_str().unwrap();

    let file_storage = FileStorage::new_def(classifier_path_str, 0)?;

    let file_node = file_storage.get_first_top_level_node()?;

    classifier.read_1(&file_node)?;
    println!("Classifier loaded successfully");

    let image_path = project_dir.clone().join("yalefaces/teste/subject09.sad.gif");
    let img = image::open(image_path).unwrap().to_luma8();

    let bin_img = image_to_mat(&img)?;
    println!("bin_img {:?}", bin_img);

    let predict = classifier.predict_label(&bin_img)?;
    println!("predict {:?}", predict);

    let mut resize_img = Mat::default();
    let width = 900;
    let height = (((bin_img.rows() as f64) / (bin_img.cols() as f64)) * (width as f64)) as i32;
    let size = Size::new(width, height);
    imgproc::resize(&bin_img, &mut resize_img, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let position = Point::new(10, 30); // Ajuste a posição conforme necessário

    let predict_text = format!("Predict: {:?}", predict);
    // Escolha da fonte
    let font = imgproc::FONT_HERSHEY_SIMPLEX;

    // Tamanho da fonte
    let font_scale = 1.0;

    // Cor do texto (BGR)
    let color = Scalar::new(0.0, 0.0, 0.0, 0.0); // Branco

    // Espessura do texto
    let thickness = 2;

    // Adicionando o texto à imagem
    imgproc::put_text(
        &mut resize_img,
        &predict_text,
        position,
        font,
        font_scale,
        color,
        thickness,
        imgproc::LINE_AA,
        false
    )?;

    highgui::named_window("Exibição da Imagem", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("Exibição da Imagem", &resize_img)?;

    loop {
        let key = highgui::wait_key(10)?;
        if key == ('q' as i32) {
            break;
        }
    }
    Ok(())
}

fn static_csrt() -> opencv::Result<()> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let classifier_path = project_dir.clone().join("./city.mp4");
    let mut video = videoio::VideoCapture::from_file(
        classifier_path.to_str().unwrap(),
        videoio::CAP_ANY
    )?;
    if !video.is_opened()? {
        panic!("Failed to open the video file.");
    }

    // Lê o primeiro frame do vídeo
    let mut frame = Mat::default();
    video.read(&mut frame)?;

    // Permite ao usuário selecionar a ROI manualmente
    let mut roi = highgui::select_roi(&frame, false, false)?;
    highgui::wait_key(0)?;

    // Verifica se a ROI foi selecionada
    if roi.width > 0 && roi.height > 0 {
        // Cria o rastreador CSRT
        let mut tracker = tracking::TrackerCSRT::create_def()?;

        // Inicializa o rastreador com o primeiro frame e a ROI
        tracker.init(&frame, roi)?;

        // Loop para ler frames do vídeo e rastrear o objeto
        while video.read(&mut frame)? {
            // Atualiza o rastreador com o novo frame
            let ok = tracker.update(&frame, &mut roi)?;

            if ok {
                // Se o rastreamento foi bem-sucedido, desenha a nova ROI
                imgproc::rectangle(
                    &mut frame,
                    roi,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0
                )?;
            } else {
                // Se o rastreamento falhou, exibe uma mensagem
                imgproc::put_text(
                    &mut frame,
                    "Tracking failure detected",
                    Point::new(100, 80),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    false
                )?;
            }

            // Exibe o frame com a ROI desenhada ou mensagem de falha
            highgui::imshow("Tracking", &frame)?;

            // Sai do loop se a tecla 'q' for pressionada
            let key = highgui::wait_key(30)?;
            if key == ('q' as i32) {
                break;
            }
        }
    } else {
        println!("Nenhuma ROI selecionada.");
    }
    Ok(())
}

fn detect_object(frame: &Mat) -> Result<Rect, Error> {
    // Converter o frame para escala de cinza
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Aplicar um desfoque gaussiano para reduzir o ruído
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(&gray, &mut blurred, Size::new(21, 21), 0.0, 0.0, 1)?;

    let mut thresh = Mat::default();
    imgproc::threshold(&blurred, &mut thresh, 200.0, 255.0, imgproc::THRESH_BINARY)?;

    // Encontrar contornos no frame binarizado
    let mut contours = opencv::types::VectorOfVectorOfPoint::new();
    imgproc::find_contours(
        &thresh,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    // Se encontrar contornos, retorna o bounding box do maior contorno
    if contours.len() > 0 {
        let largest_contour = contours.get(0).unwrap();
        let bounding_rect = imgproc::bounding_rect(&largest_contour)?;
        return Ok(bounding_rect);
    }

    Err(Error::new(1, ""))
}

fn main() -> opencv::Result<()> {
    // using_haarcascade();
    // train face
    // train_face_yale_hist();
    // test_yale_hist();

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(manifest_dir);

    let classifier_path = project_dir.clone().join("./city.mp4");
    let mut video = videoio::VideoCapture::from_file(
        classifier_path.to_str().unwrap(),
        videoio::CAP_ANY
    )?;
    if !video.is_opened()? {
        panic!("Failed to open the video file.");
    }

    // Cria uma janela para exibir os frames
    highgui::named_window("Tracking", highgui::WINDOW_AUTOSIZE)?;

    // Variáveis para rastreamento
    let mut tracker = tracking::TrackerCSRT::create_def()?;
    let mut tracking = false;
    let mut roi = Rect::default();

    // Loop para ler frames do vídeo e rastrear o objeto
    let mut frame = Mat::default();
    while video.read(&mut frame)? {
        if tracking {
            // Atualiza o rastreador com o novo frame
            let ok = tracker.update(&frame, &mut roi)?;
            if ok {
                //  desenha a nova ROI
                imgproc::rectangle(
                    &mut frame,
                    roi,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0
                )?;
            } else {
                //  desativa o rastreamento
                tracking = false;
            }
        } else {
            // Detecta o objeto no frame atual
            if let Ok(new_roi) = detect_object(&frame) {
                // Inicializa o rastreador com o frame atual e a ROI detectada
                tracker = tracking::TrackerCSRT::create_def()?;
                tracker.init(&frame, new_roi)?;
                roi = new_roi;
                tracking = true;
            }
        }

        // Exibe o frame com a ROI desenhada ou mensagem de falha
        highgui::imshow("Tracking", &frame)?;

        // Sai do loop se a tecla 'q' for pressionada
        let key = highgui::wait_key(30)?;
        if key == ('q' as i32) {
            break;
        }
    }

    Ok(())
}
