extern crate opencv;

use opencv::{
    core::{self, Scalar, Size}, 
    highgui, 
    imgcodecs, 
    imgproc::{self, resize}, 
    objdetect::{self, CascadeClassifier}, 
    prelude::*, 
    types, 
    Error, 
    Result,
};

fn main() -> Result<(), Error> {
    // Carregar a imagem de um arquivo
    let mut img: Mat = imgcodecs::imread("./sample.png", imgcodecs::IMREAD_COLOR)?;


    // Converter a imagem para escala de cinza
    let mut grayscale_image = Mat::default();
    imgproc::cvt_color(&img, &mut grayscale_image, imgproc::COLOR_BGR2GRAY, 0)?;

    // Carregar o classificador Haar para detecção de faces
    let mut classifier = CascadeClassifier::new("./haarcascade_frontalface_alt.xml")?;

    // Vetor para armazenar as faces detectadas
    let mut faces = types::VectorOfRect::new();

    // Detectar faces
    classifier.detect_multi_scale(
        &grayscale_image,
        &mut faces,
        1.1,
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
    let height = (img.rows() as f64 / img.cols() as f64 * width as f64) as i32; 
    let size = Size::new(width, height);
    imgproc::resize(&img, &mut resize_img, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Exibir a imagem com as faces detectadas em uma janela
    highgui::named_window("Exibição da Imagem", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("Exibição da Imagem", &resize_img)?;

    // Aguardar a tecla 'q' para fechar a janela
    loop {
        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
