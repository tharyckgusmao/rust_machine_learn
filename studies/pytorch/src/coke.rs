use std::{ collections::HashMap, env, error::Error, path::PathBuf };
use ab_glyph::{ FontRef, PxScale };
use image::{ DynamicImage, ImageBuffer, Rgba, RgbaImage };
use minifb::{ Key, Window, WindowOptions };
use polars::prelude::SerReader;
use polars::prelude::IntoLazy;
use tch::{
    data::Iter2,
    nn::{ self, Adam, ModuleT, OptimizerConfig, VarStore },
    vision::{ imagenet, resnet },
    Device,
    Kind,
    Reduction,
    Tensor,
};
use std::str::FromStr;
use crate::{ dataloader::DataLoader, imagedataset::ImageDataset };

const BATCH_SIZE: usize = 100;

pub fn coke_test() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.join("data/dataset-cokeornot/test/others.old");
    // let dataset_path = project_dir.join("data/dataset-cokeornot/test/coke/coke0003.jpeg");

    let model_path = project_dir.join("./binary-coke.ot");
    println!("Caminho do modelo: {:?}", dataset_path);
    println!("Caminho do modelo: {:?}", model_path);

    // Carregar e redimensionar a imagem
    let dataset = imagenet::load_image_and_resize224(dataset_path.clone())?;
    let image_with_batch_dim = dataset.unsqueeze(0);
    println!("Dataset carregado: {:?}", image_with_batch_dim);

    let device = Device::cuda_if_available();

    let mut vs = VarStore::new(device);
    let net = resnet::resnet18_no_final_layer(&vs.root());

    vs.load(model_path.as_path()).map_err(|op| {
        format!("Erro ao carregar o modelo: {:?}", op);
        op
    })?;

    let linear = nn::linear(vs.root(), 512, 1, Default::default());
    let net2 = nn
        ::seq()
        .add_fn(move |xs| net.forward_t(xs, false))
        .add(linear)
        .add_fn(|xs: &Tensor| xs.sigmoid());
    let predicted = net2.forward_t(&image_with_batch_dim, false);

    let predicted_labels = predicted.gt(0.5).to_kind(Kind::Int64);

    predicted_labels.print();
    let img = image::open(dataset_path)?;
    display_image_with_prediction(img, predicted_labels)?;

    Ok(())
}

pub fn coke_transfer_train() -> Result<(), Box<dyn Error>> {
    tch::manual_seed(123);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let project_dir = PathBuf::from(manifest_dir);

    let dataset_path = project_dir.join("data/dataset-cokeornot");
    let model_path = project_dir.join("data/resnet18.ot");
    println!("Caminho do dataset: {:?}", dataset_path);
    println!("Caminho do modelo: {:?}", model_path);

    // Carregar o dataset
    let dataset = imagenet::load_from_dir(dataset_path)?;
    println!("Dataset carregado: {:?}", dataset);

    let device = Device::cuda_if_available();

    let mut vs = VarStore::new(device);
    let net = resnet::resnet18_no_final_layer(&vs.root());

    // Carregar o modelo salvo
    vs.load(model_path.as_path())?;

    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    // Configurar a camada linear e o otimizador
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    let net2 = nn
        ::seq()
        .add(linear)
        .add_fn(|xs: &Tensor| xs.sigmoid());

    // Treinamento
    for epoch_idx in 1..=1000 {
        // Treinamento
        let predicted = train_images.apply(&net2);
        let loss = predicted.binary_cross_entropy_with_logits::<Tensor>(
            &dataset.train_labels,
            None,
            None,
            Reduction::Mean
        );
        sgd.backward_step(&loss);

        let predicted_labels = predicted.ge(0.5).to_kind(tch::Kind::Int64);
        let correct_predictions = predicted_labels
            .eq_tensor(&dataset.train_labels)
            .sum(tch::Kind::Int64);
        let accuracy =
            (100.0 * correct_predictions.double_value(&[])) /
            (dataset.train_labels.size()[0] as f64);

        println!(
            "Epoch {} - Loss: {:.4} - Accuracy: {:.2}%",
            epoch_idx,
            loss.double_value(&[]),
            accuracy
        );

        let test_accuracy = test_images
            .apply(&net2)
            .binary_cross_entropy_with_logits::<Tensor>(
                &dataset.test_labels,
                None,
                None,
                Reduction::Mean
            )
            .double_value(&[]);

        println!("Test Accuracy {}: {:.2}%", epoch_idx, 100.0 * test_accuracy);
    }

    // Salvar o modelo treinado
    let save_model_path = project_dir.join("data/binary-coke.ot");
    vs.save(save_model_path)?;

    Ok(())
}

fn display_image_with_prediction(
    img: DynamicImage,
    prediction: Tensor
) -> Result<(), Box<dyn Error>> {
    let label = if prediction.int64_value(&[]) == 1 { "Coke" } else { "Not Coke" };

    // Ajuste o caminho se necessário
    let font_data = include_bytes!(".././Roboto-Regular.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    let img_rgba = img.to_rgba8();
    let (width, height) = img_rgba.dimensions();

    // Criar o buffer `u32` a partir da imagem RGBA
    let buffer: Vec<u32> = img_rgba
        .pixels()
        .map(|pixel| {
            let Rgba([r, g, b, a]) = pixel;
            // Corrigir a conversão de &u8 para u32
            let r = *r as u32;
            let g = *g as u32;
            let b = *b as u32;
            let a = *a as u32;
            (r << 24) | (g << 16) | (b << 8) | a
        })
        .collect();

    let mut window = Window::new(
        "Image with Prediction",
        width as usize,
        height as usize,
        WindowOptions::default()
    )?;

    // Atualizar a janela com o buffer
    window.update_with_buffer(&buffer, width as usize, height as usize)?;

    // Adicionar texto à imagem (exemplo simplificado)
    let scale = PxScale::from(24.0);
    let text = format!("Prediction: {}", label);
    let mut imgbuf = img_rgba.clone();

    // (Adicione aqui o código para desenhar o texto na imagem usando `rusttype` ou outro método)

    // Atualizar a janela com a imagem modificada
    let img_with_text = DynamicImage::ImageRgba8(imgbuf);
    let buffer_with_text: Vec<u32> = img_with_text
        .to_rgba8()
        .pixels()
        .map(|pixel| {
            let Rgba([r, g, b, a]) = pixel;
            // Corrigir a conversão de &u8 para u32
            let r = *r as u32;
            let g = *g as u32;
            let b = *b as u32;
            let a = *a as u32;
            (r << 24) | (g << 16) | (b << 8) | a
        })
        .collect();
    window.update_with_buffer(&buffer_with_text, width as usize, height as usize)?;

    Ok(())
}
