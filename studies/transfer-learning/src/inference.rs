use tch::{nn::{self, Module, ModuleT}, vision, Device};
use crate::model::{self, vgg};

pub fn inference(image_path: &str) -> i64{
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    // let net = model::net(&vs.root(), 2,false);
         let vgg_net = vgg(&vs.root(), 2, false);

    let net =
        nn::seq_t()
            .add(vgg_net)
            .add(nn::linear(vs.root() / "fc", 4096, 2, Default::default()));
    vs.load("weights/best_model.ot").unwrap();
    println!("{:?}", vs.variables());

    let image = vision::imagenet::load_image_and_resize224(image_path).unwrap().unsqueeze(0);
    let out = net.forward_t(&image.to_device(device),false);
    let prediction = out.argmax(1, false);

    i64::try_from(prediction).unwrap()
}