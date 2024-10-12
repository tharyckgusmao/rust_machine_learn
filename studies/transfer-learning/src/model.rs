use tch::{ nn::{ self, Conv2D, Module, SequentialT }, Tensor };

pub struct Vgg16 {
    features: SequentialT,
    classifier: SequentialT,
}

impl Vgg16 {
    pub fn new(n_classes: i64, vs: &nn::Path) -> Self {
        let features = Self::make_layers(vs);
        let classifier = nn
            ::seq()
            .add(nn::linear(vs / "classifier" / "0", 512 * 7 * 7, 4096, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(|xs, train| xs.dropout(0.5, train))
            .add(nn::linear(vs / "classifier" / "3", 4096, 4096, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(|xs, train| xs.dropout(0.5, train))
            .add(nn::linear(vs / "classifier" / "6", 4096, n_classes, Default::default()));

        Vgg16 { features, classifier }
    }

    fn make_layers(vs: &nn::Path) -> SequentialT {
        let mut seq = nn::seq_t();
        let layers = vec![
            (3, 64),
            (64, 64),
            (64, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 512),
            (512, 512)
        ];

        let mut in_channels = 3;

        for (c_in, c_out) in layers {
            seq = seq.add(
                nn::conv2d(
                    vs / &format!("conv_{}_{}", c_in, c_out),
                    c_in,
                    c_out,
                    3,
                    Default::default()
                )
            );
            seq = seq.add_fn(|xs| xs.relu());
            seq = seq.add_fn(|xs| xs.max_pool2d_default(2));
            in_channels = c_out;
        }

        seq.add_fn(|xs| xs.flat_view());
        seq
    }
}

impl nn::Module for Vgg16 {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.features.forward(xs);
        self.classifier.forward(&xs)
    }
}

fn main() {
    let vs = nn::Path::new();
    let model = Vgg16::new(10, &vs); // Substitua 10 pelo número de classes desejado

    // Exemplo de uso
    let input_tensor = Tensor::randn(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu)); // Exemplo de entrada
    let output = model.forward(&input_tensor);

    println!("{:?}", output.size()); // Imprime o tamanho da saída
}
