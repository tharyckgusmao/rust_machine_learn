use tch::{nn::{self, Conv2D, Module, SequentialT}, Tensor};

pub fn net(vs: &nn::Path, n_class: i64, train: bool) -> impl Module {
    nn::seq()
        .add(nn::conv2d(vs, 3, 16, 16, Default::default())) // Conv2d: 3 canais de entrada, 16 de saída
        .add_fn(|xs| xs.max_pool2d_default(4)) // Max pooling
        .add_fn(|xs| xs.relu()) // ReLU
        .add_fn(move |xs: &Tensor| xs.dropout(0.2, train))
        // Dropout com probabilidade de 30%
        .add(nn::conv2d(vs, 16, 64, 4, Default::default())) // Conv2d: 16 canais de entrada, 64 de saída
        .add_fn(|xs| xs.max_pool2d_default(2)) // Max pooling
        .add_fn(|xs| xs.relu()) // ReLU
        .add_fn(move |xs: &Tensor| xs.dropout(0.2, train)) // Dropout com probabilidade de 30%
        .add(nn::conv2d(vs, 64, 128, 4, Default::default())) // Conv2d: 64 canais de entrada, 128 de saída
        .add_fn(|xs| xs.relu()) // ReLU
        .add_fn(|xs| xs.flat_view()) // Flatten
        .add(nn::linear(vs, 56448, 1024, Default::default())) // Linear camada totalmente conectada
        .add_fn(|xs| xs.relu()) // ReLU
        .add_fn(move |xs: &Tensor| xs.dropout(0.5, train)) // Dropout com probabilidade de 50%
        .add(nn::linear(vs, 1024, n_class, Default::default())) // Linear camada de saída
}

fn layers_e() -> Vec<Vec<i64>> {
    vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256, 256, 256],
        vec![512, 512, 512, 512],
        vec![512, 512, 512, 512],
    ]
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
    nn::conv2d(p, c_in, c_out, 3, conv2d_cfg)
}

pub fn  vgg(p: &nn::Path, nclasses: i64, batch_norm: bool) -> SequentialT {
        let c = p / "classifier";
        let mut seq = nn::seq_t();
        let f = p / "features";
        let mut c_in = 3;
        for channels in layers_e().into_iter() {
            for &c_out in channels.iter() {
                let l = seq.len();
                seq = seq.add(conv2d(&f / &l.to_string(), c_in, c_out));
                if batch_norm {
                    let l = seq.len();
                    seq = seq.add(nn::batch_norm2d(&f / &l.to_string(), c_out, Default::default()));
                };
                seq = seq.add_fn(|xs| xs.relu());
                c_in = c_out;
            }
            seq = seq.add_fn(|xs| xs.max_pool2d_default(2));
        }
        seq.add_fn(|xs| xs.flat_view())
            .add(nn::linear(&c / "0", 512 * 7 * 7, 4096, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(|xs, train| xs.dropout(0.5, train))
            .add(nn::linear(&c / "3", 4096, 4096, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(|xs, train| xs.dropout(0.5, train))
            
    
}