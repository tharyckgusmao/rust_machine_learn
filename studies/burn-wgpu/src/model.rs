use burn::{
    nn::{
        conv::{ Conv2d, Conv2dConfig },
        pool::{ MaxPool2d, MaxPool2dConfig },
        Dropout,
        DropoutConfig,
        Linear,
        LinearConfig,
        PaddingConfig2d,
        Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Vgg16<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    conv7: Conv2d<B>,
    conv8: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    pool: MaxPool2d,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> Vgg16<B> {
    /// Inicializa o modelo Vgg16.
    pub fn new(n_classes: usize, device: &B::Device) -> Self {
        // Camadas convolucionais
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv7 = Conv2dConfig::new([256, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv8 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        // Camadas totalmente conectadas
        let fc1 = LinearConfig::new(512 * 7 * 7, 4096).init(device);
        let fc2 = LinearConfig::new(4096, 4096).init(device);
        let fc3 = LinearConfig::new(4096, n_classes).init(device);

        // Pooling e dropout
        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let dropout = DropoutConfig::new(0.5).init();

        Self {
            activation: Relu::new(),
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            fc1,
            fc2,
            fc3,
            pool,
            dropout,
        }
    }

    /// Implementação do forward pass do modelo VGG.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = x.permute([0, 3, 1, 2]); // Permutando PARA nHWC (N=Batch,  C=Channels, H=Height, W=Width,) para NCHW
        println!("Input size: {:?}", x.shape().dims);

        // Aplicação das camadas convolucionais e pooling
        let x = self.conv1.forward(x);
        println!("After conv1: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        println!("After conv2: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        println!("After pool1: {:?}", x.shape().dims);

        let x = self.conv3.forward(x);
        println!("After conv3: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.conv4.forward(x);
        println!("After conv4: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        println!("After pool2: {:?}", x.shape().dims);

        let x = self.conv5.forward(x);
        println!("After conv5: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.conv6.forward(x);
        println!("After conv6: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        println!("After pool3: {:?}", x.shape().dims);

        let x = self.conv7.forward(x);
        println!("After conv7: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.conv8.forward(x);
        println!("After conv8: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        println!("After pool4: {:?}", x.shape().dims);

        // Flatten para a camada totalmente conectada
        let x = x.flatten(1, 3);
        println!("After flatten: {:?}", x.shape().dims);

        // Aplicação das camadas totalmente conectadas e dropout
        let x = self.fc1.forward(x);
        println!("After fc1: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        println!("After dropout1: {:?}", x.shape().dims);

        let x = self.fc2.forward(x);
        println!("After fc2: {:?}", x.shape().dims);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        println!("After dropout2: {:?}", x.shape().dims);

        let x = self.fc3.forward(x);
        println!("After fc3: {:?}", x.shape().dims);

        x
    }
}
