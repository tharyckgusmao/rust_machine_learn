use std::path::PathBuf;

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
    record::{ FullPrecisionSettings, NamedMpkFileRecorder, Record },
    tensor::activation::sigmoid,
    train::ClassificationOutput,
};
use burn_import::pytorch::{ LoadArgs, PyTorchFileRecorder };
use burn::record::Recorder;
use nn::{ loss::{ BinaryCrossEntropyLossConfig, CrossEntropyLoss }, Sigmoid };
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
    conv9: Conv2d<B>,
    conv10: Conv2d<B>,
    conv11: Conv2d<B>,
    conv12: Conv2d<B>,
    conv13: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    pool: MaxPool2d,
    dropout: Dropout,
    activation: Relu,
    activation_sig: Sigmoid,
}

impl<B: Backend> Vgg16<B> {
    /// Inicializa o modelo Vgg16.
    pub fn new(n_classes: usize, device: &B::Device) -> Self {
        // Camadas convolucionais
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv8 = Conv2dConfig::new([256, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv9 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv10 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv11 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv12 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();
        let conv13 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
            .no_grad();

        // Camadas totalmente conectadas
        let fc1 = LinearConfig::new(512 * 7 * 7, 4096)
            .init(device)
            .no_grad();
        let fc2 = LinearConfig::new(4096, 4096).init(device).no_grad();
        let fc3 = LinearConfig::new(4096, n_classes).init(device).no_grad();
        // Pooling e dropout
        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let dropout = DropoutConfig::new(0.5).init();

        Self {
            activation: Relu::new(),
            activation_sig: Sigmoid::new(),
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9,
            conv10,
            conv11,
            conv12,
            conv13,
            fc1,
            fc2,
            fc3,
            pool,
            dropout,
        }
    }

    /// Implementação do forward pass do modelo VGG.
    pub fn forward(&self, x: Tensor<B, 4>, with_fc: bool) -> Tensor<B, 2> {
        let x = x.permute([0, 3, 1, 2]);
        let x = self.activation.forward(self.conv1.forward(x));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv3.forward(x));
        let x = self.activation.forward(self.conv4.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv5.forward(x));
        let x = self.activation.forward(self.conv6.forward(x));
        let x = self.activation.forward(self.conv7.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv8.forward(x));
        let x = self.activation.forward(self.conv9.forward(x));
        let x = self.activation.forward(self.conv10.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv11.forward(x));
        let x = self.activation.forward(self.conv12.forward(x));
        let x = self.activation.forward(self.conv13.forward(x));
        let x = self.pool.forward(x);

        // Flatten para a camada totalmente conectada
        let x = x.flatten(1, 3);

        // Aplicação das camadas totalmente conectadas e dropout
        let x = self.dropout.forward(self.activation.forward(self.fc1.forward(x)));
        let x = self.dropout.forward(self.activation.forward(self.fc2.forward(x)));
        let x = self.fc3.forward(x);
        x
    }
    pub fn forward_without_fc(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = x.permute([0, 3, 1, 2]);
        let x = self.activation.forward(self.conv1.forward(x));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv3.forward(x));
        let x = self.activation.forward(self.conv4.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv5.forward(x));
        let x = self.activation.forward(self.conv6.forward(x));
        let x = self.activation.forward(self.conv7.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv8.forward(x));
        let x = self.activation.forward(self.conv9.forward(x));
        let x = self.activation.forward(self.conv10.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv11.forward(x));
        let x = self.activation.forward(self.conv12.forward(x));
        let x = self.activation.forward(self.conv13.forward(x));
        let x = self.pool.forward(x);

        return x;
    }
    pub fn new_with(n_classes: usize, device: &B::Device, args_path: PathBuf) -> Self {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let mut model = Self::new(n_classes, device)
            .load_file(args_path.clone(), &recorder, &device)
            .unwrap();
        model.fc1 = LinearConfig::new(512 * 7 * 7, 4096).init(device);
        model.fc2 = LinearConfig::new(4096, 4096).init(device);
        model.fc3 = LinearConfig::new(4096, n_classes).init(device);
        // println!("{:?}", model);
        // let load_args = LoadArgs::new(args_path.clone().into())
        //     .with_key_remap("features\\.0\\.weight", "conv1.weight")
        //     .with_key_remap("features\\.0\\.bias", "conv1.bias")
        //     .with_key_remap("features\\.2\\.weight", "conv2.weight")
        //     .with_key_remap("features\\.2\\.bias", "conv2.bias")
        //     .with_key_remap("features\\.5\\.weight", "conv3.weight")
        //     .with_key_remap("features\\.5\\.bias", "conv3.bias")
        //     .with_key_remap("features\\.7\\.weight", "conv4.weight")
        //     .with_key_remap("features\\.7\\.bias", "conv4.bias")
        //     .with_key_remap("features\\.10\\.weight", "conv5.weight")
        //     .with_key_remap("features\\.10\\.bias", "conv5.bias")
        //     .with_key_remap("features\\.12\\.weight", "conv6.weight")
        //     .with_key_remap("features\\.12\\.bias", "conv6.bias")
        //     .with_key_remap("features\\.14\\.weight", "conv7.weight")
        //     .with_key_remap("features\\.14\\.bias", "conv7.bias")
        //     .with_key_remap("features\\.17\\.weight", "conv8.weight")
        //     .with_key_remap("features\\.17\\.bias", "conv8.bias")
        //     .with_key_remap("features\\.19\\.weight", "conv9.weight")
        //     .with_key_remap("features\\.19\\.bias", "conv9.bias")
        //     .with_key_remap("features\\.21\\.weight", "conv10.weight")
        //     .with_key_remap("features\\.21\\.bias", "conv10.bias")
        //     .with_key_remap("features\\.24\\.weight", "conv11.weight")
        //     .with_key_remap("features\\.24\\.bias", "conv11.bias")
        //     .with_key_remap("features\\.26\\.weight", "conv12.weight")
        //     .with_key_remap("features\\.26\\.bias", "conv12.bias")
        //     .with_key_remap("features\\.28\\.weight", "conv13.weight")
        //     .with_key_remap("features\\.28\\.bias", "conv13.bias")
        //     // Map fully connected layers
        //     .with_key_remap("classifier\\.0\\.weight", "fc1.weight")
        //     .with_key_remap("classifier\\.0\\.bias", "fc1.bias")
        //     .with_key_remap("classifier\\.3\\.weight", "fc2.weight")
        //     .with_key_remap("classifier\\.3\\.bias", "fc2.bias");
        // //     .with_key_remap("classifier\\.6\\.weight", "fc3.weight")
        // //     .with_key_remap("classifier\\.6\\.bias", "fc3.bias");

        // let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
        // let weights = recorder.load::<Vgg16Record<B>>(load_args, &device).unwrap();

        // let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        // recorder.record(weights, args_path.into()).expect("Failed to save model record");

        return model;
    }
    pub fn new_with_pretrained(n_classes: usize, device: &B::Device, args_path: PathBuf) -> Self {
        // let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let record: Vgg16Record<B> = NamedMpkFileRecorder::<FullPrecisionSettings>
            ::new()
            .load(args_path.into(), device)
            .unwrap();

        let mut model = Self::new(n_classes, device).load_record(record);

        return model;
    }
    pub fn forward_step(
        &self,
        labels: Tensor<B, 1, Int>,
        inputs: Tensor<B, 4>,
        device: &B::Device
    ) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>, f32) {
        let output = self.forward(inputs.clone(), true);

        let accuracy = Self::accuracy(output.clone(), labels.clone());

        let loss = CrossEntropyLoss::new(None, &output.device()).forward(
            output.clone(),
            labels.clone()
        );

        (loss, output, labels, accuracy)
    }
    pub fn forward_step_sigmoid(
        &self,
        labels: Tensor<B, 1, Int>,
        inputs: Tensor<B, 4>,
        device: &B::Device
    ) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>, f32) {
        let output = self.forward(inputs.clone(), true);

        let accuracy = Self::accuracy(output.clone(), labels.clone());

        let loss_func = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(None)
            .with_logits(true)
            .init(device);

        let labels_2d = labels.clone().unsqueeze_dim(1);

        let loss = loss_func.forward(output.clone(), labels_2d.clone());
        (loss, output, labels, accuracy)
    }

    pub fn accuracy_binary(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
        let output = sigmoid(output);

        let predictions: Tensor<B, 1, Int> = output.greater_elem(0.5).int().squeeze(1);

        let num_predictions: usize = targets.dims().iter().product();
        let num_corrects = predictions.equal(targets).int().sum().into_scalar();

        (num_corrects.elem::<f32>() / (num_predictions as f32)) * 100.0
    }
    pub fn accuracy(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
        // Obtem os índices das previsões mais prováveis
        let predictions = output.argmax(1).squeeze(1);

        // Calcula o número total de previsões
        let num_predictions = targets.shape().dims[0] as f32;

        // Compara as previsões com os valores reais e conta o número de acertos
        let num_corrects = predictions.equal(targets).int().sum().into_scalar();

        // Calcula a acurácia como uma porcentagem
        (num_corrects.elem::<f32>() / num_predictions) * 100.0
    }
}
