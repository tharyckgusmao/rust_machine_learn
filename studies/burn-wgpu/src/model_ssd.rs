use std::path::PathBuf;

use burn::{
    nn::conv::{ Conv2d, Conv2dConfig },
    prelude::*,
    record::{ FullPrecisionSettings, NamedMpkFileRecorder, Recorder },
};
use nn::{ Linear, LinearConfig, PaddingConfig2d };

use crate::model::Vgg16;

#[derive(Module, Debug)]
pub struct Rcnn<B: Backend> {
    backbone: Vgg16<B>,
    rpn_conv: Conv2d<B>,
    rpn_cls: Linear<B>,
    rpn_reg: Linear<B>,
    cls_score: Linear<B>,
    bbox_pred: Linear<B>,
}

impl<B: Backend> Rcnn<B> {
    pub fn new(num_classes: usize, device: &B::Device, args_path: Option<PathBuf>) -> Self {
        let mut backbone = if let Some(path) = args_path {
            Vgg16::new_with_pretrained(num_classes, device, path)
        } else {
            Vgg16::new(num_classes, device)
        };

        // Camadas para a Proposta de Regiões (RPN)
        let rpn_conv = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let rpn_cls = LinearConfig::new(512 * 7 * 7, 18) // 2 classes para RPN (foreground/background)
            .init(device);
        let rpn_reg = LinearConfig::new(512 * 7 * 7, 36) // 4 valores de bbox por região proposta
            .init(device);

        // Camadas para classificação final
        let cls_score = LinearConfig::new(512 * 7 * 7, num_classes).init(device);
        let bbox_pred = LinearConfig::new(512 * 7 * 7, 1 * 4) // 4 valores de bbox por classe
            .init(device);

        Self {
            backbone,
            rpn_conv,
            rpn_cls,
            rpn_reg,
            cls_score,
            bbox_pred,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Passa pela backbone (VGG16)
        let features = self.backbone.forward_without_fc(x);

        // Proposta de Regiões
        let rpn_features = self.rpn_conv.forward(features);
        let rpn_cls_score: Tensor<B, 2> = self.rpn_cls.forward(rpn_features.clone().flatten(1, 3));
        let rpn_bbox_pred: Tensor<B, 2> = self.rpn_reg.forward(rpn_features.clone().flatten(1, 3));

        // Classificação final
        let cls_score = self.cls_score.forward(rpn_features.clone().flatten(1, 3));
        let bbox_pred: Tensor<B, 2> = self.bbox_pred.forward(rpn_features.clone().flatten(1, 3));

        (bbox_pred, cls_score)
    }
    pub fn new_with_pretrained(n_classes: usize, device: &B::Device, args_path: PathBuf) -> Self {
        // let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let record: RcnnRecord<B> = NamedMpkFileRecorder::<FullPrecisionSettings>
            ::new()
            .load(args_path.into(), device)
            .unwrap();

        let mut model = Self::new(n_classes, device, None).load_record(record);

        return model;
    }
}
