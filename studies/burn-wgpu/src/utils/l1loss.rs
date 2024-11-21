use burn::{ prelude::Backend, tensor::Tensor };

pub struct SmoothL1Loss;

impl SmoothL1Loss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(
        &self,
        predictions: &Tensor<impl Backend, 2>,
        targets: &Tensor<impl Backend, 2>
    ) -> Tensor<impl Backend, 1> {
        let diff = predictions.sub(targets);
        let abs_diff = diff.abs();

        // Aplicando a Smooth L1 Loss
        let loss_part1 = 0.5 * diff.mul_scalar(diff); // diff^2 / 2 para |x| < 1
        let loss_part2 = abs_diff.sub_scalar(0.5); // |x| - 0.5 para |x| >= 1

        // Combina as duas partes usando a condição
        let loss = abs_diff.lower(1.0).mask_where(loss_part1, loss_part2);

        // Calcula a média do loss
        loss.mean()
    }

    pub fn backward(
        &self,
        predictions: &Tensor<impl Backend, 2>,
        targets: &Tensor<impl Backend, 2>
    ) -> Tensor<impl Backend, 2> {
        let diff = predictions.sub(targets);
        let abs_diff = diff.abs();

        // Calculando o gradiente da Smooth L1 Loss
        let grad_part1 = diff; // Para |x| < 1
        let grad_part2 = diff.sign(); // Para |x| >= 1

        // Combina grad_part1 e grad_part2 usando mask_where
        let grad = abs_diff.lower(1.0).mask_where(grad_part1, grad_part2);

        // Normaliza pelo número de elementos
        grad.div_scalar(predictions.numel())
    }
}
