use std::error::Error;

use tch::{ nn::ModuleT, Tensor };

pub fn print_calculate_confusion_matrix(
    model: &impl ModuleT,
    inputs: &Tensor,
    targets: &Tensor
) -> Result<(), Box<dyn Error>> {
    let (true_positives, false_positives, true_negatives, false_negatives) =
        calculate_confusion_matrix(model, inputs, targets)?;
    println!("True Positives: {}", true_positives);
    println!("False Positives: {}", false_positives);
    println!("True Negatives: {}", true_negatives);
    println!("False Negatives: {}", false_negatives);
    Ok(())
}

pub fn calculate_confusion_matrix(
    model: &impl ModuleT,
    inputs: &Tensor,
    targets: &Tensor
) -> Result<(i64, i64, i64, i64), Box<dyn Error>> {
    let predicted = model.forward_t(inputs, false);
    let predicted_labels = predicted.gt(0.5).to_kind(tch::Kind::Int);

    let true_positives = predicted_labels
        .eq_tensor(targets)
        .logical_and(&predicted_labels)
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    let false_positives = predicted_labels
        .ne_tensor(targets)
        .logical_and(&predicted_labels)
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    let true_negatives = predicted_labels
        .eq_tensor(targets)
        .logical_and(&predicted_labels.logical_not())
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    let false_negatives = predicted_labels
        .ne_tensor(targets)
        .logical_and(&predicted_labels.logical_not())
        .sum(tch::Kind::Int)
        .int64_value(&[]);

    Ok((true_positives, false_positives, true_negatives, false_negatives))
}
