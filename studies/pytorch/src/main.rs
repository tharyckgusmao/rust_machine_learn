pub mod utils;
pub mod binary;
pub mod iris;
pub mod dataframe;
pub mod iris2;
pub mod mnist;
pub mod dogandcat;
pub mod imagedataset;
pub mod dataloader;
pub mod coke;
extern crate csv;
extern crate tch;
use std::{ env, error::Error, ops::Sub, path::PathBuf };
use binary::{ binary_test, binary_train };
use coke::{ coke_test, coke_train };
use dogandcat::{ dog_test, dog_train };
use iris::iris_train;
use iris2::iris_train2;
use mnist::{ mnist_test, mnist_train };

use tch::{
    nn::{ self, adam, Adam, ModuleT, Optimizer, OptimizerConfig, VarStore },
    Reduction,
    Tensor,
};
use std::str::FromStr;

const BATCH_SIZE: usize = 100;
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model = if args.len() < 2 { None } else { Some(args[1].as_str()) };
    match model {
        Some("binary_train") => binary_train()?,
        Some("binary_test") => binary_test()?,
        Some("iris_train") => iris_train()?,
        Some("iris_train2") => iris_train2()?,
        Some("mnist_train") => mnist_train()?,
        Some("mnist_test") => mnist_test()?,
        Some("catdog_train") => dog_train()?,
        Some("catdog_test") => dog_test()?,
        Some("coke_train") => coke_train()?,
        Some("coke_test") => coke_test()?,

        _ => println!("Invalid argument"),
    }

    Ok(())
}
