
use ndarray::prelude::*;

fn main() {
    println!("Exploring ndArray");
let a = array![
                [2.,1.,3.], 
                [4.,5.,6.],
            ]; 
                println!("{:?} Shape", a.shape());
                println!("{:?} Len", a.len());
                println!("{:?} Dim", a.ndim());
                println!("{:?}", a);
}
