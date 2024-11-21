# Train a CNN in Rust


export LIBTORCH=/home/tharyckgusmaometzker/Documentos/projetos/pytorch/build/lib.linux-x86_64-3.10/torch/

export LD_LIBRARY_PATH=/home/tharyckgusmaometzker/Documentos/projetos/pytorch/build/lib:$LD_LIBRARY_PATH


cargo run --bin burn-wgpu train [--preview support preview images enter to suffle] 
cargo run --bin burn-wgpu eval 
cargo run --bin burn-wgpu camera

// wip

cargo run --bin burn-wgpu train_ssd 
cargo run --bin burn-wgpu eval_sdd
cargo run --bin burn-wgpu camera2