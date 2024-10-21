# Train a CNN in Rust


export LIBTORCH=/home/tharyckgusmaometzker/Documentos/projetos/pytorch/build/lib.linux-x86_64-3.10/torch/

export LD_LIBRARY_PATH=/home/tharyckgusmaometzker/Documentos/projetos/pytorch/build/lib:$LD_LIBRARY_PATH




cargo run --bin burn-wgpu train 
cargo run --bin burn-wgpu eval 
cargo run --bin burn-wgpu camer 