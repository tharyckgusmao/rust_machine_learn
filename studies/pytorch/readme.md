binary_test
binary_train
mnist_test
mnist_train
iris_test
iris_train
catdog_train
catdog_test

command

LIBTORCH_BYPASS_VERSION_CHECK=true cargo run --bin pytorch binary_train


LIBTORCH_BYPASS_VERSION_CHECK=true cargo run --bin pytorch bee_train '/home/tharyckgusmaometzker/Documentos/projetos/rust_machine_learn/studies/pytorch/data/resnet18.ot' '/home/tharyckgusmaometzker/Documentos/projetos/rust_machine_learn/studies/pytorch/data/hymenoptera_data'