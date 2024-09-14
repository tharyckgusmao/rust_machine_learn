#!/bin/bash

source venv/bin/activate

if [ -z "$1" ]; then
    echo "Por favor, forneça um argumento: 'train' ou 'test'."
    exit 1
fi

if [ "$1" == "train" ]; then
    python coke.py train
elif [ "$1" == "test" ]; then
    python coke.py test
else
    echo "Argumento inválido. Use 'train' ou 'test'."
    exit 1
fi

deactivate
