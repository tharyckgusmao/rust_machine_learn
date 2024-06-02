import torch

model_path = "/home/tharyckgusmaometzker/Documentos/projetos/rust_machine_learn/studies/nlp/model.ot"
model = torch.load(model_path)

# Verifique os tensores no modelo
for name, param in model.named_parameters():
    print(name)

# Verifique se 'classifier.bias' está presente
if 'classifier.bias' not in [name for name, _ in model.named_parameters()]:
    print("classifier.bias tensor is missing")