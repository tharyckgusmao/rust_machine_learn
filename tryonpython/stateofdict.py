import torch
import torchvision.models as models

# Carregando o modelo VGG16 pré-treinado
model = models.vgg16(pretrained=True)

# Supondo que você queira carregar pesos de um arquivo específico
# model.load_state_dict(torch.load('caminho/para/seus/pesos.pth'))

# Imprimindo o state_dict
print("State Dict:")
for name, param in model.state_dict().items():
    print(f"{name}: {param.size()}")
