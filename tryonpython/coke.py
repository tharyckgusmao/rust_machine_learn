import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

BATCH_SIZE = 100

def build_resnet18_multiclass(num_classes):
    resnet18 = models.resnet18(pretrained=True)
    layers = list(resnet18.children())[:-2]  # Remove as últimas camadas fc e avgpool
    resnet18 = nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)  # Altere para o número de classes
    )
    return resnet18

def plot_images(images, predictions, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(transforms.ToPILImage()(images[i]))  # Converte o tensor da imagem para PIL e plota
        true_label = labels[i].item()
        predicted_label = predictions[i].item()
        ax.set_title(f"True: {true_label}, Pred: {predicted_label}")
        ax.axis('off')
    plt.show()

def coke_transfer_train():
    torch.manual_seed(123)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = os.path.abspath(os.path.join(project_dir, "../studies/pytorch/data/dataset-cokeornot/train"))

    print(f"Caminho do dataset: {dataset_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(dataset.classes)
    net = build_resnet18_multiclass(num_classes)
    net = net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  # Usa CrossEntropyLoss para multiclasse

    for epoch in range(1, 5):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)  # Obtém as previsões de classe
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch {epoch} - Loss: {running_loss:.4f} - Accuracy: {accuracy:.2f}%")

    save_model_path = os.path.join(project_dir, "./model.pt")
    torch.save(net.state_dict(), save_model_path)

def coke_test():
    torch.manual_seed(123)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = os.path.abspath(os.path.join(project_dir, "../studies/pytorch/data/dataset-cokeornot/mixing"))
    model_path = os.path.join(project_dir, "./model.pt")

    print(f"Caminho do dataset: {dataset_path}")
    print(f"Caminho do modelo: {model_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)
    net = build_resnet18_multiclass(num_classes)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    total_correct = 0
    total_samples = 0

    images_list = []
    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted_labels = torch.max(outputs, 1)

            if len(images_list) == 0:  # Adiciona apenas um batch para visualização
                images_list = images.cpu()
                predictions_list = predicted_labels.cpu()
                labels_list = labels.cpu()

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    print(f"Accuracy: {accuracy:.2f}%")

    plot_images(images_list, predictions_list, labels_list)

def main():
    if len(sys.argv) != 2:
        print("Uso: python script.py [train|test]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "train":
        coke_transfer_train()
    elif mode == "test":
        coke_test()
    else:
        print("Argumento inválido. Use 'train' para treinar ou 'test' para testar o modelo.")
        sys.exit(1)

if __name__ == "__main__":
    main()
