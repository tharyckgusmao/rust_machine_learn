import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys

BATCH_SIZE = 100

def mnist_test():
    torch.manual_seed(123)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = build_model()
    model.load_state_dict(torch.load(os.path.join(project_dir, "binary-mnist.pth")))
    model.eval()
    
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        output = model(images)
        predicted_softmax = torch.softmax(output, dim=-1)
        predicted_classes = torch.argmax(predicted_softmax, dim=1)
        predicted_values = predicted_softmax.gather(1, predicted_classes.unsqueeze(1)).squeeze()
    
    print("Probabilidades das classes:\n", predicted_softmax)
    print("Valor da previsão:\n", predicted_values)
    print("Classe prevista:\n", predicted_classes)
    
    display_image(images.squeeze().numpy(), 28, f"Classe prevista: {predicted_classes.item()}")

def mnist_train():
    torch.manual_seed(123)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "../studies/pytorch/data")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted_classes = torch.argmax(output, dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples * 100
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    test_accuracy = evaluate(model, data_dir)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    torch.save(model.state_dict(), os.path.join(project_dir, "binary-mnist.pth"))

def display_image(image, original_size, title):
    image = image * 255
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()

def evaluate(model, data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            predicted_classes = torch.argmax(output, dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct_predictions / total_samples * 100
    return accuracy

def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 10)
    )




def main():
    if len(sys.argv) != 2:
        print("Uso: python script.py [train|test]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "train":
        mnist_train()
    elif mode == "test":
        mnist_test()
    else:
        print("Argumento inválido. Use 'train' para treinar ou 'test' para testar o modelo.")
        sys.exit(1)

if __name__ == "__main__":
    main()