import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys

BATCH_SIZE = 100

class DogCatDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        for path in paths:
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                if 'dog' in img_name:
                    label = 0
                elif 'cat' in img_name:
                    label = 1
                else:
                    continue
                
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),  # Adjusted to match output size after pooling
        nn.ReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 1),  # Output layer has 1 unit
        nn.Sigmoid()        # Sigmoid activation
    )

def evaluate(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()  # Remove extra dimension
            predicted_classes = (outputs > 0.5).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = 100 * correct_predictions / total_samples
    return accuracy

def dog_test():
    torch.manual_seed(123)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = os.path.abspath(os.path.join(project_dir, "../studies/pytorch/data/dataset-dogandcat"))
    test_dirs = [os.path.join(data_dir, "test_set/cachorro/"), os.path.join(data_dir, "test_set/gato/")]

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    test_dataset = DogCatDataset(paths=test_dirs, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = build_model().to(device)
    model_path = os.path.join(project_dir, "binary-catdog.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

def dog_train():
    torch.manual_seed(123)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = os.path.abspath(os.path.join(project_dir, "../studies/pytorch/data/dataset-dogandcat"))
    train_dirs = [os.path.join(data_dir, "training_set/cachorro"), os.path.join(data_dir, "training_set/gato")]

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    train_dataset = DogCatDataset(paths=train_dirs, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()  # Remove extra dimension
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted_classes = (outputs > 0.5).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    model_path = os.path.join(project_dir, "binary-catdog.pth")
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [train/test]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "train":
        dog_train()
    elif mode == "test":
        dog_test()
    else:
        print("Invalid mode! Use 'train' or 'test'.")
        sys.exit(1)
