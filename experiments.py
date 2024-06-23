import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from cedar_dataset import CedarDataset
from cedar_network import CedarNetwork
from sklearn.metrics import precision_score, recall_score, f1_score

class VGG16Binary(nn.Module):
    def __init__(self):
        super(VGG16Binary, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16(x)
        x = self.sigmoid(x)
        return x

class ResNetBinary(nn.Module):
    def __init__(self):
        super(ResNetBinary, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience = 2
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    return accuracy

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predictions = (outputs.squeeze() > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1


# Przykład użycia:
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
# print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

# Przykład:
# train_dataset, test_dataset = split_dataset(full_dataset, 0.8)

def train_with_different_sizes(model, dataset, sizes=[0.5, 0.75, 1.0]):
    results = {}
    for size in sizes:
        train_dataset, _ = split_dataset(dataset, size)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        trained_model = train_model(model, train_loader, val_loader)
        accuracy = test_model(trained_model, test_loader)
        results[size] = accuracy
    return results


# Przykład użycia:
# results = train_with_different_sizes(model, full_dataset)
# print(results)

def train_with_hyperparams(model, train_loader, val_loader, epochs_list, lr_list, batch_sizes):
    results = {}
    for epochs in epochs_list:
        for lr in lr_list:
            for batch_size in batch_sizes:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                trained_model = train_model(model, train_loader, val_loader, num_epochs=epochs, learning_rate=lr)
                accuracy = test_model(trained_model, test_loader)
                results[(epochs, lr, batch_size)] = accuracy
    return results


# Przykład użycia:
# epochs_list = [10, 20, 50]
# lr_list = [0.01, 0.001, 0.0001]
# batch_sizes = [16, 32, 64]
# results = train_with_hyperparams(model, train_loader, val_loader, epochs_list, lr_list, batch_sizes)
# print(results)


class ModifiedCedarNetwork(nn.Module):
    def __init__(self):
        super(ModifiedCedarNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Added layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 9 * 14, 128)  # Adjusted based on new layer
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))  # Added layer
        x = x.view(-1, 256 * 9 * 14)  # Adjusted based on new layer
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Porównaj z oryginalnym modelem:
# modified_model = ModifiedCedarNetwork()
# trained_modified_model = train_model(modified_model, train_loader, val_loader)
# modified_model_accuracy = test_model(trained_modified_model, test_loader)
#
# print(f"ModifiedCedarNetwork Accuracy: {modified_model_accuracy}%")

class AugmentedCedarDataset(CedarDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((150, 220)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

# Użyj zbioru danych z augmentacją
# augmented_dataset = AugmentedCedarDataset(root_dir="signatures")
# train_dataset, val_dataset = split_dataset(augmented_dataset, 0.8)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
# # Trenuj model
# augmented_model = CedarNetwork()
# trained_augmented_model = train_model(augmented_model, train_loader, val_loader)
# augmented_model_accuracy = test_model(trained_augmented_model, test_loader)
#
# print(f"Augmented Data Model Accuracy: {augmented_model_accuracy}%")

def add_noise(image, noise_factor=0.1):
    noisy_image = image + noise_factor * torch.randn(image.size())
    noisy_image = torch.clip(noisy_image, 0., 1.)
    return noisy_image

# Dodaj zakłócenia do test_loader
# noisy_test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# noisy_test_loader.dataset.transform = transforms.Compose([
#     transforms.Resize((150, 220)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: add_noise(x))
# ])
#
# # Testuj model na zakłóconych danych
# noisy_model_accuracy = test_model(trained_model, noisy_test_loader)
# print(f"Noisy Data Model Accuracy: {noisy_model_accuracy}%")

def error_analysis(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    errors = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predictions = (outputs.squeeze() > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            # Zbierz błędy
            for i in range(len(labels)):
                if labels[i] != predictions[i]:
                    errors.append((images[i], labels[i], predictions[i]))

    return errors

# Przykład użycia:
# errors = error_analysis(trained_model, test_loader)
# print(f"Number of errors: {len(errors)}")

# Analizuj błędy
# for img, label, prediction in errors:
#     print(f"Label: {label}, Prediction: {prediction}")
#     plt.imshow(img.permute(1, 2, 0).numpy(), cmap='gray')
#     plt.show()

full_dataset = CedarDataset(root_dir="signatures")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

test_dataset = CedarDataset(root_dir="signatures/test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

cedar_model = CedarNetwork()
cedar_model = train_model(cedar_model, train_loader, val_loader)
cedar_accuracy = test_model(cedar_model, test_loader)

# VGG16
vgg16_model = VGG16Binary()
vgg16_model = train_model(vgg16_model, train_loader, val_loader)
vgg16_accuracy = test_model(vgg16_model, test_loader)

# ResNet
resnet_model = ResNetBinary()
resnet_model = train_model(resnet_model, train_loader, val_loader)
resnet_accuracy = test_model(resnet_model, test_loader)

print(f"CedarNetwork Accuracy: {cedar_accuracy}%")
print(f"VGG16 Accuracy: {vgg16_accuracy}%")
print(f"ResNet Accuracy: {resnet_accuracy}%")