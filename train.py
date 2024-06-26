import torch
import torch.nn as nn
import torch.optim as optim
from cedar_dataset import CedarDataset
from networks import Cedar, VGG16, ResNet
from torch.utils.data import random_split, DataLoader


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


def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def create_DataLoaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_Cedar(model: Cedar = None) -> Cedar:
    full_dataset = CedarDataset(root_dir="signatures")
    train_dataset, test_dataset = split_dataset(full_dataset, 0.8)
    train_loader, val_loader, _ = create_DataLoaders(train_dataset, test_dataset, batch_size=32)

    model = model if model is Cedar else Cedar()
    model = train_model(model, train_loader, val_loader)

    return model


def train_VGG16(model: VGG16 = None) -> VGG16:
    full_dataset = CedarDataset(root_dir="signatures")
    train_dataset, test_dataset = split_dataset(full_dataset, 0.8)
    train_loader, val_loader, _ = create_DataLoaders(train_dataset, test_dataset, batch_size=32)

    model = model if model is VGG16 else VGG16()
    model = train_model(model, train_loader, val_loader)

    return model


def train_ResNet(model: ResNet = None) -> ResNet:
    full_dataset = CedarDataset(root_dir="signatures")
    train_dataset, test_dataset = split_dataset(full_dataset, 0.8)
    train_loader, val_loader, _ = create_DataLoaders(train_dataset, test_dataset, batch_size=32)

    model = model if model is ResNet else ResNet()
    model = train_model(model, train_loader, val_loader)

    return model

# def train_with_different_sizes(model, dataset, sizes=[0.5, 0.75, 1.0]):
#     results = {}
#     for size in sizes:
#         train_dataset, _ = split_dataset(dataset, size)
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#         trained_model = train_model(model, train_loader, val_loader)
#         accuracy = test_model(trained_model, test_loader)
#         results[size] = accuracy
#     return results
#
#
# def train_with_hyperparams(model, train_loader, val_loader, epochs_list, lr_list, batch_sizes):
#     results = {}
#     for epochs in epochs_list:
#         for lr in lr_list:
#             for batch_size in batch_sizes:
#                 optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#                 train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#                 val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#                 trained_model = train_model(model, train_loader, val_loader, num_epochs=epochs, learning_rate=lr)
#                 accuracy = test_model(trained_model, test_loader)
#                 results[(epochs, lr, batch_size)] = accuracy
#     return results
