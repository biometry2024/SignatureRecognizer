import torch
import torch.nn as nn
import torch.optim as optim
from cedar_dataset import CedarDataset
from cedar_network import CedarNetwork
from torch.utils.data import random_split, DataLoader


def train(model: CedarNetwork = None) -> CedarNetwork:
    full_dataset = CedarDataset(root_dir="signatures")

    # Split the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = model if model is CedarNetwork else CedarNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    lowest_val_loss = float("inf")
    patience = 2
    epochs_without_improvement = 0
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}/{num_epochs}")
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            if images is None or labels is None:
                continue
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Current batch: {batch_count}/{len(train_loader)}")

        print(f"Loss: {running_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None or labels is None:
                    continue
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")

        torch.save(model.state_dict(), f"model_weights.pth{epoch}")

        # early stopping
        if accuracy >= 99.99:
            print("Early stopping triggered")
            break
        elif val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), "model_weights.pth")
    print("Training complete!")
    return model
