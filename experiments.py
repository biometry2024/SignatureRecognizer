import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets import CedarDataset
from sklearn.metrics import precision_score, recall_score, f1_score


def experiments():
    while True:
        option = input()
        match option:
            case "1":
                _, _, _, = compare()
                print("Comparing models")
            case "2":
                print("Test model")
            case "3":
                print("Evaluate model")
            case "0":
                break
            case _:
                print("Wrong option")


def compare():
    vgg16_accuracy = 0
    resnet_accuracy = 0
    cedar_accuracy = 0

    return vgg16_accuracy, resnet_accuracy, cedar_accuracy


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
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy, precision, recall, f1

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


if __name__ == '__experiments__':
    experiments()
