import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CedarDataset, TestDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from main import load_model
from networks import Cedar, VGG16, ResNet
from train import train_model, split_dataset, create_DataLoaders


def experiments():
    while True:
        option = input()
        match option:
            case "1":
                print("Comparing models")
                compare()
            case "2":
                print("Test model")
                model = load_model()
                if model != None:
                    model_tester(model)
                else:
                    print("No model exists - cannot retrain")

            case "3":
                print("Evaluate model")
            case "0":
                break
            case _:
                print("Wrong option")


def compare():
    full_dataset = CedarDataset(root_dir="signatures")
    test_dataset = TestDataset(root_dir="signatures")

    train_dataset, val_dataset = split_dataset(full_dataset, 0.8)
    train_loader, val_loader = create_DataLoaders(train_dataset, val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Cedar()
    trained_cedar = train_model(model, train_loader, val_loader)
    cedar_accuracy = test_model(trained_cedar, test_loader)

    model = VGG16()
    trained_vgg16 = train_model(model, train_loader, val_loader)
    vgg16_accuracy = test_model(trained_vgg16, test_loader)

    model = ResNet()
    trained_resnet = train_model(model, train_loader, val_loader)
    resnet_accuracy = test_model(trained_resnet, test_loader)

    print(f"Cedar Accuracy: {cedar_accuracy}%")
    print(f"VGG16 Accuracy: {vgg16_accuracy}%")
    print(f"ResNet Accuracy: {resnet_accuracy}%")


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


def model_tester(model):
    test_dataset = TestDataset(root_dir="signatures")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_accuracy = test_model(model, test_loader)


def evaluate_model(model):
    test_dataset = TestDataset(root_dir="signatures")
    data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
