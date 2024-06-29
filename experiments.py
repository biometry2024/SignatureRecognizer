from datasets import CedarDataset
from networks import Cedar, VGG16, ResNet
from train import train_model, split_dataset, create_DataLoaders


def experiments():
    print("Comparing models")
    compare()


def compare():
    full_dataset = CedarDataset(root_dir="signatures")

    train_dataset, val_dataset = split_dataset(full_dataset, 0.8)
    train_loader, val_loader = create_DataLoaders(train_dataset, val_dataset, batch_size=32)

    model = Cedar()
    trained_cedar = train_model(model, train_loader, val_loader, "cedar_model.pth")

    model = VGG16()
    trained_vgg16 = train_model(model, train_loader, val_loader, "vgg16_model.pth")

    model = ResNet()
    trained_resnet = train_model(model, train_loader, val_loader, "resnet_model.pth")


if __name__ == '__main__':
    experiments()
