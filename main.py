from PIL import Image
from torch.utils.data import DataLoader

from datasets import CedarDataset
from networks import Cedar, VGG16, ResNet
from train import train_model, split_dataset, create_DataLoaders, train_Cedar
import torch


def main():
    model = load_model()
    while True:
        print_options()
        option = input()
        match option:
            case "1":
                if model != None:
                    test_signature(model)
                else:
                    print("No model exists - you need to train model")
            case "2":
                model = train_Cedar()
            case "3":
                if model != None:
                    model = train_Cedar(model)
                else:
                    print("No model exists - cannot retrain")
            case "0":
                break
            case _:
                print("Wrong option")


def print_options():
    print("Input number of option:")
    print("1) Test signature")
    print("2) Train model from scratch")
    print("3) Retrain existing model")
    print("0) Exit")


def load_model() -> Cedar:
    try:
        model = Cedar()
        model.load_state_dict(torch.load("model_weights.pth"))
        return model
    except:
        return None


def test_signature(model: Cedar):
    print("Input image path to check:")
    image_path = input()
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        transform = CedarDataset.get_transform()
        model.eval()
        output = model(transform(image)).item()
        print(f"Is given signature original: {output > 0.5} ({output * 100}%)")
    except:
        print(f"Cannot identify image file {image_path}")



if __name__ == "__main__":
    main()
