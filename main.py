from PIL import Image
from datasets import CedarDataset
from networks import Cedar, VGG16, ResNet
from train import train_Cedar
import torch


def main():
    model = load_model("Cedar", "model_weights.pth")
    while True:
        print_options()
        option = input()
        match option:
            case "1":
                if model is not None:
                    test_signature(model)
                else:
                    print("No model exists - you need to train model")
            case "2":
                model = train_Cedar()
            case "3":
                if model is not None:
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


def load_model(model_type, file_name):
    if model_type == "Cedar":
        model = Cedar()
    elif model_type == "VGG16":
        model = VGG16()
    elif model_type == "ResNet":
        model = ResNet()
    else:
        raise ValueError("Wrong model type")
    try:
        model.load_state_dict(torch.load(file_name))
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
