from PIL import Image
from cedar_dataset import CedarDataset
from cedar_network import CedarNetwork
from train import train
import torch


def main():
    model = load_model()
    while True:
        print_options()
        option = input()
        match option:
            case "1":
                if model == None:
                    print("No model exists - you need to train model")
                else:
                    test_signature(model)
            case "2":
                model = train()
            case "3":
                if model == None:
                    print("No model exists - cannot retrain")
                else:
                    model = train(model)
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


def load_model() -> CedarNetwork:
    try:
        model = CedarNetwork()
        model.load_state_dict(torch.load("model_weights.pth"))
        return model
    except:
        return None


def test_signature(model: CedarNetwork):
    print("Input image path to check:")
    image_path = input()
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        transform = CedarDataset.get_transform()
        model.eval()
        output = model(transform(image)).item()
        print(f"Is given signature original: {output>0.5} ({output*100}%)")
    except:
        print(f"Cannot identify image file {image_path}")


if __name__ == "__main__":
    main()
