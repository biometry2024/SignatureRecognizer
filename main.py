from cedar_network import CedarNetwork
from train import train


def main():
    model = load_model()
    while True:
        print_options()
        option = input()
        match option:
            case "1":
                if model == None:
                    print("No model exists - you need to retrain model")
                else:
                    test_signature(model)
            case "2":
                model = train()
            case _:
                break


def print_options():
    print("Input number of option:")
    print("1) Test signature")
    print("2) Retrain model")
    print("0) Exit")


def load_model() -> CedarNetwork:
    try:
        model = CedarNetwork()
        model.load_state_dict("model_weights.pth")
        return model
    except:
        return None


def test_signature(model: CedarNetwork):
    model.eval()
    # TODO: allow file input with safety checks and predict image (needed Transform due to different image sizes)


if __name__ == "__main__":
    main()
