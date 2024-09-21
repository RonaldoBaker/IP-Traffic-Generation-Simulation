import torch

# Constants
ARCHITECTURE = ""
GENERATOR_PATH = "../../trained-generator/generator.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77
MIN_FLOWS_SENT = 1e2


def main():

    pass


if __name__ == "__main__":
    main()
