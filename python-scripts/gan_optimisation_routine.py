import torch
from gan_optimiser import GANOptimiser

# Constants
TRAINING_DATA_PATH = "../training-data/training_data_tensor.pt"
STORAGE_PATH = "sqlite:///demonstration_optimisation.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77
MIN_TRIALS = 10
MAX_TRIALS = 200


def main():
    optimiser = GANOptimiser(TRAINING_DATA_PATH, STORAGE_PATH, MIN_TRIALS, MAX_TRIALS, DEVICE, RANDOM_SEED)
    optimiser.initialise()
    optimiser.optimise()


if __name__ == "__main__":
    main()
