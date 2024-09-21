import torch
from gan_optimiser import GANOptimiser

# Constants
TRAINING_DATA_PATH = "../../training-data/training_data_tensor.pt"
STORAGE_PATH = "sqlite:///demonstration_optimisation.db"
OPTIMISATION_STUDY_CSV = "C:/Users/rrema/OneDrive - University College London/Year 3/ELEC0036 - Project I/Project Code/Traffic Generation 4/Optimisation Trials/gan_optimisation_2024-03-05_10.17.58.801468.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77
MIN_TRIALS = 10
MAX_TRIALS = 200


def main():
    optimiser = GANOptimiser(TRAINING_DATA_PATH, STORAGE_PATH, MIN_TRIALS, MAX_TRIALS, DEVICE, RANDOM_SEED)
    optimiser.initialise()
    optimiser.optimise()
    optimiser.visualise_parallel_plot(OPTIMISATION_STUDY_CSV)


if __name__ == "__main__":
    main()
