import torch
import numpy as np
from simulator import Simulator

# Constants
ARCHITECTURE = ""
GENERATOR_PATH = "../../trained-generator/generator.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH = "../../uk-net/UKnet.txt"
ADDRESSES = [i for i in range(1, 22)]
WAVELENGTHS = 40
RANDOM_SEED = 77
MIN_FLOWS_SENT = 1e2
EXPON_SCALE = 0.1
EXPON_DIST = np.random.exponential(scale = EXPON_SCALE, size = 20000000)


def main():
    simulator = Simulator(GENERATOR_PATH, RANDOM_SEED, DEVICE, ADDRESSES, GRAPH, WAVELENGTHS, MIN_FLOWS_SENT, EXPON_DIST)
    simulator.run_simulation()

if __name__ == "__main__":
    main()
