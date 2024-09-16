import networkx as nx
import torch
import torch.nn as nn
import random
import time
import bisect
import operator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from neural_networks import Generator
from flow_factory import FlowFactory


# Constants
ARCHITECTURE = ""
GENERATOR_PATH = "../trained-generator/generator.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77
MIN_FLOWS_SENT = 1e2


def main():

    pass


if __name__ == "__main__":
    main()
