import torch
import random
from abc import ABC, abstractmethod


class Flow:

    def __generate_data(self, generator, seed, device):
        torch.manual_seed(seed)
        random_noise = torch.randn((1, 2), device=device)
        generated_samples = generator(random_noise)
        generated_samples = generated_samples.cpu().detach().numpy()
        synthetic_dur, synthetic_size = generated_samples[0, 0], generated_samples[0, 1]
        return synthetic_dur, synthetic_size


class ConvergedFlow(Flow):
    def __init__(self, network, generator, seed, device, addresses):
        self.__create_flow(network, generator, seed, device, addresses)

    def __create_flow(self, network, generator, seed, device, addresses):
        self.dur, self.size = self.__generate_data(generator, seed, device)
        random_addresses = random.sample(addresses, 2)
        self.src, self.dst = random_addresses[0], random_addresses[1]
        self.current_node = self.src
        self.prev_node = None

        # Calculate the shortest path
        self.route = network.find_route(self.src, self.dst)

        # Removes the first node in the route which is the starting node for the flow path
        self.route = self.route[1:]

        # How long it will take to make each hop, used to schedule the next event time
        self.hop_time = self.dur / len(self.route)


class NonConvergedFlow(Flow):
    def __init__(self, network, generator, seed, device, addresses):
        self.__create_flow(network, generator, seed, device, addresses)

    def __create_flow(self, network, generator, seed, device, addresses):
        self.wavelength = ""
        self.dur, self.size = self.__generate_data(generator, seed, device)
        random_addresses = random.sample(addresses, 2)
        self.src, self.dst = random_addresses[0], random_addresses[1]

        # Calculate the shortest path
        self.route = network.find_route(self.src, self.dst)

        # Define a lightpath for the flow from its node pairs
        self.lightpath = []
        for i in range(len(self.route)):
            try:
                self.lightpath.append((self.route[i], self.route[i + 1]))
            except IndexError:  # Gotten to the end of the path list and has accounted for all the links
                pass