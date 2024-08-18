import random
from flow import Flow

# TODO: Might make a "FlowFactory" class and then make a make_flow() method so that I don't have to keep passing all
#  these parameters
"""
I can make one instance of the FlowFactory and pass the type of network, generator, seed, device and addresses
Then depending on the type of network, a different method can be used to make the flow
"""


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
