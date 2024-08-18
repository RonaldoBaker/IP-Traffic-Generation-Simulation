import random
from flow import Flow

# TODO: Might make a "FlowFactory" class and then make a make_flow() method so that I don't have to keep passing all
#  these parameters
"""
I can make one instance of the FlowFactory and pass the type of network, generator, seed, device and addresses
Then depending on the type of network, a different method can be used to make the flow
"""


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
