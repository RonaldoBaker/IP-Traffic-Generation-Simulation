import time
import torch
import bisect
import operator
import networkx as nx
from networks import ConvergedNetwork, NonConvergedNetwork
from flow_factory import FlowFactory
from neural_networks import Generator
from event import Event
class Simulator:
    def __init__(self, generator_path, seed, device, addresses, graph, num_wavelengths):
        self.generator_path = generator_path
        self.seed = seed
        self.device = device
        self.addresses = addresses
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.sim_clock = 0
        self.sent = 0
        self.blocked = 0
        self.assortment_key = operator.attrgetter("event_time")
        self.events = list()
        self.durations = list()
        self.__set_architecture()

    def __set_architecture(self):
        print("What kind of architecture would you like to simulate today?")
        print("1. Converged\n2. Non-converged")
        print("Enter '1' for converged or '2' for non-converged")
        while True:
            try:
                choice = int(input())
                if 0 < choice < 3:
                    break
                else:
                    raise ValueError("Your choice must be 1 or 2")
            except ValueError("Choice must be of type integer") as e:
                print(e)
        self.architecture = "converged" if (choice == 1) else "nonconverged"

    def __set_network(self):
        G = nx.read_adjlist(self.graph, nodetype=int)
        if self.architecture == "converged":
            self.network = ConvergedNetwork(G, self.num_wavelengths)
        else:
            self.network = NonConvergedNetwork(G, self.num_wavelengths)

    def __schedule_event(self, event_time, event_type, associated_flow):
        new_event = Event(event_time, event_type, associated_flow)
        bisect.insort(self.events, new_event, key=self.assortment_key)

    def __initialise_events_list(self):
        # new_flow = self.flow_generator.generate_flow()
        # self.durations.append(new_flow.dur)  # Cannot remember what I actually use this for?
        self.__schedule_event(0, "ARRIVAL", self.flow_factory.generate_flow())

    def __initialise(self):
        self.__set_architecture()
        self.__set_network()
        self.generator = Generator().load_state_dict(torch.load(self.generator_path, map_location=self.device))
        self.flow_factory = FlowFactory(
            self.network,
            self.generator,
            self.seed,
            self.device,
            self.addresses)
        self.__schedule_event(0, "ARRIVAL", self.flow_factory.generate_flow())




