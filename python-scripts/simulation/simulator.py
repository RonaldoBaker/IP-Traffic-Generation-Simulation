import torch
import time
import bisect
import operator
import networkx as nx
import numpy as np
from tqdm import tqdm
from networks import ConvergedNetwork, NonConvergedNetwork
from flow_factory import FlowFactory
from ..training.neural_networks import Generator
from event import Event
class Simulator:
    def __init__(self, generator_path, seed, device, addresses, graph, num_wavelengths, min_flows, expon_dist):
        self.generator_path = generator_path
        self.seed = seed
        self.device = device
        self.addresses = addresses
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.min_flows = min_flows
        self.expon_dist = expon_dist
        self.sim_clock = 0
        self.sent = 0
        self.blocked = 0
        self.assortment_key = operator.attrgetter("event_time")
        self.events = list()
        self.durations = list()
        self.iat_times = list()
        self.start_time = 0
        self.end_time = 0
        self.run_time = 0
        self.__initialise()

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

    def __timing_routine(self):
        next_event = self.events.pop(0)
        advanced_time = next_event.event_time
        self.sim_clock = advanced_time
        return next_event

    def __event_routine(self, event):
        current_flow = event.associated_flow
        type = event.event_type

        if type == "ARRIVAL" or type == "HOP":
            if self.architecture == "converged":
                next_event_type, updated_flow = self.network.push_flow(current_flow, type)
                if next_event_type != "BLOCKED":
                    self.__schedule_event(self.sim_clock + updated_flow.hop_time, next_event_type, updated_flow)
                else:
                    self.blocked += 1
            else:
                next_event_type, updated_flow = self.network.push_flow(current_flow)
                if next_event_type == "BLOCKED":
                    self.__schedule_event(self.sim_clock + updated_flow.dur, "DEPART", updated_flow)
                else:
                    self.blocked += 1
        else:
            self.network.end_flow(current_flow)
            self.sent += 1

        if type == "ARRIVAL":
            new_flow = self.flow_factory.generate_flow()
            np.random.seed(self.seed)
            iat = np.random.choice(self.expon_dist)
            self.iat_times.append(iat)
            self.__schedule_event(self.sim_clock + iat, "ARRIVAL", new_flow)

    def run_simulation(self):
        print(f"Starting {self.architecture} simulation")
        self.start_time = time.time()
        progress = 0
        progress_bar = tqdm(total=self.min_flows, desc="Simulating", unit = "flow sent", leave=False)

        while ((self.sent + self.blocked) < self.min_flows):
            next_event = self.__timing_routine()
            self.__event_routine(next_event)
            progress = self.sent + self.blocked
            progress_bar.update(progress - progress_bar.n)

        self.end_time = time.time()
        self.run_time = self.end_time - self.start_time
