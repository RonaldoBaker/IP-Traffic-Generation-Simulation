import networkx as nx

class ConvergedNetwork:
    def __init__(self, graph, num_of_wavelengths):
        self.topology = graph
        self.__allocate_capacity__(num_of_wavelengths)

    def __allocate_capacity__(self, num_of_wavelengths):
        self.links = dict()
        for edge in self.topology.edges():
            node1, node2 = edge[0], edge[1]
            self.links[(node1, node2)] = num_of_wavelengths

    def __check_capacity__(self, node1, node2):
        try:
            return self.links[(node1, node2)] > 0
        except KeyError:
            return self.links[(node2, node1)] > 0

    def __use_capacity__(self, node1, node2):
        try:
            self.links[(node1, node2)] -= 1
        except KeyError:
            self.links[(node2, node1)] -= 1

    def __release_capacity__(self, node1, node2):
        try:
            self.links[(node1, node2)] += 1
        except KeyError:
            try:
                self.links[(node2, node1)] += 1
            except KeyError:  # One of the nodes must be None
                pass  # there is no link to release capacity for

    def push_flow(self, flow, type):
        node1, node2 = flow.current_node, flow.route[0]
        if type == "HOP":
            self.__release_capacity__(flow.prev_node, flow.current_node)

        sufficient_capacity = self.__check_capacity__(node1, node2)
        if sufficient_capacity:
            self.__use_capacity__(node1, node2)
            next_event_type = flow.hop()
            return next_event_type, flow
        else:
            next_event_type = "BLOCKED"
            return next_event_type, flow

    def end_flow(self, flow):
        node1, node2 = flow.prev_node, flow.current_node
        self.__release_capacity__(node1, node2)

    def find_route(self, src, dst):
        return nx.shortest_path(self.topology, source=src, target=dst)


class NonConvergedNetwork:
    def __init__(self, graph, num_of_wavelengths):
        self.topology = graph
        self.__allocate_capacity__(num_of_wavelengths)

    def __allocate_capacity__(self, num_of_wavelengths):
        self.links = dict()
        for edge in self.topology.edges():
            node1, node2 = edge[0], edge[1]
            # Each item in the list will represent a unique wavelength in the link
            self.links[(node1, node2)] = [format(i, "02") for i in range(1, num_of_wavelengths + 1)]

    def __check_capacity__(self, node1, node2):
        # Return all the available wavelengths
        try:
            return self.links[(node1, node2)]
        except KeyError:
            return self.links[(node2, node1)]

    def __use_capacity__(self, node1, node2, wavelength):
        try:
            self.links[(node1, node2)].remove(wavelength)
        except KeyError:
            self.links[(node2, node1)].remove(wavelength)

    def __release_capacity__(self, node1, node2, wavelength):
        try:
            self.links[(node1, node2)].append(wavelength)
        except KeyError:
            self.links[(node2, node1)].append(wavelength)

    def find_route(self, src, dst):
        return nx.shortest_path(self.topology, source=src, target=dst)

    def __wavelength_assignment__(self, flow):
        wavelength_counter = dict()
        viable_lightpaths = []

        for link in flow.lightpath:
            available_wavelengths = self.__check_capacity__(link[0], link[1])
            for wavelength in available_wavelengths:
                wavelength_counter[wavelength] = wavelength_counter.get(wavelength, 0) + 1

        for key in wavelength_counter.keys():
            if wavelength_counter[key] == len(
                    flow.lightpath):  # Pick the first wavelength that is available on all the links in the path
                flow.wavelength = key
                return flow

        flow.wavelength = "00"
        return flow

    def push_flow(self, flow):
        updated_flow = self.__wavelength_assignment__(flow)
        if updated_flow.wavelength == "00":
            return "BLOCKED", updated_flow

        else:
            # Uses lightpath from flow to update the capacity on each link in the lightpath
            for link in updated_flow.lightpath:
                self.__use_capacity__(link[0], link[1], updated_flow.wavelength)
            return "DEPART", updated_flow

    def end_flow(self, flow):
        for link in flow.lightpath:
            self.__release_capacity__(link[0], link[1], flow.wavelength)
