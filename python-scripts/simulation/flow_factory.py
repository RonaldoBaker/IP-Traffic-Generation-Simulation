from flows import NonConvergedFlow, ConvergedFlow
from networks import NonConvergedNetwork, ConvergedNetwork


class FlowFactory:
    def __init__(self, network, generator, seed, device, addresses):
        self.network = network
        self.generator = generator
        self.seed = seed
        self.device = device
        self.addresses = addresses
        if isinstance(self.network, ConvergedNetwork):
            self.generate_flow = self.__create_conv_flow()
        elif isinstance(self.network, NonConvergedNetwork):
            self.generate_flow = self.__create_nonconv_flow()

    def __create_conv_flow(self):
        return ConvergedFlow(self.network, self.generator, self.seed, self.device, self.addresses)

    def __create_nonconv_flow(self):
        return NonConvergedFlow(self.network, self.generator, self.seed, self.device, self.addresses)

