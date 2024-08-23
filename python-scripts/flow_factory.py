from flows import NonConvergedFlow, ConvergedFlow
import torch

class FlowFactory:
    def __init__(self, architecture, network, generator, seed, device, addresses):
        self.network = network
        self.generator = generator
        self.seed = seed
        self.device = device
        self.addresses = addresses
        if architecture.lower() == "converged":
            self.generate_flow = self.__create_conv_flow()
        elif architecture.lower() == "nonconverged":
            self.generate_flow = self.__create_nonconv_flow()

    def __create_conv_flow(self):
        return ConvergedFlow(self.network, self.generator, self.seed, self.device, self.addresses)

    def __create_nonconv_flow(self):
        return NonConvergedFlow(self.network, self.generator, self.seed, self.device, self.addresses)

