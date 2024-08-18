import torch
from abc import ABC, abstractmethod


class Flow(ABC):

    def __generate_data(self, generator, seed, device):
        torch.manual_seed(seed)
        random_noise = torch.randn((1, 2), device=device)
        generated_samples = generator(random_noise)
        generated_samples = generated_samples.cpu().detach().numpy()
        synthetic_dur, synthetic_size = generated_samples[0, 0], generated_samples[0, 1]
        return synthetic_dur, synthetic_size

    @abstractmethod
    def __create_flow(self, network, generator, seed, device, addresses):
        pass
