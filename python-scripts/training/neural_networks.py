import torch.nn as nn

class Generator(nn.Module):
    trained = False

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input is 2D, first hidden layer is composed of 2 neurons with ReLU activation
            nn.Linear(2, 32),
            nn.ReLU(),

            # Have to use dropout to avoid overfitting
            nn.Dropout(0.3),

            # second and third layers are composed to 128 and 64 neurons, respectively
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.3),

            # output is composed of a single neuron with sigmoidal activation to represent a probability
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output
