import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neural_networks import Generator, Discriminator
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class GANTrainer:
    def __init__(self, epochs=0, learning_rate=0, batch_size=0, data_path="", device=None, seed=0):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_path = data_path
        self.device = device
        self.seed = seed
        self.epoch_count = []
        self.generator_loss_values = []
        self.discriminator_loss_values = []
        self.start_time = 0
        self.end_time = 0
        self.generator = None

    def __load_data__(self):
        self.training_data = torch.load(self.data_path)
        self.training_data = self.training_data.to(torch.float32).to(self.device)
        self.training_data_length = len(self.training_data)
        print("Finished loading data!")

    def __create_data_loader__(self):
        train_labels = torch.zeros(size=(self.training_data_length, 1))
        train_labels = train_labels.to(self.device)
        train_set = [(self.training_data[i], train_labels[i]) for i in range(self.training_data_length)]
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        print("Finished creating data loader!")

    def __create_neural_networks__(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        print("Finished creating neural networks!")

    def __create_lf_opt(self):
        self.loss_func = nn.BCELoss()
        self.discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        self.generator_optimiser = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        print("Finished creating loss function and optimisers!")

    def initialise(self):
        self.__load_data__()
        self.__create_data_loader__()
        self.__create_neural_networks__()
        self.__create_lf_opt()

    def train(self):
        self.start_time = time.time()
        for epoch in range(self.epochs):
            for n, (real_samples, _) in enumerate(tqdm(self.train_loader)):
                # DATA FOR TRAINING THE DISCRIMINATOR
                torch.manual_seed(self.seed)
                real_samples_labels = torch.ones((self.batch_size, 1), device=self.device)
                latent_space_samples = torch.randn((self.batch_size, 2), device=self.device)
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((self.batch_size, 1), device=self.device)

                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

                # TRAINING THE DISCRIMINATOR
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                discriminator_loss = self.loss_func(output_discriminator, all_samples_labels)
                discriminator_loss.backward()
                self.discriminator_optimiser.step()

                # DATA FOR TRAINING THE GENERATOR
                # Storing random data in latent_space_samples with a number of lines to equal batch_size
                torch.manual_seed(self.seed)
                latent_space_samples = torch.randn((self.batch_size, 2), device=self.device)

                # TRAINING THE GENERATOR
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                generator_loss = self.loss_func(output_discriminator_generated, real_samples_labels)
                generator_loss.backward()
                self.generator_optimiser.step()

                # Show loss
                if n == self.batch_size - 1:
                    self.epoch_count.append(epoch)
                    self.generator_loss_values.append(generator_loss.cpu().detach().numpy())
                    self.discriminator_loss_values.append(discriminator_loss.cpu().detach().numpy())
                    print(f"Epoch: {epoch} | D Loss: {discriminator_loss} | G Loss: {generator_loss}")

        self.end_time = time.time()
        self.generator.trained = True
        print("Training complete!")

    def __visualise__(self):
        plt.plot(self.epoch_count, self.generator_loss_values, label="Generator Loss")
        plt.plot(self.epoch_count, self.discriminator_loss_values, label="Discriminator Loss")
        plt.title("Generator and Discriminator Loss Curves")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def report(self):
        run_time = round(self.end_time - self.start_time, 2)
        converted_time = time.strftime("%H hours %M minutes %S seconds", time.gmtime(run_time))
        print(f"Report:\nRun time: {converted_time}")
        print(f"Epochs: {self.epochs} epochs\nLearning rate: {self.learning_rate}\nBatch size: {self.batch_size} samples/batch")
        self.__visualise__()

    def __load_default_generator__(self):
        self.generator = Generator()

    def save_model(self, path):
        if self.generator is None:
            raise ValueError("An error has occurred: Generator has not been initialised. Unable to save Generator model")
        elif (self.generator is not None) & (self.generator.trained is False):
            raise ValueError("An error has occurred: Generator has not been trained. Unable to save Generator model")
        else:
            try:
                torch.save(self.generator.state_dict(), path)
            except RuntimeError as E:
                raise RuntimeError(f"An error has occurred: {E}")

    @staticmethod
    def load_model(self, path):
        if self.generator is not None:
            # A Trained Generator already exists within the class
            pass
            # TODO: Finish this

