import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import optuna
import functools
from neural_networks import Generator, Discriminator

TRAINING_DATA_LENGTH = 3030487
TRAINING_DATA = "training_data_tensor.pt"
MIN_TRIALS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77


class GANOptimiser:
    def __init__(self, data_path, min_trials, max_trials, device, seed):
        self.data_path = data_path
        self.min_trials = min_trials
        self.max_trials = max_trials
        self.seed = seed
        self.device = device
        self.study = None
        self.interrupted = False

    def __load_data(self):
        training_data = torch.load(self.data_path).to(torch.float32).to(self.device)
        training_data_length = len(training_data)
        train_labels = torch.zeros(size=(training_data_length, 1)).to(self.device)
        self.train_set = [(training_data[i], train_labels[i]) for i in range(training_data_length)]

    def __create_data_loader(self, params):
        train_loader = DataLoader(self.train_set, batch_size=params["batch_size"], shuffle=True, drop_last=True)
        print("Finished creating data loader!")
        return train_loader

    def __create_opt_lf(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.loss_func = nn.BCELoss().to(self.device)

    def __create_study(self):
        self.study = optuna.create_study(directions=["minimize", "maximize"],
                                         study_name="GAN-Optimiser",
                                         sampler=optuna.samplers.NSGAIISampler(),
                                         storage="sqlite:///demonstration_optimisation.db",
                                         load_if_exists=True)
        self.wrapped_objective = functools.partial(self.__objective__, self.study)

    def __objective(self, trial, additional_arg):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", low=1e-5, high=1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", low=500, high=2000, step=100),
            "epochs": trial.suggest_int("epochs", low=5, high=50, step=5),
        }

        g_loss, d_loss = self.__train_and_optimise(params, additional_arg)

        return g_loss, d_loss

    def initialise(self):
        self.__load_data()
        self.__create_opt_lf()
        self.__create_study()

    def __evaluate_trial(self, study, current_g_loss, current_d_loss):
        should_prune = False

        if current_g_loss == 0.0 or current_d_loss == 0.0:  # one of the networks are no longer learning
            should_prune = True

        else:
            historic_g_loss = []
            historic_d_loss = []

            for trial in study.trials:
                # Only considering/calculating trials that have been completed
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    historic_g_loss.append(
                        trial.values[0])  # accessing and appending the historic value for generator loss
                    historic_d_loss.append(trial.values[1])  # doing the same for the discriminator loss

                    # Convert to numpy arrays
                    array_g_loss = np.array(historic_g_loss)
                    array_d_loss = np.array(historic_d_loss)

                    # Calculate the mean losses
                    mean_g_loss = np.mean(array_g_loss)
                    mean_d_loss = np.mean(array_d_loss)

            should_prune = current_g_loss > mean_g_loss or current_d_loss < mean_d_loss

        return should_prune

    def __prune(self, study, min_trials, min_epochs, current_epoch, current_g_loss, current_d_loss):
        completed_trials = 0

        # checking how many trials are complete to compare to the current trial
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                completed_trials += 1

        # if there are sufficient trials to compare, it will compare
        if completed_trials >= min_trials:
            if current_epoch == min_epochs:
                should_prune = self.__evaluate_trial(study, current_g_loss, current_d_loss)
                if should_prune:
                    print("Trial has been pruned")
                    raise optuna.TrialPruned()

    def __train_and_optimise(self, params, study):
        discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=params["learning_rate"])
        generator_optimiser = optim.Adam(self.generator.parameters(), lr=params["learning_rate"])
        train_loader = self.__create_data_loader(params)
        warmup_epochs = params["epochs"] // 2

        start_time = time.time()
        for epoch in range(params["epochs"]):

            for n, (real_samples, _) in enumerate(tqdm(train_loader)):

                # DATA FOR DISCRIMINATOR
                torch.manual_seed(self.seed)
                real_samples_labels = torch.ones((params["batch_size"], 1), device=self.device)

                latent_space_samples = torch.randn((params["batch_size"], 2), device=self.device)
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((params["batch_size"], 1), device=self.device)

                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

                # TRAINING DISCRIMINATOR
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                discriminator_loss = self.loss_func(output_discriminator, all_samples_labels)
                discriminator_loss.backward()
                discriminator_optimiser.step()

                # DATA FOR GENERATOR
                torch.manual_seed(self.seed)
                latent_space_samples = torch.randn((params["batch_size"], 2), device=DEVICE)

                # TRAINING GENERATOR
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                generator_loss = self.loss_func(output_discriminator_generated, real_samples_labels)
                generator_loss.backward()
                generator_optimiser.step()

                if epoch % 1 == 0 and n == params["batch_size"] - 1:
                    print(f"Epoch: {epoch} | G. Loss: {generator_loss} | D. Loss: {discriminator_loss}")

            # Pruning
            if epoch == warmup_epochs:
                # The trial is halfway through completion, now checking whether to prune or not
                self.__prune(study=study, min_trials=self.min_trials, min_epochs=warmup_epochs, current_epoch=epoch,
                      current_g_loss=generator_loss, current_d_loss=discriminator_loss)

        end_time = time.time()
        run_time = round(end_time - start_time, 2)
        print(f"Trial Complete!\nRun time for this trial was {run_time} seconds.\n")

        return generator_loss, discriminator_loss

    def optimise(self):
        # Accessing the last trial in the study to check whether the previous optimisation loop was halted or not
        interrupted = self.study.trials[-1].state in {optuna.trial.TrialState.FAIL, optuna.trial.TrialState.WAITING}
        if interrupted:
            complete_trials = sum(1 for trial in self.study.trials if trial.state in {optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE})
            print("Continuing previous optimisation loop...")
            self.study.optimize(self.wrapped_objective, n_trials=self.max_trials - complete_trials)
        else:
            print("Starting new optimisation loop")
            self.study.optimize(self.wrapped_objective, n_trials=self.max_trials)

        print(f"Optimisation complete!")
        # TODO: work out if I am going to find a way to store the times taken for halted optimisation loops
        # TODO: implement report method to include save_trials, display_all_trials, and display_best_trial
