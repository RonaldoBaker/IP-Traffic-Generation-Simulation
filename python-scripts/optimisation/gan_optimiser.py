import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from datetime import datetime
import pandas as pd
import plotly.express as px
import numpy as np
import optuna
import functools
from ..training.neural_networks import Generator, Discriminator

TRAINING_DATA_LENGTH = 3030487
TRAINING_DATA = "training_data_tensor.pt"
MIN_TRIALS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77


class GANOptimiser:
    def __init__(self, data_path, storage_path, min_trials, max_trials, device, seed):
        self.data_path = data_path
        self.storage_path = storage_path
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
                                         storage=self.storage_path,
                                         load_if_exists=True)
        self.wrapped_objective = functools.partial(self.__objective, self.study)

    def __objective(self, additional_arg, trial):
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
        try:
            interrupted = self.study.trials[-1].state in {optuna.trial.TrialState.FAIL, optuna.trial.TrialState.WAITING}
        except IndexError:
            # There is no last trial because it is being stored in a new path
            print("Starting new optimisation loop")
            self.study.optimize(self.wrapped_objective, n_trials=self.max_trials)
        else:
            # A previous trial has been halted and the optimisation routine isn't finished
            complete_trials = sum(1 for trial in self.study.trials if
                                  trial.state in {optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE})
            print("Continuing previous optimisation loop...")
            self.study.optimize(self.wrapped_objective, n_trials=self.max_trials - complete_trials)

        print(f"Optimisation complete!")

    def save_trials(self):
        df = self.study.trials_dataframe()
        now = str(datetime.now())
        date, current_time = now.split(" ")
        current_time = current_time.replace(":", ".")
        df.to_csv(f"gan_optimisation_{date}_{time}.csv", index=False)
        print(F"Trials saved to gan_optimisation_{date}_{time}.csv\n")

    def __display_all_trials(self):
        df = self.study.trials_dataframe()
        df = pd.DataFrame(df, columns=['number', 'values_0', 'values_1', 'params_batch_size', 'params_epochs',
                                       'params_learning_rate', 'state'])
        df = df.rename(
            columns={"number": "Trial #", "values_0": "G Loss", "values_1": "D Loss", "params_batch_size": "Batch Size",
                     "params_epochs": "Epochs", "params_learning_rate": "Learning Rate", "state": "State"})
        df["Trial #"] += 1  # Adjust the trial numbers

        total = len(df)
        complete = 0
        pruned = 0
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                complete += 1
            elif trial.state == optuna.trial.TrialState.PRUNED:
                pruned += 1

        print(f"{total} TOTAL TRIALS \n{complete} COMPLETED TRIALS \n{pruned} PRUNED TRIALS\n")
        print(df)
        print("\n")

    def __display_best_trial(self):
        best_trial = self.study.best_trials
        best_trial_number = best_trial[0].number
        best_trial_params = best_trial[0].params
        best_trial_values = best_trial[0].values
        print("~Best trial~")
        print(f"Trial #: {best_trial_number + 1}")
        print(f"G Loss: {best_trial_values[0]}\nD Loss: {best_trial_values[1]}")
        print(
            f"Batch Size: {best_trial_params['batch_size']}\nEpochs: {best_trial_params['epochs']}\nLearning Rate: {best_trial_params['learning_rate']}")

    def report(self):
        self.__display_all_trials()
        self.__display_best_trial()

    def __prepare_parallel_plot_data(self, path):
        df = pd.read_csv(path)
        rows_to_remove = []
        epch = []
        lr = []
        bs = []
        g_loss = []
        d_loss = []
        trial_numbers = []
        for index, row in df.iterrows():
            if row["state"] == "PRUNED":
                rows_to_remove.append(index)
        df = df.drop(rows_to_remove)

        # Separating the optimisation data for the generator and discriminator
        for index, row in df.iterrows():
            epch.append(row["params_epochs"])
            lr.append(row["params_learning_rate"])
            bs.append(row["params_batch_size"])
            g_loss.append(row["values_0"])
            d_loss.append(row["values_1"])
            trial_numbers.append(row["number"] + 1)

        generator_df = pd.DataFrame({"Trial Number": trial_numbers, "Epochs": epch, "Learning Rate": lr,
                                     "Batch Size": bs, "Generator Loss": g_loss})
        discriminator_df = pd.DataFrame({"Trial Number": trial_numbers, "Epochs": epch, "Learning Rate": lr,
                                         "Batch Size": bs, "Discriminator Loss": d_loss})
        return generator_df, discriminator_df

    def visualise_parallel_plot(self, path):
        gen_df, disc_df = self.__prepare_parallel_plot_data(path)
        gen_plot = px.parallel_coordinates(gen_df, color="Generator Loss",
                                           color_continuous_scale=px.colors.sequential.haline, width=750)

        disc_plot = px.parallel_coordinates(disc_df, color="Discriminator Loss", width=840,
                                            color_continuous_scale=px.colors.sequential.Agsunset)
        gen_plot.show()
        disc_plot.show()

    # TODO: work out if I am going to find a way to store the times taken for halted optimisation loops
