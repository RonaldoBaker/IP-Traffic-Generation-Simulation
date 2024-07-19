# Import dependencies
import torch
from gan_trainer import GANTrainer

# Constants

NUM_EPOCHS = 5
LEARNING_RATE = 0.02
BATCH_SIZE = 1300
TRAINING_DATA = "../training-data/training_data_tensor.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 77
# PATH = "../trained-generator/generator.pt"
# BINS = 50


def main():
    trainer = GANTrainer(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, TRAINING_DATA, DEVICE, RANDOM_SEED)
    trainer.initialise()
    trainer.train()
    trainer.report()


if __name__ == "__main__":
    main()

# TODO: Add visualisation method to the GANTrainer class, add save and load model method as well

# Visualising Generator and Discriminator loss


# plt.plot(epoch_count, generator_loss_values, label="Generator loss")
# plt.plot(epoch_count, discriminator_loss_values, label="Discriminator loss")
# plt.title("Generator and Discriminator loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
#
# # Generating synthetic data and save generator model
#
# torch.manual_seed(RANDOM_SEED)
# latent_space_samples = torch.randn((TRAINING_DATA_LENGTH, 2), device = DEVICE)
# generated_samples = generator(latent_space_samples)
# generated_samples = generated_samples.cpu().detach().numpy()
# synth_df = pd.DataFrame(generated_samples, columns = ["Duration", "Size"])
# print(generated_samples)
#
# # Visualise synthetic data
#
# plt.scatter(synth_df["Size"], synth_df["Duration"])
# plt.title("Relation betweeen flow duration and flow size in synthetic data")
# plt.xlabel("Flow Size (bytes)")
# plt.ylabel("Flow Duration (s)")
# plt.show()
#
# # ----------------------------------------------
# fig, ax = plt.subplots(1, 2)
#
# plt.subplot(1, 2, 1)
# plt.hist(synth_df["Duration"], bins = BINS)
# plt.title("Distribution of synthetic flow duration")
# plt.xlabel("Flow duration (s)")
# plt.ylabel("Frequency")
#
# plt.subplot(1, 2, 2)
# plt.hist(synth_df["Size"], bins = BINS)
# plt.title("Distribution of synthetic flow size")
# plt.xlabel("Flow size (bytes)")
# plt.ylabel("Frequency")
#
# fig.tight_layout()
# plt.show()

# Save Generator model


# while True:
#     answer = input("Would you like to save the Generator Model? (Y or N)")
#     if answer.isalpha():
#         if answer.upper() == "N":
#             break
#         elif answer.upper() == "Y":
#             torch.save(generator.state_dict(), PATH)
#             print(f"Generator saved to {PATH}")
#             break
#         else:
#             print("Enter Y or N")
#     else:
#         print("No numbers allowed")