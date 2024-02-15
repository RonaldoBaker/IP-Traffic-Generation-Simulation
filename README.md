# IP Traffic Simulation

This repository details an IP Traffic Simulation created for the 3rd-year Electronic & Electrical Engineering project at University College London in 2024. This project is split into two parts: generation of synthetic IP traffic on the flow level, and the simulation of that synthetic IP traffic. 

## Traffic Generation
### Generation of synthetic IP traffic
A Generative Adversarial Network (GAN), a type of neural network, was used to generate synthetic IP traffic on the flow level. There was a focus on 3 flow attributes: flow size in bytes, flow duration in seconds, and interarrival time between flows entering the network in seconds. PyTorch was used for creating the GAN. The training data inputted into one of the neural networks comes from a public dataset on Kaggle: [https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps/data]. The training data of interest had to be converted into a PyTorch tensor to use in the model, which is saved in the file with the `.pt` extension. 

### Model Optimisation
The GAN was optimised using Optuna, a hyperparameter optimisation framework. The GAN contain two neural networks, hence it is classed as a multi-objective optimisation. At the time of development, Optuna lacks sufficient support for multi-objective optimisation. Custom pruner functions were written to deal with the lack of multi-objective optimisation support in Optuna. 

## Traffic Simulation
To have more control of the simulation, it was built from scratch by following the flow of control for the next-event time-advance approach outlined in "Simulation Modeling and Analysis" by Averill M. Law.

The network topology emulated in the simulation comes from UKNet 
![UKNet, a telephone network located in the UK, internet network follows the same topology](https://github.com/RonaldoBaker/IP-Traffic-Simulation/blob/main/UKNet.png)
