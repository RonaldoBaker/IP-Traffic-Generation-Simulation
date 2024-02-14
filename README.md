# IP-Traffic-Generation

With the need to accommodate more Internet users each year, we must be able to plan and optimise Internet networks appropriately. To do this, we must be able to simulate Internet Protocol (IP) traffic in different scenarios to evaluate the capacity requirements of growing demands. 

As part of a 3rd-year Electronic & Electrical Engineering project at University College London, it was found that Generative Adversarial Networks (GANs) are becoming a common method of generating synthetic internet traffic.

## Generation of synthetic IP traffic
The GAN model in a Jupyter Notebook can be reviewed in the repository. The training data used comes from a public dataset on Kaggle: [https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps/data]. 

## GAN Optimisation
The GAN was optimised using Optuna, a hyperparameter optimisation framework. Custom pruner functions were written to deal with the lack of multi-objective support.

