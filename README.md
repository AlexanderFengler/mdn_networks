# mdn_networks

You can run the *train_network.py* script to train the network on a *gaussian pillars* dataset. 
It saves the *loss history*, *training data* and *model predictions* as a set of three graphs. 
One can inspect whether the trained model makes reasonable predictions.

The datasets are random, so expect quality of results to vary.

Currently this trains the *iso* model, which estimates a *mixture of isotropic gaussians* conditioned on the *input features*.


TODO:

1. Add arguments to python scripts --> choose toy dataset, choose mdn from *1d*, *iso*, *full*
2. Add config files
3. Translate to pytorch
4. Use Tensorflow probability / Pytorch Distributions instead of handcoding the loss

