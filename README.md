# Dynamic SpatioTemporal Attention Model (DSAM)

We propose a novel interpretable deep learning framework that learns goal-specific functional connectivity matrix directly from time series and employs a specialized graph neural network for the final classification. 

Our model, DSAM, leverages:
- Temporal causal convolutional networks to capture the temporal dynamics in both low- and high-level feature representations.
- A temporal attention unit to identify important time points.
- A self-attention unit to construct the goal-specific connectivity matrix.
- A novel variant of graph neural network to capture the spatial dynamics for downstream classification.

## Implementation

- Specify the model under: `conf/config.yaml`
- All other specifications and hyperparameters can be found under: `/conf/*`
- Run the main model: python __main__.py 

## Baselines

This repository also contains numerous baseline implementations for brain network analysis. Part of the code was adapted from the [Brain Network Transformer Repository](https://github.com/Wayfear/BrainNetworkTransformer).
