# Policy Analysis using Synthetic Controls in Continuous Time

This is a python implementation of algorithms and experiments presented in the paper [*"Policy Analysis using Synthetic Controls in Continuous Time"*](https://arxiv.org/abs/2102.01577). 

Counterfactual estimation using synthetic controls is one of the most successful recent methodological developments in causal inference. Despite its popularity, the current description only considers time series aligned across units and synthetic controls expressed as linear combinations of observed control units. 

The goal of this project is to introduce a continuous-time alternative, called **Neural Continuous Synthetic Controls (NC-SC)**, that models the latent counterfactual path explicitly using the formalism of controlled differential equations. This model is directly applicable to the general setting of irregularly-aligned multivariate time series and may be optimized in rich function spaces.

## Dependencies
The only significant dependencies are python 3.6 or later, pytorch and the torchcde package.

## First steps
To get started, check our tutorials which will guide you through NC-SC from the beginning. 

