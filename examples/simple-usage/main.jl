# # Description
# This example shows a simple use case for EnsembleKalmanFilters.
#
# First, we import the necessary packages.
using EnsembleKalmanFilters

# Then, we make a filter object holding the hyperparameters for the EnKF.
R = reshape([2.0], (1, 1))
filter = EnKF(R, false, 0)

# We generate an ensemble.

## Two ensemble members with value ±1.
prior_state = [1.0 -1.0]

## Identity observation operator with no noise.
prior_obs_clean = [1.0 -1.0]
prior_obs_noisy = [1.0 -1.0]

# Then we assimialte an observation. Here, we just pick an arbitrary one.
y_obs = [0.0]
posterior = assimilate_data(filter, prior_state, prior_obs_clean, prior_obs_noisy, y_obs)
