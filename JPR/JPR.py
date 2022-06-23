from sstudentt import SST
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def inverse_skewed_t_cdf(mu, sigma, nu, tau, tau_vec):
    """
    Get the quantiles of the skewed t distribution given the parameters.
    """
    dist = SST(mu=mu, sigma=sigma, nu=nu, tau=tau)
    skew_t_quantiles = dist.q(tau_vec)
    return skew_t_quantiles


def skewed_t_objective(params, tau_vec, est_quantiles):
    """
    Calculate the mse between the predicted quantiles and the proposed skewed t dist.
    """
    mu = params[0]
    sigma = params[1]
    nu = params[2]
    tau = params[3]
    skew_t_quantiles = inverse_skewed_t_cdf(mu, sigma, nu, tau, tau_vec)
    mean_squared_error = np.mean(np.square(est_quantiles - skew_t_quantiles))
    return mean_squared_error


def get_skewed_t_parameters(tau_vec, est_quantiles):
    """
    Optimizer function which returns the optimal skewed t parameters
    """

    x_ini = [0, 1, 1, 5]
    bounds = [(None, None), (0.001, None), (None, None), (2.001, None)]
    options = {'maxiter': 100, 'disp': False}

    res = minimize(skewed_t_objective, x0=x_ini, args=(tau_vec, est_quantiles), bounds=bounds, options=options)
    return res.x


def get_bootstrap_sample(params, M):
    """
    Get normalized draws of the skewed student t distribution
    """
    dist = SST(*params)
    random_draws = dist.r(M)
    normalized_draws = (random_draws - params[0]) / (params[1])
    return normalized_draws


def compute_BJPR_jpr(univariate_pr_quantiles, tau_vec, jpr_size, M):
    """
    Computes the joint prediction region out of marginal prediction regions for the BJPR.
    To do this we first fit a skewed t distribution.
    univariate_pr_quantiles - #timeseries x test_size x #quantiles
    """

    jpr_store = np.zeros(
        [len(univariate_pr_quantiles[0, :, 0]), len(univariate_pr_quantiles[:, 0, 0])])  # test_size  x  columns
    for step in range(len(univariate_pr_quantiles[0, :, 0])):
        bootstrap_list = []  # store bootstrap samples
        mean_list = []  # store individual conditional mean of skewed t
        sig_list = []  # store individual conditional variance of skewed t

        for jdx, quantiles in enumerate(univariate_pr_quantiles[:, step, :]):
            sst_params = get_skewed_t_parameters(tau_vec, quantiles)
            bootstrap_replications = get_bootstrap_sample(sst_params, M)  # draws samples form rescaled skewed_t

            mean = sst_params[0]
            sig = sst_params[1]

            bootstrap_list.append(bootstrap_replications)
            mean_list.append(mean)
            sig_list.append(sig)

        bootstrap_array = np.column_stack(bootstrap_list)
        mean_array = np.array(mean_list)
        sig_array = np.array(sig_list)

        u_array = np.min(bootstrap_array, axis=0)  # take min of all individual the samples
        d_value = np.quantile(u_array, 1 - jpr_size)  # take the 1- jpr_size quantile
        joint_pr = mean_array + d_value * (sig_array)  # scale back to space of original data
        jpr_store[step, :] = joint_pr  # store the joint pr

    return np.column_stack(jpr_store)
