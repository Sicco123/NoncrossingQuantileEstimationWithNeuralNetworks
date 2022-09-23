import numpy as np
import pandas as pd
import scipy
from arch.univariate import arch_model
import matplotlib.pyplot as plt

class garchOneOne(object):

    def __init__(self, rates):
        self.T = np.shape(rates)[0]
        self.N = np.shape(rates)[1]
        self.rates = rates  # T x N series. T is time, N is number of individuals
        self.coefficients = self.composite_garch_optimization()
        self.output, self.sigma_2 = self.garch_filter(self.coefficients)
        self.conditional_volatility = self.sigma_2
        self.resid = self.compute_resid()


    def garch_filter(self, parameters):
        "Returns the variance expression of a GARCH(1,1) process."

        # Slicing the parameters list
        omega = parameters[:-4] # Length N
        mu = parameters[-4]
        phi = parameters[-3]
        alpha = parameters[-2]
        beta = parameters[-1]

        # Initializing an empty array
        output = np.zeros((self.T, self.N))
        sigma_2 = np.zeros((self.T, self.N))

        # Filling the array, if i == 0 then uses the long term variance.
        for t in range(self.T):
            if t == 0:
                output[t, :] = np.zeros(output[t,:].shape)
                sigma_2[t,:] = omega / (1 - alpha - beta)
            else:
                output[t,:] = mu + phi*output[t-1,:]
                sigma_2[t,:] = omega + alpha * (-output[t-1,:]+self.rates[t - 1]) ** 2 + beta * sigma_2[t - 1,:]

        return output, sigma_2

    def garch_loglikehihood(self, parameters):
        "Defines the log likelihood sum to be optimized given the parameters."

        output, sigma_2 = self.garch_filter(parameters)
        loglikelihood = - np.sum(-np.log(sigma_2[1:,:]) - np.divide((-output[1:,:]+self.rates[1:,:]) ** 2, sigma_2[1:,:]))/self.N
        return loglikelihood


    def composite_garch_optimization(self):
        "Optimizes the log likelihood function and returns estimated coefficients"

        # Parameters initialization
        parameters = np.append(np.ones(self.N)*.1, [0,0.7, .92, .05])

        # Set bounds
        bounds = list(((.001,4) for i in range(self.N))) + list(((-10,10),(-0.99,0.99),(.001,1),(0.001,1)))

        # Set constraints
        constraint = lambda x: -(x[-1]+x[-2]-1)
        con = {'type': 'ineq', 'fun': constraint}

        opt = scipy.optimize.minimize(self.garch_loglikehihood, parameters,
                                      bounds=bounds, constraints= con)

        return opt.x

    def compute_resid(self):
        resid = self.rates - self.output
        return resid



def get_jpr_bootstrap_sample(self, y, h, bootstrap_choices):
    "Get joint prediction region based bootstrap."

    model = arch_model(y, mean='AR', lags=1)        # NEEDS TO BECOME COMPOSITE ESTIMATION # Better to store residuals of models and use here.
    model = model.fit(disp='off')
    y = y.values
    residuals = model.resid[1:].values

    conditional_volatility = model.conditional_volatility[1:].values

    nlized_resid = residuals / np.sqrt(conditional_volatility)
    time = np.arange(1, len(nlized_resid), 1)
    bootstrap_resid = nlized_resid[bootstrap_choices]

    phi0 = model.params['Const']
    phi1 = model.params['gdp[1]']
    omega = model.params['omega']
    alpha = model.params['alpha[1]']
    beta = model.params['beta[1]']

    mu_fut = phi0 + phi1 * y[-1]
    sigma_2_fut = model.conditional_volatility.values[-1]

    for j in range(h):
        vol_fut = omega + alpha * np.square(y[-1] - mu_fut) + beta * sigma_2_fut
        mu_fut = phi0 + phi1 * y[-1]
        y_fut = mu_fut + bootstrap_resid / np.sqrt(vol_fut)

    sig_boot = np.sqrt(vol_fut)
    mu_boot = mu_fut
    y_boot = y_fut

    return y_boot, mu_boot, sig_boot

def estimate_garch_quantiles(params, conditional_volatility, resid, y, tau_vec, h, bootstrap_choices):
    residuals = resid[1:]
    conditional_volatility = conditional_volatility[1:]

    nlized_resid = (residuals - np.mean(residuals)) / np.sqrt(conditional_volatility)
    time = np.arange(1,len(nlized_resid),1)

    bootstrap_resid = nlized_resid[bootstrap_choices]

    omega = params[0]
    mu = params[1]
    phi = params[2]
    alpha = params[3]
    beta = params[4]


    sigma_2_fut = conditional_volatility[-1]

    y_bootstrap = []

    yt1 = y[-2]
    yt = y[-1]
    last_sigma = sigma_2_fut

    for j in range(h):
            vol_fut = omega + alpha*np.square(yt-mu - phi*yt1) + beta*last_sigma
            y_fut = mu + phi*yt + bootstrap_resid/np.sqrt(vol_fut)
            y_bootstrap.append(y_fut)

            yt1 = yt
            yt = y_fut
            last_sigma = vol_fut

    y_bootstrap = np.append(np.sqrt(conditional_volatility), y_bootstrap)
    quantiles = np.quantile(y_bootstrap, tau_vec)

    return quantiles

def GARCH_precidtions(train_horizon, test_df, df, total_df, idx, tau_vec, h, bootstrap_choices):
    test_results = []
    for t in range(0,len(test_df)):
        res = garchOneOne(total_df.loc[t:train_horizon+t, :].values)
        params = np.array([res.coefficients[idx], res.coefficients[-4], res.coefficients[-3], res.coefficients[-2], res.coefficients[-1]])#

        conditional_volatility = res.conditional_volatility[:,idx]
        resid = res.resid[:,idx]

        predicted_quantiles = estimate_garch_quantiles(params, conditional_volatility, resid, df['gdp'].values[:train_horizon+t],tau_vec,  h, bootstrap_choices[:,t])
        test_results.append(predicted_quantiles)

    return np.column_stack(test_results)