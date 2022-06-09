import numpy as np
from JPR import JPR
quantiles = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
M = 1000 # bootstrap replications
jpr_size = 0.90
test_size = 18
h = 1
path_1 = "data/s2s_GBR.csv"   # GBR gdp rate marginal GaR data
path_2 = "data/s2s_USA.csv"   # USA gdp rate marginal GaR data

data_1 = np.loadtxt(path_1, delimiter= ',')
data_2 = np.loadtxt(path_2, delimiter= ',')

test_estimates_1 = data_1[-test_size-h:-h,:]
test_estimates_2 = data_2[-test_size-h:-h,:]

univariate_pr_quantiles = np.array([test_estimates_1, test_estimates_2])
jpr = JPR.compute_BJPR_jpr(univariate_pr_quantiles, quantiles, jpr_size, M)

np.savetxt("data/jpr.csv", jpr, delimiter= ",")