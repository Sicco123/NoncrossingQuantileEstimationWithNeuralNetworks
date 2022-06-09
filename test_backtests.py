import numpy as np
import pandas as pd
from backtests import coverage, dynamic_quantile_test, pred_region_length, tick_loss

quantiles = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
test_size =18
h = 1 # horizion
path_labels_1 = "data/GBR.csv"
path_labels_2 = "data/USA.csv"
labels_1 = pd.read_csv(path_labels_1, index_col = 0)['gdp']
labels_2 = pd.read_csv(path_labels_2, index_col = 0)['gdp']
test_labels_1 = labels_1[-test_size:]
test_labels_2 = labels_2[-test_size:]

### marginal GaR backtests
path_mgar_1 = "data/s2s_GBR.csv"
mgar_estimates = np.loadtxt(path_mgar_1, delimiter = ",")[-test_size-h:-h,:].T # align correctly

# compute coverage
cover = coverage.calculate_coverage(mgar_estimates, test_labels_1)
print(cover)

# compute length
length = pred_region_length.calculate_length(mgar_estimates, test_labels_1)
print(length)

# compute tick loss
tl = tick_loss.quantile_risk(mgar_estimates, test_labels_1, quantiles)
print(tl)

# compute dynamic quantile tests
dq_1 = dynamic_quantile_test.dynamic_quantile_test(mgar_estimates, test_labels_1, quantiles, 0, 0)
print(dq_1) # statistic , pvalue
dq_2 = dynamic_quantile_test.dynamic_quantile_test(mgar_estimates, test_labels_1, quantiles, 4, 0)
print(dq_2)
dq_3 = dynamic_quantile_test.dynamic_quantile_test(mgar_estimates, test_labels_1, quantiles, 0, 4)
print(dq_3)

### joint GaR backtests
path_jgar = "data/jpr.csv"
jgar_estimates = np.loadtxt(path_jgar, delimiter = ",").T # align correctly
jpr_size = 0.90
test_labels = np.column_stack([test_labels_1, test_labels_2])

# compute coverage
cover = coverage.calculate_joint_coverage(jgar_estimates, test_labels)
print(cover)

# compute length
length = pred_region_length.calculate_jpr_length(jgar_estimates, test_labels)
print(length)

# compute dynamic quantile tests
dq_1 = dynamic_quantile_test.joint_dynamic_quantile_test(jgar_estimates, test_labels, jpr_size , 0, 0)
print(dq_1) # statistic , pvalue
dq_2 = dynamic_quantile_test.joint_dynamic_quantile_test(jgar_estimates, test_labels, jpr_size, 4, 0)
print(dq_2)
dq_3 = dynamic_quantile_test.joint_dynamic_quantile_test(jgar_estimates, test_labels, jpr_size, 0, 4)
print(dq_3)







