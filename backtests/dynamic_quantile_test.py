import numpy as np
import scipy.stats
from .hit_border import jpr_hit_series, hit_series
def joint_dynamic_quantile_test(forecast_total, actual, jpr_size, hit_lags=0, forecast_lags=1):
    """Dynamic Quantile Test (Engle & Manganelli, 2004)"""
    alpha = 1 - jpr_size

    hits = jpr_hit_series(actual.T, forecast_total.T)
    p, q, n = hit_lags, forecast_lags, len(hits)
    pq = max(p, q - 1)
    y = hits[pq:] - alpha  # Dependent variable
    x = np.zeros((n - pq, 1 + p + q))

    x[:, 0] = 1  # Constant


    for i in range(p):  # Lagged hits
        new_var = hits[pq - (i + 1):-(i + 1)]
        x[:, i+1] = new_var


    for country in range(len(forecast_total[0])):
        for j in range(q):  # Actual + lagged GaR forecast
            if j > 0:
                x[:, 1 + p + j] = forecast_total[:,country][pq - j:-j]
            else:
                x[:, 1 + p + j] = forecast_total[:,country][pq:]

    x = x[:, ~np.all(x == 0, axis=0)] # remove zero columns
    x = np.column_stack({tuple(x[:,idx]) for idx in range(len(x[0]))})


    beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (alpha * (1 - alpha))
    p_dq = 1 - scipy.stats.chi2.cdf(lr_dq, 1 + p + q)


    return lr_dq, p_dq

def dynamic_quantile_test(forecast_total, actual, tau_vec, hit_lags=0, forecast_lags=1):
    """Dynamic Quantile Test (Engle & Manganelli, 2004)"""
    lr_dq_list = []
    p_dq_list = []

    for idx, tau in enumerate(tau_vec):
        alpha = 1 -tau
        forecast = forecast_total[idx,:]
        hits = hit_series(actual, forecast)
        p, q, n = hit_lags, forecast_lags, len(hits)
        pq = max(p, q - 1)
        y = hits[pq:] - alpha  # Dependent variable
        x = np.zeros((n - pq, 1 + p + q))

        x[:, 0] = 1  # Constant

        col_num = 0
        for i in range(p):  # Lagged hits
            if np.sum(hits[pq - (i + 1):-(i + 1)]) > 0:
                x[:,  col_num+1] = hits[pq - (i + 1):-(i + 1)]
                col_num+=1
            else:
                x = np.delete(x, col_num+1, 1)

        for j in range(q):  # Actual + lagged GaR forecast
            if j > 0:
                x[:, 1 + p + j] = forecast[pq - j:-j]
            else:
                x[:, 1 + p + j] = forecast[pq:]

        try:
            np.linalg.inv(np.dot(x.T, x))
        except:
            lr_dq_list.append(100)
            p_dq_list.append(1)
            continue

        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (alpha * (1 - alpha))
        p_dq = 1 - scipy.stats.chi2.cdf(lr_dq, 1 + p + q)

        lr_dq_list.append(lr_dq)
        p_dq_list.append(p_dq)

    return lr_dq_list, p_dq_list