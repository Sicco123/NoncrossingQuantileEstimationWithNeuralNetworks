import numpy as np

def hit_series(actual, forecast):
    """
    Univariate hit series
    """
    return (actual < forecast) * 1


def jpr_hit_series(actual, forecast_total):
    """
    Joint hit series
    """
    hit_mat = (actual < forecast_total) * 1 # actual T x N matrix, forecast_total TxN matrix
    hit_sum = np.sum(hit_mat, axis=0)
    hit_sum[np.where(hit_sum > 0)] = 1
    return hit_sum