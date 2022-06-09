import numpy as np

def calculate_jpr_length(forecast, labels):
    lengths = []

    for idx in range(len(labels[0])):
        uncond_quantile = np.quantile(labels[:,idx], 0.99)
        jpr_idx = forecast[:,idx]
        length = (uncond_quantile - jpr_idx)
        length[np.where(length < 0)] = 0
        lengths.append(np.mean(length))
    return np.mean(lengths)

def calculate_length(forecast, labels):
    uncond_quantile = np.quantile(labels, 0.99)
    lengths = []

    for column in forecast:
        length = (uncond_quantile - column)
        length[np.where(length<0)] = 0
        lengths.append(np.mean(length))

    return lengths