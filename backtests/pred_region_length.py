import numpy as np

def calculate_jpr_length(forecast, labels):
    """
    Calculate the average length of the joint prediction regions.
    The length is the distance between the 0.99 unconditional quantile and the conditional quantile of interest.
    """

    lengths = []

    for idx in range(len(labels[0])):
        uncond_quantile = np.quantile(labels[:,idx], 0.99)
        jpr_idx = forecast[:,idx]
        length = (uncond_quantile - jpr_idx)
        length[np.where(length < 0)] = 0
        lengths.append(np.mean(length))
    return np.mean(lengths)

def calculate_length(forecast, labels):
    """
    Calculate the average length of the prediction region.
    The length is the distance between the 0.99 unconditional quantile and the conditional quantile of interest.
    """

    uncond_quantile = np.quantile(labels, 0.99)
    lengths = []

    for column in forecast:
        length = (uncond_quantile - column)
        length[np.where(length<0)] = 0
        lengths.append(np.mean(length))

    return lengths