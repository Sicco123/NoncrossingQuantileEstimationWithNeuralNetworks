import numpy as np
from .hit_border import jpr_hit_series

def calculate_coverage(forecast, labels):
    "Calculate the coverage by counting the data inside the prediction interval"
    labels_tile = np.tile(labels, [len(forecast),1])
    hits = (forecast > labels_tile)*1
    violation_ratios = np.mean(hits, axis=1)
    return 1 - violation_ratios

def calculate_joint_coverage(forecast, labels):
    "Calculate the joint coverage, counts the data outside the prediction interval"
    hit_array = jpr_hit_series(labels, forecast)
    return 1-np.mean(hit_array)
