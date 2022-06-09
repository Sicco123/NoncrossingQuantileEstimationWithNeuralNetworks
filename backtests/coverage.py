import numpy as np
from .hit_border import jpr_hit_series

def calculate_coverage(forecast, labels):
    labels_tile = np.tile(labels, [len(forecast),1])
    hits = (forecast < labels_tile)*1
    violation_ratios = np.mean(hits, axis=1)
    return violation_ratios

def calculate_joint_coverage(forecast, labels):
    hit_array = jpr_hit_series(labels, forecast)
    return 1-np.mean(hit_array)
