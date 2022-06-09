import numpy as np
from .l1_penalization_layer import non_cross_transformation

def predictions(df, model, split, window_length):
    """
    One step predictons based for neural network.
    """
    weights = model.layers[-1].weights
    store_feasible_output = []

    for t in range(len(df) - window_length):
        output = model.predict(df[t:t + window_length].values[np.newaxis])
        feasible_output = non_cross_transformation(output, weights[0], weights[1]).numpy()[0]
        feasible_output = np.squeeze(feasible_output)
        store_feasible_output.append(feasible_output)

    forecast = np.column_stack(store_feasible_output[split - window_length:])
    return forecast

def total_sample_predictions(df, model, window_length):
    """
    Sample predictions for the complete interval.
    """

    weights = model.layers[-1].weights
    store_feasible_output = []

    for t in range(len(df) - window_length):
        output = model.predict(df[t:t + window_length].values[np.newaxis])
        feasible_output = non_cross_transformation(output, weights[0], weights[1]).numpy()[0]
        feasible_output = np.squeeze(feasible_output)
        store_feasible_output.append(feasible_output)

    forecast = np.column_stack(store_feasible_output)
    return forecast



