import numpy as np

def quantile_risk(forecast, labels, quantiles):
    """
    The quantile risk, aka tick loss, is the same as the objective function
    of the neural nets. This function is written in numpy only.
    """
    ### prepare quantiles
    quantile_length = len(quantiles)
    quantile_tiled = np.repeat(quantiles.T, [len(labels)])

    ### prepare objective value
    objective_y = np.squeeze(labels)
    output_y_tiled = np.tile(labels, [quantile_length])

    ### prepare predicted values
    predicted_y_tiled = np.reshape(forecast, [-1]).T

    ### objective function
    diff_y = output_y_tiled - predicted_y_tiled
    quantile_losses = diff_y * (quantile_tiled - (np.sign(-diff_y) + 1) / 2)
    individual_quantile_loss = np.reshape(quantile_losses, (quantile_length, len(labels)))

    quantile_loss_per_quantile = np.mean(individual_quantile_loss, axis = 1)
    return quantile_loss_per_quantile