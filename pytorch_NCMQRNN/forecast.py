import numpy as np

def out_of_sample_forecast(net, df, length, test_length, horizon, quantiles):
    """
    Produces the out of sample forecasts via a rolling window.
    """
    predictions = []
    for t in range(test_length):
        train_target_df = df.loc[t:t+length - 1, "gdp":"gdp"]
        train_target_df.columns = [1]
        train_covariate_df = df.loc[t:t+length - 1, df.columns != "gdp"]

        predict_result = net.predict(train_target_df, train_covariate_df, 1) # quantiles * horizon
        predictions.append(predict_result)

    predictions = np.column_stack(predictions)

    predictions = np.reshape(predictions, [horizon, test_length, len(quantiles)])

    return predictions


