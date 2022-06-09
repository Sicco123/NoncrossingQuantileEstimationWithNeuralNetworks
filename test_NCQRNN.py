import numpy as np
import pandas as pd
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow_NCQRNN.prepare_data import WindowGenerator, train_val_test_split
from tensorflow_NCQRNN.LSTM import LstmModel
from tensorflow_NCQRNN.CNN import CnnModel
from tensorflow_NCQRNN.DNN import DnnModel
#from tests import calculate_coverage, calculate_length, quantile_risk, dynamic_quantile_test, hit_border
from tensorflow_NCQRNN.train_func import optimize_neural_net, objective_function
#from plot_funcs import visualize_coverage, visualize_qr_risk, create_latex_table, t_test, make_long_table_countries
import matplotlib.pyplot as plt




### Magic numbers
h = 4 # forecast_horizon
train_frac = 0.7   # train test split fraction
val_frac = 0.2
test_frac = 0.1
tau_vec = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
epochs = 500
data_path = "data/USA.csv"

df = pd.read_csv(f'data/USA.csv', index_col = 0)
train_df, val_df, test_df = train_val_test_split(df, train_frac, val_frac)
test_split = len(train_df) + len(val_df)


### DNN
hyper_params = [20, 0.007, 40, 0] # units, learning_rate, p1, p2
label_name = ['gdp']

model = DnnModel(hyper_params, train_df, val_df, test_df, label_name)
model.train(tau_vec, epochs)
forecasts = model.predict(df)
print(forecasts)

### CNN
hyper_params = [12, 0.01, 60, 0] # units, learning_rate, p1, p2
conv_width = 4
label_name = ['gdp']

model = CnnModel(hyper_params, train_df, val_df, test_df, conv_width, label_name)
model.train(tau_vec, epochs)
forecasts = model.predict(df)
print(forecasts)

### LSTM
hyper_params = [32, 0.008, 30, 0, 0] # units, learning_rate, p1, p2, dropout
wide_window_length = 4
label_name = ['gdp']

model = LstmModel(hyper_params, train_df, val_df, test_df, wide_window_length, label_name)
model.train(tau_vec, epochs)
forecasts = model.predict(df)
print(forecasts)