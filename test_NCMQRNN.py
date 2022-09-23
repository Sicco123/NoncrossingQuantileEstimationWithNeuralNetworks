from pytorch_NCMQRNN.NCMQRNN import NCMQRNN
from pytorch_NCMQRNN.data import NCMQRNN_dataset, read_df
import pandas as pd
import numpy as np
import torch

### Magic Numbers
horizon_size = 4
hidden_size = 12
tau_vec = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
quantile_size = len(tau_vec)
columns = [1]
dropout = 0.0
layer_size = 2
by_direction = False
lr = 1e-2
batch_size= 1
num_epochs = 400
context_size = 10
p1 = 50
val_size =0.2
save_nn_name = "usa_nn"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
data_path = "data/USA.csv"
train_size = 123
val_size = 35
test_size = 18

### Prepare Data
df = pd.read_csv(data_path, index_col = 0)
prepared_data = read_df(df, train_size, val_size)
target_df = prepared_data[0]
covariate_df = prepared_data[1]
train_target_df = prepared_data[2]
train_covariate_df = prepared_data[3]
train_val_target_df = prepared_data[4]
train_val_covariate_df = prepared_data[5]

dset = NCMQRNN_dataset(train_target_df,train_covariate_df,horizon_size )
train_dataset = NCMQRNN_dataset(train_target_df,train_covariate_df,horizon_size)
train_val_data = NCMQRNN_dataset(train_val_target_df, train_val_covariate_df, horizon_size)

covariate_size = train_covariate_df.shape[1]

### Set Model
net = NCMQRNN(horizon_size, hidden_size, 1 - tau_vec, columns, dropout, layer_size, by_direction, lr, batch_size, num_epochs, context_size, covariate_size, p1, save_nn_name, device)

### Train Model
net.train(train_dataset, train_val_data)

### Get Quantiles
col_name = 1
quantile_predictions = net.predictions(target_df, covariate_df, col_name)

np.savetxt('s2s_USA.csv',quantile_predictions[:,0,:], delimiter = ",")



