import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset


def read_df(df, train_size, val_size):
    """
    This function is for reading the sample testing dataframe
    """
    target_df = df.iloc[:, 4:5]
    target_df.columns = [1]
    covariate_df = df.iloc[:, df.columns != "gdp"]
    train_target_df = df.iloc[:train_size, 4:5]
    train_target_df.columns = [1]
    val_target_df = df.iloc[:(train_size + val_size), 4:5]
    val_target_df.columns = [1]
    test_target_df = df.iloc[(train_size + val_size):, 4:5]
    test_target_df.columns = [1]
    train_covariate_df = df.iloc[:train_size, df.columns != "gdp"]
    val_covariate_df = df.iloc[:(train_size + val_size), df.columns != "gdp"]
    return target_df, covariate_df, train_target_df, train_covariate_df, val_target_df, val_covariate_df


class NCMQRNN_dataset(Dataset):
    
    def __init__(self,
                series_df:pd.DataFrame,
                covariate_df:pd.DataFrame, 
                horizon_size:int):
        
        self.series_df = series_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size



    def __len__(self):
        return self.series_df.shape[1]

    def __getitem__(self,idx):
        cur_series = np.array(self.series_df.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(self.covariate_df.iloc[:-self.horizon_size, :]) # covariate used in generating hidden states

        covariate_size = self.covariate_df.shape[1]
        #next_covariate = np.array(self.covariate_df.iloc[1:-self.horizon_size+1,:]) # covariate used in the MLP decoders

        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(np.array(self.series_df.iloc[i: self.series_df.shape[0]-self.horizon_size+i, idx]))
        real_vals_array = np.array(real_vals_list) #[horizon_size, seq_len]
        real_vals_array = real_vals_array.T #[seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)
        
        cur_series_tensor = torch.unsqueeze(cur_series_tensor,dim=1) # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate) #[seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor],dim=1)

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, cur_real_vals_tensor


