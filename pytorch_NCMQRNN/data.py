import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset


def read_df(df, label_name, train_size, val_size):
    """
    This function is for reading the sample testing dataframe.

    df: pandas dataframe with the whole dataset
    label_name: target variable name (ex. 'gdp')
    train_size: size of the training set
    val_size: size of the validation set
    """
    target_df = df.loc[:, label_name]
    target_df.columns = [1]
    train_target_df = target_df.iloc[:train_size]
    val_target_df = target_df.iloc[train_size:val_size]
    test_target_df = target_df.iloc[(train_size + val_size):]

    covariate_df = df.iloc[:, df.columns != label_name]
    train_covariate_df = covariate_df.iloc[:train_size]
    val_covariate_df = covariate_df.iloc[train_size:(val_size)]
    test_covariate_df = covariate_df.iloc[(train_size + val_size):]

    return target_df, covariate_df, train_target_df, train_covariate_df, val_target_df, val_covariate_df, test_target_df, test_covariate_df


class NCMQRNN_dataset(Dataset):

    def __init__(self,
                target_df:pd.DataFrame,
                covariate_df:pd.DataFrame, 
                horizon_size:int):
        """
        Prepare NCMQRNN dataset

        target_df: the endogenous variable
        covariate_df: the exogenous variables
        horizon_size: the number of steps to forcast into the future
        """

        self.target_df = target_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size

        print(target_df.shape)
        print(covariate_df.shape)


    def __len__(self):
        return self.target_df.shape[1]

    def __getitem__(self,idx):
        cur_series = np.array(self.target_df.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(self.covariate_df.iloc[:-self.horizon_size, :]) # covariate used in generating hidden states

        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(np.array(self.target_df.iloc[i: self.target_df.shape[0]-self.horizon_size+i, idx]))
        real_vals_array = np.array(real_vals_list) #[horizon_size, seq_len]
        real_vals_array = real_vals_array.T #[seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)
        
        cur_series_tensor = torch.unsqueeze(cur_series_tensor,dim=1) # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate) #[seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor],dim=1)

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, cur_real_vals_tensor


