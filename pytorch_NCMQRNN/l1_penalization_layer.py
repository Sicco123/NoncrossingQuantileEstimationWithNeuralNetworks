import torch
import torch.nn as nn
import numpy as np

class l1_p(nn.Module): #This class builds on the layer class of pytorch
    """
    l1 penalization module
    """
    def __init__(self, size_in, number_of_quantiles,**kwargs):
        super(l1_p, self).__init__()
        self.size_in = size_in
        self.number_of_quantiles = number_of_quantiles
        #self.penalty_1 = penalty_1

        self.input_shape = size_in # Equals the number of neurons in the previous layer
        delta_coef_matrix = torch.Tensor(self.input_shape, self.number_of_quantiles)
        self.delta_coef_matrix = nn.Parameter(delta_coef_matrix)
        delta_0_matrix = torch.Tensor(1, self.number_of_quantiles)
        self.delta_0_matrix = nn.Parameter(delta_0_matrix)

        # initialization
        nn.init.constant_(self.delta_coef_matrix, 0)
        nn.init.constant_(self.delta_0_matrix, 0)


    def forward(self, inputs, **kwargs):
        """
        Forward pass through layer.
        """
        ### Build beta matrix
        delta_mat = torch.cat([self.delta_0_matrix, self.delta_coef_matrix], dim=0)
        beta_mat = torch.t(torch.cumsum(torch.t(delta_mat),dim=0))


        delta_vec = delta_mat[1:, 1:] # leave out the first column
        delta_0_vec = delta_mat[0:1, 1:]
        delta_minus_vec = torch.maximum(torch.tensor(0.0), -delta_vec)
        delta_minus_vec_sum = torch.sum(delta_minus_vec, dim=0)
        delta_0_vec_clipped = torch.clip(delta_0_vec, # clip to ensure feasibility of delta_0_vec
                                               min=torch.reshape(delta_minus_vec_sum, delta_0_vec.shape),
                                               max=torch.tensor(
                                                   (np.ones(np.shape(delta_0_vec)) * np.inf), dtype= torch.float64))
        if inputs.dim() != beta_mat.dim():
            inputs = torch.squeeze(inputs, dim = 1)


        predicted_y = torch.add(torch.mm(inputs, beta_mat[1:, :]), beta_mat[0, :])
        predicted_y_modified = torch.mm(inputs, beta_mat[1:, :]) + torch.cumsum(torch.cat([beta_mat[0:1, 0:1],
                                                                                              delta_0_vec_clipped],
                                                                                             dim=1),
                                                                                   dim=1)

        delta_constraint = delta_0_vec_clipped - delta_minus_vec_sum
        delta_l1_penalty = torch.mean(torch.abs(
           delta_0_vec - delta_0_vec_clipped))

        return  predicted_y, delta_l1_penalty #predicted_y_modified

def non_cross_transformation(predicted_y, delta_coef_matrix, delta_0_matrix):
    """
    Function which ensure quantiles to be noncrossing in case of no convergence.
    This output can be far from optimal.
    """

    ### Build beta matrix
    delta_mat = torch.cat([delta_0_matrix, delta_coef_matrix], dim=0)
    beta_mat = torch.t(torch.cumsum(torch.t(delta_mat), dim=0))

    delta_vec = delta_mat[1:, 1:]  # leave out the first column
    delta_0_vec = delta_mat[0:1, 1:]
    delta_minus_vec = torch.maximum(torch.tensor(0.0), -delta_vec)
    delta_minus_vec_sum = torch.sum(delta_minus_vec, dim=0)
    delta_0_vec_clipped = torch.clip(delta_0_vec,  # clip to ensure feasibility of delta_0_vec
                                     min=torch.reshape(delta_minus_vec_sum, delta_0_vec.shape),
                                     max=torch.tensor(
                                         (np.ones(np.shape(delta_0_vec)) * np.inf), dtype=torch.float64))


    part_1 = predicted_y -  beta_mat[0, :]
    transformed_y = part_1 + torch.cumsum(torch.cat([beta_mat[0:1, 0:1],
                                                 delta_0_vec_clipped], dim=1), dim=1)


    return transformed_y


