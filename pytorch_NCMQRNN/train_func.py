import torch
import numpy as np
from .data import NCMQRNN_dataset
from .Encoder import Encoder
from .Decoder import GlobalDecoder, Penalizer
from torch.utils.data import DataLoader


def forward_pass(cur_series_covariate_tensor : torch.Tensor,
            cur_real_vals_tensor: torch.Tensor,
            encoder: Encoder,
            gdecoder: GlobalDecoder,
            penalizer: Penalizer,
            device):

    ### Set Type and load to device
    cur_series_covariate_tensor = cur_series_covariate_tensor.double()  # [batch_size, seq_len, 1+covariate_size]
    cur_real_vals_tensor = cur_real_vals_tensor.double()  # [batch_size, seq_len, horizon_size]

    cur_series_covariate_tensor = cur_series_covariate_tensor.to(device)
    cur_real_vals_tensor = cur_real_vals_tensor.to(device)
    encoder.to(device)
    gdecoder.to(device)
    penalizer.to(device)

    ### Reshape
    cur_series_covariate_tensor = cur_series_covariate_tensor.permute(1, 0,
                                                                      2)  # [seq_len, batch_size, 1+covariate_size]
    enc_hs = encoder(cur_series_covariate_tensor)  # [seq_len, batch_size, hidden_size]
    hidden_and_covariate = enc_hs  # [seq_len, batch_size, hidden_size+covariate_size * horizon_size]

    ### Forward pass
    gdecoder_output = gdecoder(hidden_and_covariate)  # [seq_len, batch_size, (horizon_size+1)]
    quantile_size = penalizer.quantile_size
    horizon_size = encoder.horizon_size

    penalizer_output, l1_penalties = penalizer(gdecoder_output)

    seq_len = penalizer_output.shape[0]
    batch_size = 1  # local_decoder_output.shape[1]

    penalizer_output = penalizer_output.view(batch_size, seq_len, horizon_size,
                                             quantile_size)  # [[seq_len, batch_size, horizon_size, quantile_size]]

    return penalizer_output, l1_penalties

def calc_loss(cur_real_vals_tensor: torch.Tensor,
              penalizer_output: torch.Tensor,
              l1_penalties: torch.Tensor,
              penalizer: Penalizer,
            device,
            p1: float):

    total_loss = torch.tensor([0.0], device=device)

    for i in range(penalizer.quantile_size):
      p = penalizer.quantiles[i]
      errors = cur_real_vals_tensor - penalizer_output[:,:,:,i]
      cur_loss = torch.max((p-1)*errors, p*errors ) # CAUTION
      total_loss += torch.sum(cur_loss)
    total_loss += p1 * torch.mean(l1_penalties) # L1 penalty of the l1_penalziation layer
    return total_loss

def weight_optimization_step(data_iter, encoder_optimizer, gdecoder_optimizer, penalizer_optimizer, encoder, gdecoder, penalizer, device, p1):
    epoch_loss_sum = 0.0
    train_sample = 0
    for (cur_series_tensor, cur_real_vals_tensor) in data_iter:
        batch_size = cur_series_tensor.shape[0]
        seq_len = cur_series_tensor.shape[1]
        horizon_size = 4  # cur_covariate_tensor.shape[-1]
        train_sample += batch_size * seq_len * horizon_size
        encoder_optimizer.zero_grad()
        gdecoder_optimizer.zero_grad()
        penalizer_optimizer.zero_grad()

        # forward pass
        output, l1_penalties = forward_pass(cur_series_tensor, cur_real_vals_tensor,
                                            encoder, gdecoder, penalizer, device)
        # calc loss
        loss = calc_loss(cur_real_vals_tensor, output, l1_penalties, penalizer, device, p1)
        loss.backward()

        encoder_optimizer.step()
        gdecoder_optimizer.step()
        penalizer_optimizer.step()

        epoch_loss_sum += loss.item()

    return epoch_loss_sum / train_sample

def early_stopping_validation(val_iter, encoder, gdecoder, penalizer, device, epoch_loss_mean, min_valid_loss, i, train_len, name, p1):
    valid_loss = 0.0
    for (cur_series_tensor, cur_real_vals_tensor) in val_iter:

        horizon_size = 4  # cur_covariate_tensor.shape[-1]
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            cur_series_tensor, cur_real_vals_tensor = cur_series_tensor.cuda(), cur_real_vals_tensor.cuda()

        # forward pass
        output, l1_penalties = forward_pass(cur_series_tensor, cur_real_vals_tensor,
                                            encoder, gdecoder, penalizer, device)

        # calc loss
        loss = calc_loss(cur_real_vals_tensor[:, train_len:, :], output[:, train_len:, :, :], l1_penalties, penalizer,
                         device, p1)

        # Calculate Loss
        valid_loss += loss.item()
    if (i + 1) % 50 == 0:
        print(
            f'Epoch {i + 1} \t\t Training Loss: {epoch_loss_mean} \t\t Validation Loss: {valid_loss / (horizon_size * (len(cur_real_vals_tensor[0, :, 0]) - train_len))}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({round(min_valid_loss / (horizon_size * (len(cur_real_vals_tensor[0, :, 0]) - train_len)), 6)}--->{round(valid_loss / (horizon_size * (len(cur_real_vals_tensor[0, :, 0]) - train_len)), 6)}) \t Saving The Model')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(encoder.state_dict(), f'{name}_saved_encoder.pth')
        torch.save(gdecoder.state_dict(), f'{name}_saved_gdecoder.pth')
        torch.save(penalizer.state_dict(), f'{name}_saved_penalizer.pth')
    return min_valid_loss

def train_fn(encoder:Encoder, 
            gdecoder: GlobalDecoder,
            penalizer: Penalizer,
            dataset: NCMQRNN_dataset,
            val_data: NCMQRNN_dataset,
            lr: float, 
            batch_size: int,
            num_epochs: int,
            p1: float,
            name: str,
            device,
            ):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=lr)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(),lr=lr)
    penalizer_optimizer = torch.optim.Adam(penalizer.parameters(), lr=lr)

    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    val_iter = DataLoader(dataset= val_data, batch_size = batch_size, shuffle = True, num_workers=0)

    train_len= len(dataset.series_df)
    min_valid_loss = np.inf
    for i in range(num_epochs):

        epoch_loss_mean = weight_optimization_step(data_iter, encoder_optimizer, gdecoder_optimizer, penalizer_optimizer, encoder,
                                 gdecoder, penalizer, device, p1)

        encoder.eval()  # Optional when not using Model Specific layer
        gdecoder.eval()
        penalizer.eval()

        min_valid_loss = early_stopping_validation(val_iter, encoder, gdecoder, penalizer, device, epoch_loss_mean, min_valid_loss, i, train_len, name, p1)

    encoder.load_state_dict(torch.load(f'{name}_saved_encoder.pth'))
    gdecoder.load_state_dict(torch.load(f'{name}_saved_gdecoder.pth'))
    penalizer.load_state_dict(torch.load(f'{name}_saved_penalizer.pth'))
    encoder.eval()
    gdecoder.eval()
    penalizer.eval()
