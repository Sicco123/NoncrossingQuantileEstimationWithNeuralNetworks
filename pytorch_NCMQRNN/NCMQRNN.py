import torch
from .Encoder import Encoder
from .Decoder import GlobalDecoder, Penalizer
from .train_func import train_fn
from .data import NCMQRNN_dataset
from pytorch_NCMQRNN.l1_penalization_layer import non_cross_transformation

class NCMQRNN(object):
    """
    This class holds the encoder and the global decoder and local decoder.
    """
    def __init__(self, 
                horizon_size:int, 
                hidden_size:int, 
                quantiles:list,
                columns:list, 
                dropout:float,
                layer_size:int,
                by_direction:bool,
                lr:float,
                batch_size:int, 
                num_epochs:int,
                context_size:int,
                covariate_size:int,
                p1: float,
                name: str,
                device):
        print(f"device is: {device}")
        self.device = device
        self.horizon_size = horizon_size
        self.quantile_size = len(quantiles)
        self.quantiles = quantiles
        self.lr = lr 
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.covariate_size = covariate_size
        quantile_size = self.quantile_size
        self.p1 = p1
        self.name = name
        self.encoder = Encoder(horizon_size=horizon_size,
                               covariate_size=covariate_size,
                               hidden_size=hidden_size, 
                               dropout=dropout,
                               layer_size=layer_size,
                               by_direction=by_direction,
                               device=device)
        
        self.gdecoder = GlobalDecoder(hidden_size=hidden_size,
                                    covariate_size=covariate_size,
                                    horizon_size=horizon_size,
                                    context_size=context_size)

        self.penalizer = Penalizer(quantile_size=quantile_size,
                                    context_size=context_size,
                                    quantiles=quantiles,
                                    horizon_size=horizon_size)
        self.encoder.double()
        self.gdecoder.double()
        self.penalizer.double()
    
    def train(self, dataset:NCMQRNN_dataset, val_data:NCMQRNN_dataset):
        
        train_fn(encoder=self.encoder, 
                gdecoder=self.gdecoder,
                penalizer = self.penalizer,
                dataset=dataset,
                val_data = val_data,
                lr=self.lr,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                p1 = self.p1,
                name = self.name,
                device=self.device)
        print("training finished")

    def load(self):

        self.encoder.load_state_dict(torch.load(f'{self.name}_saved_encoder.pth'))
        self.encoder.eval()

        self.gdecoder.load_state_dict(torch.load(f'{self.name}_saved_gdecoder.pth'))
        self.gdecoder.eval()

        self.penalizer.load_state_dict(torch.load(f'{self.name}_saved_penalizer.pth'))
        self.encoder.eval()
    
    def predict(self,train_target_df, train_covariate_df, col_name):

        input_target_tensor = torch.tensor(train_target_df[[col_name]].to_numpy())
        full_covariate = train_covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)


        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2) #[seq_len, 1, 1+covariate_size]
            print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            outputs = self.encoder(input_target_covariate_tensor) #[seq_len,1,hidden_size]
            hidden = torch.unsqueeze(outputs[-1],dim=0) #[1,1,hidden_size]

            #next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) # [1,1, covariate_size * horizon_size]

            print(f"hidden shape: {hidden.shape}")
            gdecoder_input = hidden #[1,1, hidden + covariate_size* horizon_size]
            gdecoder_output = self.gdecoder(gdecoder_input) #[1,1,(horizon_size+1)*context_size]


            local_decoder_input = gdecoder_output #[1, 1,(horizon_size+1)*context_size + covariate_size * horizon_size]

            penalizer_output, loss = self.penalizer(gdecoder_output)


            penalizer_output = penalizer_output.view(self.horizon_size,self.quantile_size)

            delta_coef_matrix = None
            delta_0_matrix = None
            i = 1
            for parameter in self.penalizer.parameters():
                if i % 2 != 0:
                    delta_coef_matrix = parameter
                    i+=1
                elif i % 2 == 0 :
                    delta_0_matrix = parameter
                    penalizer_output[int(i/2-1),:] = non_cross_transformation(penalizer_output[int(i/2)-1,:], delta_coef_matrix, delta_0_matrix)

            output_array = penalizer_output.cpu().numpy()


            return output_array

    def predictions(self,target_df, covariate_df, col_name):

        input_target_tensor = torch.tensor(target_df[[col_name]].to_numpy())
        full_covariate = covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)


        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2) #[seq_len, 1, 1+covariate_size]
            print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            outputs = self.encoder(input_target_covariate_tensor) #[seq_len,1,hidden_size]
            #hidden = torch.unsqueeze(outputs,dim=0) #[1,1,hidden_size]

            #next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) # [1,1, covariate_size * horizon_size]

           # print(f"hidden shape: {hidden.shape}")
            gdecoder_input = outputs #[1,1, hidden + covariate_size* horizon_size]
            gdecoder_output = self.gdecoder(gdecoder_input) #[1,1,(horizon_size+1)*context_size]


            #local_decoder_input = gdecoder_output #[1, 1,(horizon_size+1)*context_size + covariate_size * horizon_size]

            penalizer_output, loss = self.penalizer(gdecoder_output)


            penalizer_output = penalizer_output.view(len(target_df), self.horizon_size,self.quantile_size)

            delta_coef_matrix = None
            delta_0_matrix = None
            i = 1
            for parameter in self.penalizer.parameters():
                if i % 2 != 0:
                    delta_coef_matrix = parameter
                    i+=1
                elif i % 2 == 0 :
                    delta_0_matrix = parameter
                    penalizer_output[int(i/2-1),:] = non_cross_transformation(penalizer_output[int(i/2)-1,:], delta_coef_matrix, delta_0_matrix)

            output_array = penalizer_output.detach().cpu().numpy()

            return output_array