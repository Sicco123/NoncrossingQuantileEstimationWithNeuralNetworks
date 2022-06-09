from .prepare_data import WindowGenerator
from .train_func import optimize_neural_net, objective_function
from .pred import total_sample_predictions
from .l1_penalization_layer import l1_p, non_cross_transformation
import tensorflow as tf
from keras.layers import LSTM, Flatten, Dropout


class LSTM_nn(tf.keras.Model):
    """
    LSTM, Long Short Term Memory Cell Neural Network, a neural network consisting out of LSTM cells + noncrossing layer.
    """
    def __init__(self, hidden_dim, output_dim, penalty_1 = 0, penalty_2 = 0, dropout = 0.0):
        super().__init__()
        self.layer_1 = LSTM(hidden_dim, return_sequences = False, kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_2 = Dropout(dropout)
        self.layer_3 = tf.keras.layers.Activation('sigmoid')
        self.layer_4 = l1_p(number_of_quantiles=output_dim, penalty_1 = penalty_1, penalty_2 = penalty_2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        self.pred, self.pred_mod = self.layer_4(x) # Output is a dictionary with the objective func input and intermediate results.
        return self.pred

class LstmModel():
    def __init__(self,hyper_params, train_df, val_df, test_df, wide_window_length, label_name):
        ### Hyperparameters
        self.units = hyper_params[0]
        self.lr = hyper_params[1]
        self.p1 = hyper_params[2]
        self.p2 = hyper_params[3]
        self.dropout = hyper_params[4]

        self.wide_window_length = wide_window_length
        self.wide_window = WindowGenerator(
            input_width=wide_window_length, label_width=1, shift=1,
            train_df=train_df, val_df=val_df, test_df=test_df,
            label_columns=label_name)

    def train(self, tau_vec, epochs):
        ### Clear tensorflow
        tf.keras.backend.clear_session()
        model = LSTM_nn(self.units, len(tau_vec), self.p1, self.p2, self.dropout)

        ### Fit model
        loss_fn = lambda x, z: objective_function(x, z, 1-tau_vec)
        self.res = optimize_neural_net(self.wide_window, model, loss_fn, epochs, self.lr)

    def predict(self, df):
        # output
        forecast = total_sample_predictions(df, self.res, self.wide_window_length)
        return forecast