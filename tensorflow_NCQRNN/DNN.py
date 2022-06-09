from .prepare_data import WindowGenerator
from .train_func import optimize_neural_net, objective_function
from .pred import total_sample_predictions
from .l1_penalization_layer import l1_p, non_cross_transformation
import tensorflow as tf
from keras.layers import LSTM, Flatten, Dropout, Dense, Conv1D

class DNN(tf.keras.Model):
    """
    DNN, Dense Neural Network, a neural network consisting out of two dense layers + noncrossing layer.
    """
    def __init__(self, hidden_dim_1, hidden_dim_2, output_dim, penalty_1 = 0, penalty_2 = 0):
        super().__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(units=hidden_dim_1, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L2(penalty_2))  # Choose correct number of units
        self.layer_3 = Dense(units=hidden_dim_2, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_4 = l1_p(number_of_quantiles=output_dim, penalty_1 = penalty_1, penalty_2 = penalty_2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        self.pred, self.pred_mod = self.layer_4(x) # Output is a dictionary with the output and the enforced noncrossing output.
        return self.pred

class DnnModel():
    def __init__(self,hyper_params, train_df, val_df, test_df, label_name):
        ### Hyperparameters
        self.units = hyper_params[0]
        self.lr = hyper_params[1]
        self.p1 = hyper_params[2]
        self.p2 = hyper_params[3]

        self.window_width = 1
        # set window
        self.window = WindowGenerator(
            input_width=self.window_width, label_width=1, shift=1,
            train_df=train_df, val_df=val_df, test_df=test_df,
            label_columns=label_name)

    def train(self, tau_vec, epochs):
        ### Clear tensorflow
        tf.keras.backend.clear_session()

        ### Initialize
        model = DNN(self.units, self.units, len(tau_vec), self.p1, self.p2)
        ### Fit model
        loss_fn = lambda x, z: objective_function(x, z, 1 - tau_vec)
        self.res = optimize_neural_net(self.window, model, loss_fn, epochs, self.lr)

    def predict(self, df):
        # output
        forecast = total_sample_predictions(df, self.res, self.window_width)

        return forecast
