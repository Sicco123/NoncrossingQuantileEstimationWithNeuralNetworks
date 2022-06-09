import tensorflow as tf

def quantile_risk(predicted_y, objective_y, quantiles):
    """
    Compute quantile loss for multiple quantiles.
    """

    ### prepare quantiles
    quantile_length = len(quantiles)
    quantile_tf = tf.convert_to_tensor(quantiles, dtype='float32')
    quantile_tf_tiled = tf.repeat(tf.transpose(quantile_tf), [len(objective_y)])

    ### prepare objective value
    objective_y = tf.squeeze(tf.cast(objective_y, dtype='float32'))
    output_y_tiled = tf.tile(objective_y, [quantile_length])

    ### prepare predicted values
    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), [-1])  # output_y_tiled.shape

    ### objective function
    diff_y = output_y_tiled - predicted_y_tiled
    quantile_loss = tf.reduce_mean(diff_y * (quantile_tf_tiled - (tf.sign(-diff_y) + 1) / 2))
    return quantile_loss

def objective_function(objective_y_stack, input, quantiles):
     """
     Wrapper function which makes it possible to estimate a combined quantile loss for estimating multiple quantiles
     of multiple timeseries at once.
     """
     ret_quantile_loss = 0
     if len(objective_y_stack.shape) == 2: # Calculate quantile risk for one objective series. --> one country
         predicted_y = input
         objective_y = objective_y_stack
         ret_quantile_loss = quantile_risk(predicted_y, objective_y, quantiles)

     elif len(objective_y_stack.shape) == 3: # Calculate quantile risk for multiple series at once. --> more countries
        window_length = objective_y_stack.shape[1]
        for idx in range(window_length):
             predicted_y = input[:, idx, :]
             objective_y = objective_y_stack[:, idx, :]
             quantile_loss = quantile_risk(predicted_y, objective_y, quantiles)

             ret_quantile_loss += quantile_loss

     return ret_quantile_loss


def optimize_neural_net(window, model, lambda_objective_function, max_deep_iter, learning_rate, patience = 75):
    """
    Core of the optimization. We use keras to optimze the neural nets. There are lots of parameters one might
    to change.
    """
    #### Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience= patience,
                                                      mode='min',
                                                      restore_best_weights = False)

    #####  Compile keras model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=lambda_objective_function,
                  steps_per_execution=1,
                  run_eagerly = False # set true to return numpy arrays. When True much slower.
                  )

    ##### Fit keras model on the dataset
    model.fit(window.train,
              epochs=max_deep_iter,
              verbose=0,
              validation_data=window.val,
              shuffle=False,
              callbacks = [early_stopping],
              workers=4,
              use_multiprocessing=True,
              )
    return model