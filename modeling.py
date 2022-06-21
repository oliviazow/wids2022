from utils import set_seed, series_to_supervised, split
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math
from pprint import pprint


class EarlyStopCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.MAE_HISTORY = []
    def on_epoch_end(self, epoch, logs={}):
        mae = logs.get('mae')
        self.MAE_HISTORY.append(mae)
        if len(self.MAE_HISTORY) > 5:
            self.MAE_HISTORY.pop(0)
            # print(self.MAE_HISTORY)
            stop = True
            for i in range(1, 5):
                if self.MAE_HISTORY[i] < self.MAE_HISTORY[i-1]:
                    stop = False
                    break
            if stop:
                print("MAE does not decrease in last 5 epochs... Stopping training")
                self.model.stop_training = True

class SetSeedCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        set_seed()


def build_model(train_X, train_y, test_X, test_y, plot=False, verbose=0
                , lstm_units=73, lstm_l2=0.01, epochs=50, batch_size=200
                , patience=50, factor=0.5, min_lr=0.00001
                ):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(lstm_units, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True
                   , activation='relu', recurrent_activation='sigmoid'
                   # , kernel_regularizer=None if l1 is None else tf.keras.regularizers.l1(l1)
                   , recurrent_regularizer=tf.keras.regularizers.l2(lstm_l2)
                  ))
    model.add(tf.keras.layers.Dense(train_y.shape[1]))
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'accuracy'])
    # fit network
    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=patience, factor=factor, min_lr=min_lr)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size
                        , validation_split = 0.2
                        # , validation_data=(test_X, test_y)
                        , verbose=verbose, shuffle=False
                        , callbacks=[lr_reduced, EarlyStopCallback(), SetSeedCallback()]
                       )
    # plot history
    if plot:
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
    return model


def evaluate_forecast_one_result(ytest, yhat):
    mse_ = tf.keras.losses.MeanSquaredError()
    mae_ = tf.keras.losses.MeanAbsoluteError()
    mape_ = tf.keras.losses.MeanAbsolutePercentageError()
    r2 = tfa.metrics.r_square.RSquare()
    mae = mae_(ytest,yhat)
    mse = mse_(ytest,yhat)
    mape = mape_(ytest,yhat)
    r2.update_state(ytest, yhat)
    result = r2.result()
    return {'mae': mae.numpy(),              # Mean Absolute Error
            'rmse': math.sqrt(mse.numpy()),  # Root Mean Squared Error
            'mape': mape.numpy(),            # Mean Absolute Percentage Error
            'R^2': result.numpy()            # Coefficient of determination
           }

def evaluate_forecast_sklearn(ytest, yhat):
    return {'mae': mean_absolute_error(ytest, yhat),              # Mean Absolute Error
            'rmse': math.sqrt(mean_squared_error(ytest, yhat)),  # Root Mean Squared Error
            'mape': mean_absolute_percentage_error(ytest, yhat),            # Mean Absolute Percentage Error
            'R^2': r2_score(ytest, yhat)            # Coefficient of determination
           }

def evaluate_forecast(y_test_inverse, yhat_inverse):
    if y_test_inverse.shape != yhat_inverse.shape:
        raise Exception('Y Test and Y Hat do not have the same shape!')
    if len(y_test_inverse.shape) == 1:
        return evaluate_forecast_one_result(y_test_inverse, yhat_inverse)
    res = {'mae':  0,
           'rmse': 0,
           'mape': 0,
           'R^2':  0}
    n_results = y_test_inverse.shape[1]
    for i in range(n_results):
        res_ = evaluate_forecast_one_result(y_test_inverse[:, i], yhat_inverse[:, i])
        pprint(res_)
        for key in res_:
            res[key] += res_[key]
    for key in res:
            res[key] /= n_results
    return res

def predict(model, test_X, test_y, scaler, n_features, n_pm25):
    # make a prediction
    yhat = model.predict(test_X)
    # print('yhat shape:', yhat.shape)
    yhat = yhat.reshape((yhat.shape[0], yhat.shape[-1]))
    # print('yhat shape:', yhat.shape)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # print('test_X shape:', test_X.shape)
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, :(n_features-n_pm25)]), axis=1)
    # print('inv_yhat shape:', inv_yhat.shape)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    # print('inv_yhat shape:', inv_yhat.shape)
    inv_yhat = inv_yhat[:,:n_pm25]
    # print('inv_yhat shape:', inv_yhat.shape)
    # invert scaling for actual
    # print('test_y shape:', test_y.shape)
    test_y = test_y.reshape((len(test_y), n_pm25))
    # print('test_y shape:', test_y.shape)
    inv_y = np.concatenate((test_y, test_X[:, :(n_features-n_pm25)]), axis=1)
    # print('inv_y shape:', inv_y.shape)
    inv_y = scaler.inverse_transform(inv_y)
    # print('inv_y shape:', inv_y.shape)
    inv_y = inv_y[:,:n_pm25]
    # print('inv_y shape:', inv_y.shape)
    # print('inv_y shape:', inv_y.shape)
    # print('inv_yhat shape:', inv_yhat.shape)
    return inv_y, inv_yhat

def run(  values, scaler, n_pm25=1, plot=False, verbose=0
        , window=1, split_ratio=0.15
        , lstm_units=73, lstm_l2=0.01
        , epochs=50, batch_size=500
        , patience=50, factor=0.5, min_lr=0.00001
        ):
    set_seed()
    # print(f'Data shape: {values.shape}')
    n_features = values.shape[1]
    # print(f'# Features: {n_features}, # of PM2.5s: {n_pm25}')
    reframed = series_to_supervised(values, n_in=window)
    train_X, train_y, test_X, test_y = split(reframed.values, int(len(reframed)*(1-split_ratio)), n_pm25)
    # print(f'train_X shape: {train_X.shape}')
    # print(f'train_y shape: {train_y.shape}')
    # print(f'test_X shape: {test_X.shape}')
    # print(f'test_y shape: {test_y.shape}')
    model = build_model(
          train_X, train_y, test_X, test_y, plot=plot, verbose=verbose
        , lstm_units=lstm_units, lstm_l2=lstm_l2, epochs=epochs, batch_size=batch_size
        , patience=patience, factor=factor, min_lr=min_lr)
    # print('Completed model training!')
    inv_y, inv_yhat = predict(model, test_X, test_y, scaler, n_features, n_pm25)
    measures = evaluate_forecast(inv_y, inv_yhat)
    measures_sklearn = evaluate_forecast_sklearn(inv_y, inv_yhat)
    print('-'*5, 'SKLearn Metrics:')
    pprint(measures_sklearn)
    print('-'*5)
    return measures