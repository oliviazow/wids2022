import random
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

AQS_IDs = ['32-003-0043',
     '32-003-0071',
     '32-003-0073',
     '32-003-0075',
     '32-003-0298',
     '32-003-0540',
     '32-003-0561',
     '32-003-1019',
     '32-003-1501']
MONTHS = ['', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', ]
WIND_DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
CORR_THRESHOLD = 0.4


def set_seed():
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def get_datetime_index(start='20220603', end='20220604', freq='H'):
    full_range = pd.date_range(start=start, end=end, freq=freq)
    full_range = pd.DataFrame(full_range, columns = ['DATE'])
    full_range = full_range.set_index('DATE')
    return full_range


def series_to_supervised(data, n_in=1, n_pm25=1, n_out=1, dropnan=True):
    '''

    :param data: data built for predicting PM2.5
    :param n_in: window size
    :param n_pm25: num of PM2.5s to predict
    :param n_out: num of future time points to predict, use only 1 in WiDS 2022 experiments
    :param dropnan:
    :return:  pandas.core.frame.DataFrame
    '''
    n_features = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_features)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_features)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_features)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg = agg.dropna(inplace=False)
    # drop the non-target columns for time t, s.t. the last n_pm25 columns are the target columns
    agg = agg.drop(agg.columns[[i for i in range(-n_features, -n_pm25)]], axis=1, inplace=False)
    return agg


def split(values, n_rows=8760, n_pm25=1):
    '''

    :param values: numpy.ndarray, spuervised dataset
    :param n_rows: num of row to be in training dataset
    :param n_pm25: num of PM2.5 (target) columns
    :return: train_X, train_y, test_X, test_y, 4 numpy.ndarray instances
    '''
    train = values[:n_rows, :]
    test = values[n_rows:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-n_pm25], train[:, -n_pm25:]
    test_X, test_y = test[:, :-n_pm25], test[:, -n_pm25:]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, test_X, test_y


def calc_pearson_correlation(data, aqs_id=None):
    df = None
    site_num = ''
    if bool(aqs_id):
        _, _, site_num = aqs_id.split('-')
    for key, col in data.items():
        if site_num in key:
            df = col if df is None else df.join(col, how='inner', on='DATE')
    if df is not None:
        return df.corr()
    return None


def get_correlated_cols_one_pm25(corr, col='1501_PM25', threshold=0.5):
    df = corr[[col]]
    df = df[abs(df[col])>threshold]
    return set(list(df.index))


def get_own_correlated_cols_one_pm25(corr, col='1501_PM25', threshold=0.5):
    site_num, _ = col.split('_')
    a = get_correlated_cols_one_pm25(corr, col=col, threshold=threshold)
    return [e for e in a if site_num in e]


def get_correlated_cols_all_pm25(corr, threshold=0.5):
    cols = set()
    for col in corr.index:
        if 'PM25' in col:
            tmp = get_correlated_cols_one_pm25(corr, col, threshold=threshold)
            # print(col, len(tmp))
            cols |= tmp
    return cols


def _add_wind_directions(full_data, data, site_num):
    wind_dir = f'{site_num}_WindDirection'
    if wind_dir in data:
        wind_dir_data = data[wind_dir]
        wind_dir_data['WindDirCat'] = \
            wind_dir_data.apply(lambda x: None if pd.isna(x[wind_dir])
                                else WIND_DIRS[math.floor(x[wind_dir] % 360 / 22.5)], axis=1)
        wind_dir_data = wind_dir_data.drop(columns=[wind_dir])
        wind_dir_data = wind_dir_data.rename(columns={'WindDirCat': wind_dir})
        full_data = full_data.join(wind_dir_data)
    return full_data


def build_data(
        data, corr, site_num=None, pm_only=False, current_site_only=False, threshold=0.4, with_wind=False, with_lockdown=False,
        byear=2020, eyear=2021, freq='H', encoder_name='LabelEncoder', scalar_name='MinMax'):
    # create datetime index as the init of the full data
    full_data = get_datetime_index(start=f'{byear}0101', end=f'{eyear}1231',
                                   freq=freq)  # init the full data as datetime index
    # get correlated columns and target columns
    if site_num is None:
        correlated_cols = get_correlated_cols_all_pm25(corr, threshold=threshold)
        print(correlated_cols)
        if pm_only:
            correlated_cols = [col for col in correlated_cols if '_PM' in col]
            print(correlated_cols)
        target_cols = [col for col in correlated_cols if col.endswith('_PM25')]
    else:
        target_cols = [f'{site_num}_PM25']
        correlated_cols = get_correlated_cols_one_pm25(corr, col=target_cols[0], threshold=threshold)
        if current_site_only:
            correlated_cols = [e for e in correlated_cols if site_num in e]
    # get site_numbers
    site_numbers = set([col.split('_')[0] for col in correlated_cols])
    # add month
    full_data['Month'] = full_data.index.month
    full_data['Month'] = full_data.apply(lambda x: MONTHS[x['Month']], axis=1)
    n_cat_cols = 1  # the Month column
    if with_wind:
        # add wind directions
        for site_num in site_numbers:
            n_cat_cols += 1
            full_data = _add_wind_directions(full_data, data, site_num)
        # add wind speed
        for site_num in site_numbers:
            wind = f'{site_num}_WindSpeed'
            if wind in data:
                full_data = full_data.join(data[wind])
    if with_lockdown:
        full_data = full_data.join(data['CovidDelta'])
    # add other columns except the target columns
    for col in (set(correlated_cols) - set(target_cols)):
        full_data = full_data.join(data[col])
    # add target columns
    for col in target_cols:
        full_data = full_data.join(data[col])
    # encoding
    if encoder_name == 'LabelEncoder':
        encoder = LabelEncoder()
        for col in full_data.columns[:n_cat_cols]:
            full_data[col] = encoder.fit_transform(full_data[col])
    full_data = full_data.astype('float32')
    # fill in missing values
    full_data = full_data.interpolate()
    # scaling
    scalar = None
    if scalar_name == 'MinMax':
        scalar = MinMaxScaler(feature_range=(0, 1))
        full_data = scalar.fit_transform(full_data)
    # return results
    return full_data, scalar, len(target_cols)  # numpy.ndarray, MinMaxScaler, num of outputs


def test():
    from data_loader import load_lv_data
    data = load_lv_data()
    corr = calc_pearson_correlation(data)
    full_data, scalar, n_pm25 = build_data(data, corr, site_num=None, current_site_only=True, threshold=0.4,
                                           with_wind=False, with_lockdown=False)
    print('=' * 100)
    print(full_data, scalar, n_pm25)
    print(full_data.shape)
    full_data, scalar, n_pm25 = build_data(data, corr, site_num=None, pm_only=True, current_site_only=True,
                                           threshold=0.4,
                                           with_wind=False, with_lockdown=False)
    print('=' * 100)
    print(full_data, scalar, n_pm25)
    print(full_data.shape)
    full_data, scalar, n_pm25 = build_data(data, corr, site_num='1501', current_site_only=True, threshold=0.4,
                                           with_wind=True, with_lockdown=True)
    print('=' * 100)
    print(full_data, scalar, n_pm25)
    print(full_data.shape)
    full_data, scalar, n_pm25 = build_data(data, corr, site_num='1501', current_site_only=False, threshold=0.4,
                                           with_wind=True, with_lockdown=True)
    print('=' * 100)
    print(full_data, scalar, n_pm25)
    print(full_data.shape)
    full_data, scalar, n_pm25 = build_data(data, corr, site_num='1501', current_site_only=False, threshold=0.4,
                                           with_wind=False, with_lockdown=True)
    print('=' * 100)
    print(full_data, scalar, n_pm25)
    print(full_data.shape)


if __name__ == "__main__":
    test()