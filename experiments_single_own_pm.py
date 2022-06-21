from utils import get_datetime_index, MONTHS, AQS_IDs, _add_wind_directions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from data_loader import load_lv_data
from tuning import GATuner
import math


def build_single_site_data_site_pm_only(
        aqs_id, data, with_wind=False, with_lockdown=False,
        byear=2020, eyear=2021, freq='H',
        encoder_name='LabelEncoder', scaler_name='MinMax'):
    full_data = get_datetime_index(
        start=f'{byear}0101', end=f'{eyear}1231', freq=freq)  # init the full data as datetime index
    _, _, site_num = aqs_id.split('-')
    target_col = f'{site_num}_PM25'
    correlated_cols = [target_col]
    # add month
    full_data['Month'] = full_data.index.month
    full_data['Month'] = full_data.apply(lambda x: MONTHS[x['Month']], axis=1)
    n_cat_cols = 1  # only the Month column
    # get site_numbers
    site_numbers = set([col.split('_')[0] for col in correlated_cols])
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
    # add target column
    full_data = full_data.join(data[target_col])
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
    if scaler_name == 'MinMax':
        scalar = MinMaxScaler(feature_range=(0, 1))
        full_data = scalar.fit_transform(full_data)

    return full_data, scalar  # numpy.ndarray, MinMaxScaler


kwarg_names = [
#       'window',
#       'split_ratio',
      'lstm_units',
      'lstm_l2',
      'epochs',
      'batch_size',
      'patience',
      'factor',
      'min_lr',]

map_func = [
    # lambda x: max(1,       min(5,     abs(1 + int(4 * (math.ceil(x)-x))))),         # window
#     lambda x: max(0.1,     min(0.6,   abs(0.1 + 0.5 * (math.ceil(x)-x)))),          # split_ratio
    lambda x: max(NF,      min(3*NF,  abs(NF + int(2*NF*(math.ceil(x)-x))))),        # lstm_units
    lambda x: max(0.01,    min(0.5,   abs(0.01 + 0.49 * (math.ceil(x)-x)))),         # lstm_l2
    lambda x: max(20,      min(100,   abs(20 + 8 * int((math.ceil(x)-x)/0.1)))),     # epochs
    lambda x: max(24,      min(124,   abs(24 + 10 * int((math.ceil(x)-x)/0.1)))),    # batch_size
    lambda x: max(3,       min(100,   abs(3 + int(97 * (math.ceil(x)-x))))),         # patience
    lambda x: max(0.1,     min(0.8,   abs(0.01 + 0.79 * (math.ceil(x)-x)))),         # factor
    lambda x: max(0.00001, min(0.0001, abs(0.00001 + 0.00009 * (math.ceil(x)-x)))),  # min_lr
]


def learn_models(all_data, aqs_id, with_wind=False, with_lockdown=False):
    print('--*--' * 20, aqs_id)
    dataset, scalar = build_single_site_data_site_pm_only(aqs_id, all_data, with_wind=with_wind, with_lockdown=with_lockdown)
    log_ext='single_pm25_only'
    if with_wind:
        log_ext += '_wind'
    if with_lockdown:
        log_ext += '_covid'
    gatuner =GATuner(dataset
                     , scalar
                     , kwarg_names
                     , map_func
                     , 1
                     , aqs_id=aqs_id
                     , log_ext=log_ext
                     , num_generations=5  # 20
                     , sol_per_pop=10  # 100
                     , num_parents_mating=5  # 50
                     , keep_parents=2  # 20
                     , num_genes=len(kwarg_names)
                     , init_range_low=0.0001
                     , init_range_high=0.9999
                     , parent_selection_type="sss"
                     , crossover_type="single_point"
                     , mutation_type="random"
                     , mutation_percent_genes=50
                     , mutation_by_replacement=False
                     , random_mutation_min_val=0.0001
                     , random_mutation_max_val=0.9999
                     , save_best_solutions=False)
    gatuner.tune()
    return


def run(all_data, aqs_ids=AQS_IDs):
    global NF
    for aqs_id in aqs_ids:
        # NF = 2
        # learn_models(all_data, aqs_id, with_wind=False, with_lockdown=False)
        NF = 4
        learn_models(all_data, aqs_id, with_wind=True, with_lockdown=False)
        # NF = 3
        # learn_models(all_data, aqs_id, with_wind=False, with_lockdown=True)
        # NF = 5
        # learn_models(all_data, aqs_id, with_wind=True, with_lockdown=True)

if __name__ == "__main__":
    # pyplot.show(block=False)
    # pyplot.interactive(False)
    all_data = load_lv_data()
    run(all_data, ['32-003-1019'])

