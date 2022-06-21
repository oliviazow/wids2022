from utils import CORR_THRESHOLD, build_data, calc_pearson_correlation, AQS_IDs
from data_loader import load_lv_data
from tuning import GATuner
import math
import experiments_single_own_pm

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
    lambda x: max(24,      min(324,   abs(24 + 15 * int((math.ceil(x)-x)/0.05)))),   # batch_size
    lambda x: max(3,       min(100,   abs(3 + int(97 * (math.ceil(x)-x))))),         # patience
    lambda x: max(0.1,     min(0.8,   abs(0.01 + 0.79 * (math.ceil(x)-x)))),         # factor
    lambda x: max(0.00001, min(0.0001, abs(0.00001 + 0.00009 * (math.ceil(x)-x)))),  # min_lr
]


def learn_model(data, corr, aqs_id=None, pm_only=False, current_site_only=False, threshold=CORR_THRESHOLD, with_wind=False, with_lockdown=False):
    global NF  # number of features
    if aqs_id is None:  # multi-outputs
        aqs_id = ''
        dataset, scalar, n_pm25 = build_data(
            data, corr, site_num=None, pm_only=pm_only, current_site_only=current_site_only,
            threshold=threshold, with_wind=with_wind, with_lockdown=with_lockdown)
        log_ext = 'all_pm25'
        if pm_only:
            log_ext += '_PmOnly'
        if with_wind:
            log_ext += '_wind'
        if with_lockdown:
            log_ext += '_covid'
    else:  # per aqs_id
        _, _, site_num = aqs_id.split('-')
        dataset, scalar, n_pm25 = build_data(
            data, corr, site_num=site_num, pm_only=pm_only, current_site_only=current_site_only,
            threshold=threshold, with_wind=with_wind, with_lockdown=with_lockdown)
        log_ext = 'single'
        if current_site_only:
            log_ext += '_own'
        else:
            log_ext += '_neighbor'
        if with_wind:
            log_ext += '_wind'
        if with_lockdown:
            log_ext += '_covid'

    NF = dataset.shape[-1] - n_pm25

    gatuner =GATuner(dataset
                     , scalar
                     , kwarg_names
                     , map_func
                     , n_pm25
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


def run():
    data = load_lv_data()
    corr = calc_pearson_correlation(data)

    # print('/* ' + '=' * 100 + ' */')
    # print('   Single-outputs, Current Site Only, PM2.5 Only')
    # print('/* ' + '=' * 100 + ' */')
    # experiments_single_own_pm.run(data)
    #
    # print('/* ' + '=' * 100 + ' */')
    # print('   Multi-outputs, PM Only')
    # print('/* ' + '=' * 100 + ' */')
    # learn_model(data, corr, aqs_id=None, pm_only=True, current_site_only=False,
    #             threshold=CORR_THRESHOLD, with_wind=False, with_lockdown=False)
    #
    # print('/* ' + '=' * 100 + ' */')
    # print('   Multi-outputs, All correlated features')
    # print('/* ' + '=' * 100 + ' */')
    # learn_model(data, corr, aqs_id=None, pm_only=False, current_site_only=False,
    #             threshold=CORR_THRESHOLD, with_wind=False, with_lockdown=False)

    print('/* ' + '=' * 100 + ' */')
    print('   Single output')
    print('/* ' + '=' * 100 + ' */')
    for aqs_id in ['32-003-1019']:  # AQS_IDs:
        print('--*--' * 20)
        print(aqs_id, 'current site only')
        print('--*--' * 20)
        learn_model(data, corr, aqs_id=aqs_id, pm_only=False, current_site_only=True,
                    threshold=CORR_THRESHOLD, with_wind=True, with_lockdown=False)
        print('--*--' * 20)
        print(aqs_id, 'with neighboring sites')
        print('--*--' * 20)
        learn_model(data, corr, aqs_id=aqs_id, pm_only=False, current_site_only=False,
                    threshold=CORR_THRESHOLD, with_wind=True, with_lockdown=False)
        # don't use wind and lockdown data for now


if __name__ == "__main__":
    run()

