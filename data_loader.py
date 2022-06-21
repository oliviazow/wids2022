import datetime
import os
import pandas as pd
from pprint import pprint

def load_lv_data(byear=2020, eyear=2021, null_frac=0.05):
    def parse(x):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    data = {}
    n_hours = (eyear-byear+1) * 365 * 24
    n_hours_missed = int(n_hours * null_frac)
    n_hours_needed = n_hours - n_hours_missed
    for dirname, _, filenames in os.walk('./lasvegas'):
        for filename in filenames:
            p = os.path.join(dirname, filename)
            if filename == 'CovidDelta.csv':
                data['CovidDelta'] = pd.read_csv(p,  parse_dates = ['DATE'], index_col=0, date_parser=parse)
            else:
                base, ext = filename.split('.')
                parts = base.split('_')
                if parts[0] == '32': # measure data
                    if int(parts[-1]) <= byear: # only load parameters that have data from byear
                        dataset = pd.read_csv(p,  parse_dates = ['DATE'], index_col=0, date_parser=parse)
                        num_null = dataset.isna().sum()
                        if len(dataset) >= n_hours_needed and num_null[0] < n_hours_missed:
                            key = '_'.join(parts[2:4])
                            dataset.rename(columns={parts[3]: key}, inplace=True)
                            data[key] = dataset
    return data

if __name__ == "__main__":
    data = load_lv_data(null_frac=0.01)
    print('0.01:', len(data.keys()))
    pprint([col for col in list(data.keys()) if 'PM25' in col])
    data = load_lv_data(null_frac=0.05)
    print('0.05:', len(data.keys()))
    pprint([col for col in list(data.keys()) if 'PM25' in col])
    data = load_lv_data(null_frac=0.1)
    print('0.1:', len(data.keys()))
    pprint([col for col in list(data.keys()) if 'PM25' in col])
    data = load_lv_data(null_frac=0.2)
    print('0.2:', len(data.keys()))
    pprint([col for col in list(data.keys()) if 'PM25' in col])