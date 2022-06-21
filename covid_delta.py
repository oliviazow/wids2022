from utils import get_datetime_index
import datetime
import pandas as pd


def make_data(byear=2020, eyear=2021, freq='H'):
    def parse(x):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    full_data = get_datetime_index(start=f'{byear}0101', end=f'{eyear}1231',
                                   freq=freq)  # init the full data as datetime index
    full_data['Day'] = full_data.index.date
    full_data['Day'] = full_data.apply(lambda x: '%04d-%02d-%02d' % (x['Day'].year, x['Day'].month, x['Day'].day), axis=1)
    full_data = full_data.reset_index()
    dataset = pd.read_csv('./covid/Covid_Delta_Clark_NV_2020_2021.csv')  #, parse_dates=['DATE'], date_parser=parse)
    dataset.rename(columns={'DATE': 'Day'}, inplace=True)
    full_data = full_data.merge(dataset, how='left', on='Day')
    # full_data = pd.concat([full_data, dataset], axis=1, ignore_index=True, keys=['Day'])
    print(full_data)
    full_data = full_data.drop(columns=['Day'])
    full_data = full_data.set_index('DATE')
    full_data.to_csv('./covid/CovidDelta.csv')
    print(full_data)
    print('Done')


if __name__ == '__main__':
    make_data()

