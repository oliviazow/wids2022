import requests
import json
import pandas as pd
import os
from pprint import pprint

AQS_IDs_LV_Near = [
    '32-003-0075',
    '32-003-0073',
    '32-003-0071',
    '32-003-2003',
    '32-003-1502',
    '32-003-0561',
    '32-003-1501',
    '32-003-0540',
    '32-003-0043',
    '32-003-0298',
    '32-003-0044',
    '32-003-0299',]
AQS_IDs_LV_Far = [
    '32-003-0602',
    '32-003-1019',]
AQS_IDs_LV = AQS_IDs_LV_Near + AQS_IDs_LV_Far
AQS_API_EMAIL = 'o.zhao.0001@gmail.com'
AQS_API_KEY = 'sandbird21'
PARAMETERS = {
    'O3': ('44201', 'Ozone'),
    'SO2': ('42401', 'Sulfur dioxide'),
    'SO2Max': ('42406', 'SO2 max 5-min avg'),
    'CO': ('42101', 'Carbon monoxide'),
    'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
    'PM25': ('88101', 'PM2.5 - Local Conditions'),
    'PM10': ('81102', 'PM10 Total 0-10um STP'),
    'WindSpeed': ('61103', 'Wind Speed - Resultant'),
    'WindDirection': ('61104', 'Wind Direction - Resultant'),
    'Temp': ('62101', 'Outdoor Temperature'),
    'Pressure': ('64101', 'Barometric pressure'),
    'RH': ('62201', 'Relative Humidity '),
}
PARAMETERS_LV = \
{'32-003-0043': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0044': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0071': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0073': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0075': {'CO': ('42101', 'Carbon monoxide'),
                 'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
                 'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0298': {'CO': ('42101', 'Carbon monoxide'),
                 'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0299': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0540': {'CO': ('42101', 'Carbon monoxide'),
                 'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
                 'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Pressure': ('64101', 'Barometric pressure'),
                 'RH': ('62201', 'Relative Humidity '),
                 'SO2': ('42401', 'Sulfur dioxide'),
                  # 'SO2Max': ('42406', 'SO2 max 5-min avg'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0561': {'CO': ('42101', 'Carbon monoxide'),
                 'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-0602': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-1019': {'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-1501': {'CO': ('42101', 'Carbon monoxide'),
                 'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-1502': {'CO': ('42101', 'Carbon monoxide'),
                 'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')},
 '32-003-2003': {'CO': ('42101', 'Carbon monoxide'),
                 'NO2': ('42602', 'Nitrogen dioxide (NO2)'),
                 'O3': ('44201', 'Ozone'),
                 'PM10': ('81102', 'PM10 Total 0-10um STP'),
                 'PM25': ('88101', 'PM2.5 - Local Conditions'),
                 'Temp': ('62101', 'Outdoor Temperature'),
                 'WindDirection': ('61104', 'Wind Direction - Resultant'),
                 'WindSpeed': ('61103', 'Wind Speed - Resultant')}}


def build_url_monitor_by_site(aqs_id, bdate, edate, email, key):
    state, county, site = aqs_id.split('-')
    url = f'https://aqs.epa.gov/data/api/monitors/bySite?email={email}&key={key}&bdate={bdate}&edate={edate}&state={state}&county={county}&site={site}'
    return url


def build_url_data_by_site(aqs_id, param, bdate, edate, duration, email, key):
    ''' Only 1 year of data is permitted '''
    state, county, site = aqs_id.split('-')
    url = f'https://aqs.epa.gov/data/api/sampleData/bySite?email={email}&key={key}&param={param}&bdate={bdate}&edate={edate}&state={state}&county={county}&site={site}&duration={duration}'
    return url


def rest_request(url):
    response_API = requests.get(url)
    # print(response_API.status_code)
    data = response_API.text
    parse_json = json.loads(data)
    return parse_json


def get_parameters(aqs_id_list, bdate, edate, email, key):
    parameters = None
    for aqs_id in aqs_id_list:
        data = rest_request(build_url_monitor_by_site(aqs_id, bdate, edate, email, key))
        df = pd.json_normalize(data, record_path =['Data'])
        # print(df.head(1))
        df = df[['state_code', 'county_code', 'site_number', 'parameter_code', 'parameter_name', 'open_date', 'latitude', 'longitude']]
        df = df.assign(aqs_id=lambda x: x.state_code + '-' + x.county_code + '-' + x.site_number)
        df = df.drop(labels=['state_code', 'county_code', 'site_number'], axis=1)
        parameters = df if parameters is None else pd.concat([parameters, df])
    parameters = parameters.drop_duplicates()
    return parameters


def get_datetime_index(start='20220603', end='20220604', freq='H'):
    full_range = pd.date_range(start=start, end=end, freq=freq)
    full_range = pd.DataFrame(full_range, columns = ['DATE'])
    full_range = full_range.set_index('DATE')
    return full_range


def get_data(aqs_id='32-003-1501', param='88101', param_alias='PM25', bdate='20220101', edate='20220101', duration='1', email=AQS_API_EMAIL, key=AQS_API_KEY):
    ''' bdate to edate must be less than 1 year '''
    url = build_url_data_by_site(aqs_id, param, bdate, edate, duration, email, key)
    json_data = rest_request(url)
    df = pd.json_normalize(json_data, record_path =['Data'])
    df = df[['date_local', 'time_local', 'sample_measurement']]
    df = df.assign(DATE=lambda x: pd.to_datetime(x.date_local + ' ' + x.time_local, format='%Y-%m-%d %H:%M'))
    df = df[['DATE', 'sample_measurement']]
    df = df.rename(columns={'sample_measurement': param_alias})
    df = df.set_index('DATE')
    return df


def get_data_for_years(aqs_id='32-003-1501', param='88101', param_alias='PM25', byear=2019, eyear=2022, duration='1', email=AQS_API_EMAIL, key=AQS_API_KEY):
    data = []
    start_year = byear
    for year in range(byear, eyear+1):
        bdate = '%s0101' % year
        edate = '%s1231' % year
        try:
            res = get_data(aqs_id=aqs_id, param=param, param_alias=param_alias, bdate=bdate, edate=edate, duration=duration, email=email, key=key)
            # print(param_alias, len(res))
            data.append(res)
        except Exception as e:
            start_year = year
    if data:
        data = pd.concat(data)
        return data, start_year
    return None, start_year


def get_lv_data_by_site(aqs_id='32-003-1501', byear=2019, eyear=2022, email=AQS_API_EMAIL, key=AQS_API_KEY):
    data = {}
    for alias, param in PARAMETERS_LV[aqs_id].items():
        data[alias] = get_data_for_years(aqs_id=aqs_id, param=param[0], param_alias=alias, byear=byear, eyear=eyear, duration='1', email=email, key=key)
        if data[alias][0] is not None:
            print(alias, 'start_year =', data[alias][1], len(data[alias][0]))
        else:
            print(alias, 'No data')
    return data


def get_lv_data(byear=2019, eyear=2022, email=AQS_API_EMAIL, key=AQS_API_KEY):
    data = {}
    for aqs_id in PARAMETERS_LV.keys():
        print('-' * 10, aqs_id)
        data[aqs_id] = get_lv_data_by_site(aqs_id=aqs_id, byear=byear, eyear=eyear, email=email, key=key)
    return data


def get_and_save_lv_parameters(aqs_id_list=AQS_IDs_LV, bdate='20190101', edate='20221231', email=AQS_API_EMAIL, key=AQS_API_KEY):
    if not os.path.isdir('./raw_data'):
        os.mkdir('./raw_data')
    parameters_lv = get_parameters(aqs_id_list, bdate, edate, email, key)
    parameters_lv.to_csv('./raw_data/parameters_las_vegas.csv')
    pprint(parameters_lv)
    return


def get_and_save_lv_data(byear=2019, eyear=2022):
    if not os.path.isdir('./raw_data'):
        os.mkdir('./raw_data')
    data = get_lv_data(byear=byear, eyear=eyear)
    for aqs_id in data:
        for alias, value in data[aqs_id].items():
            print(alias, len(value))
            if value[0] is not None:
                f_path = f"./raw_data/{aqs_id.replace('-', '_')}_{alias}_from_{value[1]}.csv"
                value[0].to_csv(f_path)
    print('Data saved!')
    return


def get_lv_all_data():
    get_and_save_lv_parameters()
    get_and_save_lv_data()


if __name__ == '__main__':
    get_lv_all_data()
