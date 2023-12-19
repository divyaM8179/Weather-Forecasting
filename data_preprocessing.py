import pandas as pd
from data_analysis import data_analysis
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats

def data_preprocess():
    data = data_analysis()
    data.rename(columns={' _conds': 'conditions', ' _dewptm': 'dewpoint',
                     ' _fog': 'fog', ' _hail': 'hail', ' _heatindexm': 'heatindex', ' _hum': 'humidity',
                     ' _precipm': 'precipitation', ' _pressurem': 'pressure', ' _rain': 'rain', ' _snow': 'snow',
                     ' _tempm': 'temp', ' _thunder': 'thunder', ' _tornado': 'tornado', ' _vism': 'visibility',
                     ' _wdird': 'wdirdegrees', ' _wdire': 'winddirection', ' _wgustm': 'windgust',
                     ' _windchillm': 'windchill', ' _wspdm': 'windspeed'}, inplace=True)
    data.drop(columns=['precipitation', 'windchill', 'heatindex', 'windgust'], inplace=True)
    print(f'dataset shape (rows, columns) - {data.shape}')
    data = data.replace(to_replace = -9999, value = np.nan)
    data.ffill(inplace=True)
    print(data[data.isnull()].count())
    return data

data_preprocess()
