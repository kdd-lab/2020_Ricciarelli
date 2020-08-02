import json
import numpy as np
import sys
import warnings

from pmdarima.arima import AutoARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import kpss
from statsmodels.tsa.arima_model import ARIMA
from tqdm import tqdm

warnings.filterwarnings("ignore")

time_series_raws = dict()

with open(sys.argv[1], 'r') as time_series_json_file:
    for row in tqdm(time_series_json_file, desc='READING TIME SERIES JSON'):
        creator = json.loads(row)

        time_series_raws.update(creator)

mse = list()

for mag_id in tqdm(time_series_raws, desc='PROCESSING THE SERIES'):
    series = list()

    for year in sorted(time_series_raws[mag_id]):
        series.append(time_series_raws[mag_id][year]['entropy'])

    series = np.array(series)

    kpss_stat, p_value, lags, critical_values = kpss(series, nlags='auto')
    is_stationary = abs(kpss_stat) < abs(critical_values['5%'])

    if not is_stationary:
        print('NOT STATIONARY')

    split = int(len(series) * 0.66)
    train_series, test_series = series[0:split], series[split:len(series)]
    history, predictions = [sample for sample in train_series], list()

    for time_step in range(len(test_series)):
        model = AutoARIMA(start_p=0, d=None, start_q=0, max_p=5, max_d=5,
                          max_q=5, m=1, seasonal=False, test='kpss',
                          suppress_warnings=True, error_action='ignore')
        model.fit(history, disp=0)

        next_step = model.predict(1)

        predictions.append(next_step[0])
        history.append(test_series[time_step])

    mse.append(mean_squared_error(test_series, predictions))

print(np.mean(mse), np.std(mse))
