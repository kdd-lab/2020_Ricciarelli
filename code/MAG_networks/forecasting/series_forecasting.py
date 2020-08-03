import json
import numpy as np
import sys
import warnings

from pmdarima.arima import AutoARIMA
from pmdarima.model_selection import cross_val_score, RollingForecastCV
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

    model = AutoARIMA(start_p=0, d=None, start_q=0, max_p=5, max_d=5,
                      max_q=5, m=1, seasonal=False, stationary=False,
                      information_criterion='aic', alpha=0.05, test='kpss',
                      stepwise=True, suppress_warnings=True,
                      error_action='ignore')
    cv = RollingForecastCV()

    cv_score = cross_val_score(model, series, scoring='mean_squared_error',
                               cv=cv)

    mse.append(np.mean(cv_score))
