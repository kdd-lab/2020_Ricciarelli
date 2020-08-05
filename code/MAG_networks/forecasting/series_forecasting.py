import numpy as np
import pandas as pd
import sys
import warnings

from pmdarima.arima import AutoARIMA
from pmdarima.model_selection import RollingForecastCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.filterwarnings("ignore")

time_series_df = pd.read_csv(sys.argv[1])
mses = list()

for index, row in tqdm(time_series_df.iterrows(), desc='FORECASTING',
                       total=len(time_series_df)):
    change_points, chunks = list(), list()

    for year in row['change_points'].strip('][').split(', '):
        change_points.append(int(year.strip("'")))

    for idx, change_point in enumerate(change_points):
        if idx == 0:
            chunks.append([row[str(y)] for y in np.arange(1980, change_point)
                          if not np.isnan(row[str(y)])])
        else:
            chunks.append([row[str(y)] for y in np.arange(
                          change_points[idx - 1], change_point) if not
                          np.isnan(row[str(y)])])
    chunks.append([row[str(y)] for y in np.arange(
                  change_points[-1], 2020) if not
                  np.isnan(row[str(y)])])

    model = AutoARIMA(start_p=0, d=None, start_q=0, max_p=5, max_d=5,
                      max_q=5, m=1, seasonal=False, stationary=False,
                      information_criterion='aic', alpha=0.05,
                      test='kpss', stepwise=True,
                      suppress_warnings=True, error_action='ignore')

    dataset = np.array(chunks[0])
    row_mse = list()

    for idx, chunk in enumerate(chunks[1:]):
        dataset = np.concatenate((dataset, chunk))
        predictions, weights = list(), list()

        if len(chunk) == 1:
            if len(dataset[:-1]) > 1:
                model.fit(dataset[:-1])
                predictions.append(model.predict(n_periods=1)[0])
                weights.append(len(dataset[:-1]))
        else:
            initial = sum([len(c) for c in chunks[:idx + 1]])

            if initial > 1:
                cv = RollingForecastCV(initial=initial)

                for train_series, test_series in cv.split(dataset):
                    try:
                        model.fit(dataset[train_series])
                        predictions.append(model.predict(n_periods=1)[0])
                        weights.append(len(train_series))
                    except Exception as e:
                        pass

                model.fit(dataset[:-1])
                predictions.append(model.predict(n_periods=1)[0])
                weights.append(len(dataset[:-1]))

        if len(predictions) == len(chunk):
            error = mean_squared_error(chunk, predictions,
                                       sample_weight=weights
                                       if len(weights) != 1 else None)
            row_mse.append(error)

    if len(row_mse) != 0:
        mses.append(np.mean(row_mse))
