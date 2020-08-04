import json
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm

to_append = list()

with open(sys.argv[1], 'r') as time_series_json_file:
    time_range = np.arange(1980, 2020)

    for row in tqdm(time_series_json_file, desc='READING TIME SERIES JSON'):
        creator, new_row = json.loads(row), list()

        for mag_id in creator:
            years, change_points = [y for y in sorted(creator[mag_id])], list()

            for year in time_range:
                if str(year) in years:
                    entropy = creator[mag_id][str(year)]['entropy']

                    new_row.append(entropy if entropy != -0.0 else 0.0)
                else:
                    new_row.append(np.nan)

            for idx, year in enumerate(sorted(creator[mag_id])):
                if idx != 0:
                    if creator[mag_id][year]['affiliation'] != \
                       creator[mag_id][years[idx - 1]]['affiliation']:
                        change_points.append(int(year))

            new_row.append(change_points)
            new_row.append(mag_id)
            to_append.append(new_row)

time_series_dataframe = pd.DataFrame(data=to_append, columns=[str(year) for
                                     year in np.arange(1980, 2020)] +
                                     ['change_points'] + ['mag_id'])

time_series_dataframe.to_csv(path_or_buf=sys.argv[2], index=False)
