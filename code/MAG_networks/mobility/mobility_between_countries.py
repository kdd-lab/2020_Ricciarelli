import json
import numpy as np
import sys

from collections import Counter
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm


def check_country(country):
    if country == 'United States':
        country = 'United States of America'
    elif country == 'Korea':
        country = 'South Korea'
    elif country == 'Russian Federation':
        country = 'Russia'
    elif country == 'Dominican Republic':
        country = 'Dominican Rep.'
    elif country == 'Bosnia and Herzegovina':
        country = 'Bosnia and Herz.'
    elif country == "Lao People's Democratic Republic":
        country = 'Laos'
    elif country == 'Cyprus':
        country = 'N. Cyprus'
    elif country == 'Central African Republic':
        country = 'Central African Rep.'
    elif country == 'South Sudan':
        country = 'S. Sudan'
    elif country == 'Syrian Arab Republic':
        country = 'Syria'
    elif country == 'Viet Nam':
        country = 'Vietnam'

    return country


mobility_dict = dict()

with open(sys.argv[1], 'r') as mobility_file:
    for row in tqdm(mobility_file, desc='READING MOBILITY FILE'):
        from_country, to_country, year, times = row.strip().split('\t')

        if int(year) >= 1980 and int(year) < 2020:
            from_country = check_country(from_country)
            to_country = check_country(to_country)

            if from_country not in mobility_dict:
                mobility_dict[from_country] = dict()
            if to_country not in mobility_dict:
                mobility_dict[to_country] = dict()

            if year not in mobility_dict[from_country]:
                mobility_dict[from_country][year] = Counter()
            if year not in mobility_dict[to_country]:
                mobility_dict[to_country][year] = Counter()

            mobility_dict[from_country][year]['out'] += int(times)
            mobility_dict[to_country][year]['in'] += int(times)

for key_type in ['in', 'out']:
    mobility_matrix = list()

    for country in sorted(mobility_dict):
        row = list()

        for year in np.arange(1980, 2020):
            if str(year) in mobility_dict[country]:
                if key_type in mobility_dict[country][str(year)]:
                    row.append(mobility_dict[country][str(year)][key_type])
                else:
                    row.append(0)
            else:
                row.append(0)

        mobility_matrix.append(row)

    mobility_matrix = np.array(mobility_matrix)

    for idx, year in enumerate(np.arange(1980, 2020)):
        qt = QuantileTransformer()
        n_column = qt.fit_transform(mobility_matrix[:, idx].reshape(-1, 1))

        for x, country in enumerate(sorted(mobility_dict)):
            if str(year) in mobility_dict[country]:
                mobility_dict[country][str(year)][key_type] = n_column[x][0]
            else:
                mobility_dict[country][str(year)] = {key_type: n_column[x][0]}

for country in mobility_dict:
    for year in mobility_dict[country]:
        mobility_dict[country][year]['balance'] = \
            mobility_dict[country][year]['in'] - \
            mobility_dict[country][year]['out']

with open(sys.argv[2], 'w') as mobility_jsonl:
    for country in tqdm(mobility_dict.items(), desc='WRITING MOBILITY JSONL'):
        to_write = dict()
        to_write[country[0]] = country[1]

        json.dump(to_write, mobility_jsonl)
        mobility_jsonl.write('\n')
