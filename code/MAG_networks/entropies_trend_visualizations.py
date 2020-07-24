import json
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from tqdm import tqdm

entropies_dict = dict()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

cl_df = pd.read_csv(sys.argv[2], dtype={'MAG_id': str, 'cluster': int})

for cluster in [1, 2]:
    entropies_per_country, mean_entropies_per_country = dict(), dict()

    for MAG_id in cl_df[cl_df.cluster == cluster]['MAG_id']:
        for year in entropies_dict[MAG_id]:
            country = entropies_dict[MAG_id][year]['affiliation']
            entropy = entropies_dict[MAG_id][year]['entropy']

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

            if country not in entropies_per_country:
                entropies_per_country[country] = defaultdict(list)

            entropies_per_country[country][year].append(entropy)

    for country in entropies_per_country:
        entropies = [entropies_per_country[country][year] for year
                     in entropies_per_country[country]]
        mean_entropies_per_country[np.mean(np.concatenate(entropies))] = \
            country
