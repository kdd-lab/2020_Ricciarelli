import json
import matplotlib.pyplot as plt
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

    top_5_higher_entropies = sorted(mean_entropies_per_country,
                                    reverse=True)[:5]
    top_5_lower_entropies = sorted(mean_entropies_per_country)[:5]

    top_5_higher_entropies = [mean_entropies_per_country[mean] for mean in
                              top_5_higher_entropies]
    top_5_lower_entropies = [mean_entropies_per_country[mean] for mean in
                             top_5_lower_entropies]

    for l in [top_5_higher_entropies, top_5_lower_entropies]:
        fig, axes = plt.subplots(nrows=5, ncols=1, constrained_layout=True)

        suptitle = 'Top 5 Countries with higher Entropy - Cluster {}' \
            .format(cluster) if l == top_5_higher_entropies else \
            'Top 5 Countries with lower Entropy - Cluster {}'.format(cluster)

        fig.suptitle(suptitle, fontsize=10)

        for x in np.arange(0, 5):
            stats = dict()

            for year in entropies_per_country[l[x]]:
                stats[year] = {
                    'mean': np.mean(entropies_per_country[l[x]][year]),
                    'std': np.std(entropies_per_country[l[x]][year]),
                    'samples': len(entropies_per_country[l[x]][year])}

            ys = list()

            for year in np.arange(1980, 2020):
                if str(year) in stats:
                    ys.append(stats[str(year)]['mean'])
                else:
                    ys.append(np.nan)

            axes[x, 0].plot(np.arange(1980, 2020), ys, linewidth=2)
            axes[x, 0].set_xlim(1979, 2020)
            axes[x, 0].set_xticks(np.arange(1980, 2020, 10))
            axes[x, 0].set_xticks(np.arange(1980, 2020), minor=True)
            axes[x, 0].set_title(l[x], fontsize=8)
            axes[x, 0].set_xlabel('Year', fontsize=8)
            axes[x, 0].set_ylabel('Entropy', fontsize=8)

        save_title = 'top_5_higher_entropies_cluster_{}'.format(cluster) \
            if l == top_5_higher_entropies else \
            'top_5_lower_entropies_cluster_{}'.format(cluster)

        fig.savefig('./images/' + save_title + '.pdf', format='pdf',
                    bbox_inches='tight')

        plt.close(fig)
