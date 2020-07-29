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

entropies_per_country = {1: dict(), 2: dict()}

for cluster in [1, 2]:
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

            if country not in entropies_per_country[cluster]:
                entropies_per_country[cluster][country] = defaultdict(list)

            entropies_per_country[cluster][country][year].append(entropy)

entropies = dict()

for country in ['Italy', 'Germany', 'France', 'Spain', 'United Kingdom',
                'United States of America', 'Russia', 'China']:
    entropies[country] = defaultdict(list)

    for c in entropies_per_country:
        for y in entropies_per_country[c][country]:
            for value in entropies_per_country[c][country][y]:
                if value not in [0.0, -0.0]:
                    entropies[country][y].append(value)

    entropies[country] = [np.mean(entropies_per_country[c][country][year])
                          for year in
                          sorted(entropies_per_country[c][country])]

fig, ax = plt.subplots(4, 2, constrained_layout=True)

fig.suptitle("Xenofilia/Xenophobia's trend for various Countries",
             fontsize=10)

for country, coordinates in zip(entropies, [[0, 0], [0, 1], [1, 0], [1, 1],
                                [2, 0], [2, 1], [3, 0], [3, 1]]):
    ax[coordinates[0], coordinates[1]].plot(np.arange(1980, 2020),
                                            entropies[country], linewidth=2,
                                            label=country, color='steelblue')
    ax[coordinates[0], coordinates[1]].set_xlim(1979, 2020)
    ax[coordinates[0], coordinates[1]].set_ylim(-1.0, 1.0)
    ax[coordinates[0], coordinates[1]].set_xticks(np.arange(1980, 2020, 10))
    ax[coordinates[0], coordinates[1]].set_xticks(np.arange(1980, 2020),
                                                  minor=True)
    ax[coordinates[0], coordinates[1]].tick_params(axis='both', which='major',
                                                   labelsize=6)
    ax[coordinates[0], coordinates[1]].set_title(country, fontsize=8)

fig.savefig('../images/clustering/entropies_trends_of_various_countries.pdf',
            format='pdf')

plt.close(fig)

for c in entropies_per_country:
    mean_entropies_per_country = dict()

    for country in entropies_per_country[c]:
        entropies = [entropies_per_country[c][country][year] for year
                     in sorted(entropies_per_country[c][country])]

        if len(entropies) == 40:
            mean_entropies_per_country[np.mean(np.concatenate(entropies))] = \
                country

    top_5_higher_entropies = sorted(mean_entropies_per_country,
                                    reverse=True)[:5]
    top_5_lower_entropies = sorted(mean_entropies_per_country)[:5]

    top_5_higher_entropies = [mean_entropies_per_country[mean] for mean in
                              top_5_higher_entropies]
    top_5_lower_entropies = [mean_entropies_per_country[mean] for mean in
                             top_5_lower_entropies]

    fig, axes = plt.subplots(nrows=5, ncols=1, constrained_layout=True)

    fig.suptitle('Top 5 Countries with higher/lower Xenofilia/Xenophobia - '
                 'Cluster {}'.format(c), fontsize=10)

    for x in np.arange(0, 5):
        for l in [top_5_higher_entropies, top_5_lower_entropies]:
            stats = dict()

            for year in entropies_per_country[c][l[x]]:
                stats[year] = {
                    'mean': np.mean(entropies_per_country[c][l[x]][year]),
                    'std': np.std(entropies_per_country[c][l[x]][year]),
                    'samples': len(entropies_per_country[c][l[x]][year])}

            ys, ys_fill_up, ys_fill_down = list(), list(), list()

            for year in np.arange(1980, 2020):
                if str(year) in stats:
                    ys.append(stats[str(year)]['mean'])
                    ys_fill_up.append(stats[str(year)]['mean'] +
                                      stats[str(year)]['std'])
                    ys_fill_down.append(stats[str(year)]['mean'] -
                                        stats[str(year)]['std'])
                else:
                    ys.append(np.nan)
                    ys_fill_up.append(np.nan)
                    ys_fill_down.append(np.nan)

            axes[x].plot(np.arange(1980, 2020), ys, linewidth=2,
                         color='steelblue' if l == top_5_higher_entropies
                         else 'tomato', label=l[x], alpha=0.5)
            axes[x].fill_between(np.arange(1980, 2020), ys_fill_down,
                                 ys_fill_up, color='steelblue' if
                                 l == top_5_higher_entropies else 'tomato',
                                 alpha=0.3)

        axes[x].set_xlim(1979, 2020)
        axes[x].set_ylim(-1.0, 1.0)
        axes[x].set_xticks(np.arange(1980, 2020, 10))
        axes[x].set_xticks(np.arange(1980, 2020), minor=True)
        axes[x].tick_params(axis='both', which='major', labelsize=6)
        axes[x].legend(loc='center left', fontsize=6,
                       bbox_to_anchor=(1, 0.5))

    save_title = 'top_5_higher_lower_entropies_cluster_{}'.format(c)

    fig.savefig('../images/clustering/' + save_title + '.pdf', format='pdf')

    plt.close(fig)
