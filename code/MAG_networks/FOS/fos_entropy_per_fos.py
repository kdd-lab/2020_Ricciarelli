import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from tqdm import tqdm


fos_dict = dict()

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        fos_dict.update(creator)

entropies_dict = dict()

with open(sys.argv[2], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

cl_df = pd.read_csv(sys.argv[3], dtype={'MAG_id': str, 'cluster': int})

entropy_per_fos_per_year = dict()

for mag_id in cl_df[cl_df.cluster.isin([1, 2])]['MAG_id']:
    if mag_id in fos_dict:
        fos_years = set([year for year in fos_dict[mag_id]])
        entropies_years = set([year for year in entropies_dict[mag_id]])

        for year in fos_years.intersection(entropies_years):
            score = entropies_dict[mag_id][year]['entropy']

            for fos in fos_dict[mag_id][year]:
                if fos not in entropy_per_fos_per_year:
                    entropy_per_fos_per_year[fos] = defaultdict(list)

                entropy_per_fos_per_year[fos][year].append(score)

for fos in entropy_per_fos_per_year:
    for year in np.arange(1980, 2020):
        mean, std = np.nan, np.nan

        if str(year) in entropy_per_fos_per_year[fos]:
            mean = np.mean(entropy_per_fos_per_year[fos][str(year)])
            std = np.std(entropy_per_fos_per_year[fos][str(year)])

        entropy_per_fos_per_year[fos][str(year)] = {'mean': mean, 'std': std}

fig, ax = plt.subplots(3, 2, constrained_layout=True)
ax = ax.reshape((1, -1))[0]

fig.suptitle('Xenofilia/Xenofobia per Field of Study over the Years',
             fontsize=10)

checked = list()

for idx, fos in enumerate(sorted(entropy_per_fos_per_year)):
    if idx > 5:
        break

    ax[idx].plot(np.arange(1980, 2020),
                 [entropy_per_fos_per_year[fos][y]['mean'] for y in
                 sorted(entropy_per_fos_per_year[fos])], lw=2,
                 color='steelblue')
    ax[idx].fill_between(np.arange(1980, 2020),
                         [entropy_per_fos_per_year[fos][y]['mean'] -
                         entropy_per_fos_per_year[fos][y]['std'] for y in
                         sorted(entropy_per_fos_per_year[fos])],
                         [entropy_per_fos_per_year[fos][y]['mean'] +
                         entropy_per_fos_per_year[fos][y]['std'] for y in
                         sorted(entropy_per_fos_per_year[fos])],
                         color='steelblue', alpha=0.3)
    ax[idx].set_xlim(1979, 2018)
    ax[idx].set_xticks(np.arange(1980, 2018, 10))
    ax[idx].set_xticks(np.arange(1980, 2018), minor=True)
    ax[idx].tick_params(axis='both', which='major', labelsize=6)
    ax[idx].set_xlabel('Year', fontsize=8)
    ax[idx].set_ylabel('Xenofilia/Xenofobia', fontsize=8)

    checked.append(fos)

fig.savefig('../images/fos/xenofilia_xenofobia_per_fos_per_year_1.pdf',
            format='pdf')
plt.close(fig)

fig, ax = plt.subplots(4, 2, constrained_layout=True)
ax = ax.reshape((1, -1))[0]

fig.suptitle('Xenofilia/Xenofobia per Field of Study over the Years',
             fontsize=10)

remaining_fos = [fos for fos in entropy_per_fos_per_year if fos not in checked]

for idx, fos in enumerate(sorted(remaining_fos)):
    ax[idx].plot(np.arange(1980, 2020),
                 [entropy_per_fos_per_year[fos][y]['mean'] for y in
                 sorted(entropy_per_fos_per_year[fos])], lw=2,
                 color='steelblue')
    ax[idx].fill_between(np.arange(1980, 2020),
                         [entropy_per_fos_per_year[fos][y]['mean'] -
                         entropy_per_fos_per_year[fos][y]['std'] for y in
                         sorted(entropy_per_fos_per_year[fos])],
                         [entropy_per_fos_per_year[fos][y]['mean'] +
                         entropy_per_fos_per_year[fos][y]['std'] for y in
                         sorted(entropy_per_fos_per_year[fos])],
                         color='steelblue', alpha=0.3)
    ax[idx].set_xlim(1979, 2018)
    ax[idx].set_xticks(np.arange(1980, 2018, 10))
    ax[idx].set_xticks(np.arange(1980, 2018), minor=True)
    ax[idx].tick_params(axis='both', which='major', labelsize=6)
    ax[idx].set_xlabel('Year', fontsize=8)
    ax[idx].set_ylabel('Xenofilia/Xenofobia', fontsize=8)

fig.savefig('../images/fos/xenofilia_xenofobia_per_fos_per_year_2.pdf',
            format='pdf')
plt.close(fig)
