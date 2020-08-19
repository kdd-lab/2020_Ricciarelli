import json
import matplotlib.pyplot as plt
import numpy as np
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

entropy_per_fos_per_year = dict()

for mag_id in tqdm(fos_dict, desc='ASSIGNING ENTROPY TO FOS'):
    fos_years = set([year for year in fos_dict[mag_id]])
    entropies_years = set([year for year in entropies_dict[mag_id]])

    for year in fos_years.intersection(entropies_years):
        score = entropies_dict[mag_id][year]['entropy']

        for fos in fos_dict[mag_id][year]:
            if fos not in entropy_per_fos_per_year:
                entropy_per_fos_per_year[fos] = defaultdict(list)

            entropy_per_fos_per_year[fos][year].append(score)

for fos in entropy_per_fos_per_year:
    for year in entropy_per_fos_per_year[fos]:
        mean = np.mean(entropy_per_fos_per_year[fos][year])
        std = np.std(entropy_per_fos_per_year[fos][year])

        entropy_per_fos_per_year[fos][year] = {'mean': mean, 'std': std}

fig, ax = plt.subplots(1, 1, constrained_layout=True)

for fos in sorted(entropy_per_fos_per_year):
    ax.plot(np.arange(1980, 2020),
            [entropy_per_fos_per_year[fos][y]['mean'] for y in
            sorted(entropy_per_fos_per_year[fos])], lw=2, color='steelblue')
    ax.set_xlim(1979, 2018)
    ax.set_xticks(np.arange(1980, 2018, 10))
    ax.set_xticks(np.arange(1980, 2018), minor=True)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_ylabel('Xenofilia/Xenofobia', fontsize=8)

    break

fig.savefig('../images/fos/xenofilia_xenofobia_per_fos_per_tear.pdf',
            format='pdf')
plt.close(fig)
