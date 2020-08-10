import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import Counter, defaultdict
from tqdm import tqdm

fos_dict = dict()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        fos_dict.update(creator)

cl_df = pd.read_csv(sys.argv[2], dtype={'MAG_id': str, 'cluster': int})

fos_counter_per_year = defaultdict(Counter)

for mag_id in tqdm(fos_dict, desc='COUNTING FOS PER YEAR'):
    for year in fos_dict[mag_id]:
        for field_of_study in fos_dict[mag_id][year]:
            fos_counter_per_year[year][field_of_study] += 1

markers = list(mpl.markers.MarkerStyle.markers.keys())

fields_of_study = set([fos_counter_per_year[str(x)].most_common()[0][0]
                      for x in np.arange(1980, 2020)])
field_of_study_markers = dict()

for idx, fos in enumerate(fields_of_study):
    field_of_study_markers[fos] = markers[idx]

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.set_title('Most represented Field of Study per Year', fontsize=10)
ax.plot(np.arange(1980, 2020),
        [fos_counter_per_year[year].most_common()[0][1]
        for year in fos_counter_per_year], lw=2, color='steelblue', alpha=0.7)

for idx, x in enumerate(np.arange(1980, 2020)):
    y = ax.get_children()[idx].properties()['data'][1][idx]
    field_of_study = fos_counter_per_year[str(x)].most_common()[0][0]

    ax.scatter(x, y, c='steelblue',
               marker=field_of_study_markers[field_of_study])

ax.set_xlim(1979, 2020)
ax.set_xticks(np.arange(1980, 2020, 10))
ax.set_xticks(np.arange(1980, 2020), minor=True)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Year', fontsize=8)
ax.set_ylabel('Registered Entries', fontsize=8)
fig.savefig('../images/fos/most_represented_fos_per_year.pdf',
            format='pdf')
plt.close(fig)

fig, ax = plt.subplots(1, 2, constrained_layout=True)
ax = ax.reshape((1, -1))[0]
fig.suptitle('Most represented Fields of Study per Cluster', fontsize=10)

for idx, cluster in enumerate([1, 2]):
    fos_counter_per_cluster = defaultdict(Counter)

    for mag_id in cl_df[cl_df.cluster == cluster]['MAG_id']:
        if mag_id in fos_dict:
            for year in fos_dict[mag_id]:
                for field_of_study in fos_dict[mag_id][year]:
                    fos_counter_per_cluster[year][field_of_study] += 1

    fields_of_study = set([fos_counter_per_cluster[str(x)].most_common()[0][0]
                          for x in np.arange(1980, 2020)])
    field_of_study_markers = dict()

    for i, fos in enumerate(fields_of_study):
        field_of_study_markers[fos] = markers[i]

    ax[idx].plot(np.arange(1980, 2020),
                 [fos_counter_per_cluster[year].most_common()[0][1]
                 for year in fos_counter_per_year], lw=2, color='steelblue',
                 alpha=0.7)

    for i, x in enumerate(np.arange(1980, 2020)):
        y = ax[idx].get_children()[i].properties()['data'][1][i]
        field_of_study = fos_counter_per_cluster[str(x)].most_common()[0][0]

        ax[idx].scatter(x, y, c='steelblue',
                        marker=field_of_study_markers[field_of_study])

    ax[idx].set_title('Cluster {}'.format(cluster), fontsize=8)
    ax[idx].set_xlim(1979, 2020)
    ax[idx].set_xticks(np.arange(1980, 2020, 10))
    ax[idx].set_xticks(np.arange(1980, 2020), minor=True)
    ax[idx].tick_params(axis='both', which='major', labelsize=6)
    ax[idx].set_xlabel('Year', fontsize=8)
    ax[idx].set_ylabel('Registered Entries', fontsize=8)

fig.savefig('../images/fos/most_represented_fos_per_cluster.pdf',
            format='pdf')
plt.close(fig)
