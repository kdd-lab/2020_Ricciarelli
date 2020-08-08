import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# log_name = '../logs/entropies_clustering_gs_decade_{}.log'.format(sys.argv[2])

# logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO,
#                     format='%(message)s')

entropies_dict = dict()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

entropies_matrix = list()

for mag_id in tqdm(sorted(entropies_dict), desc='BUILDING ENTROPIES MATRIX'):
    row = list()

    for year in np.arange(1980, 2020):
        if str(year) in entropies_dict[mag_id]:
            row.append(entropies_dict[mag_id][str(year)]['entropy'])
        else:
            row.append(np.nan)

    entropies_matrix.append(row)

entropies_matrix = np.array(entropies_matrix)

means = [np.nanmean(entropies_matrix[:, idx])
         for idx in np.arange(0, entropies_matrix.shape[1])]

for i, row in tqdm(enumerate(entropies_matrix), desc='PREPROCESSING',
                   total=entropies_matrix.shape[0]):
    for j, val in enumerate(np.isnan(row)):
        if val == 1:
            entropies_matrix[i, j] = means[j]

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs = axs.reshape((1, -1))[0]

fig.suptitle("Xenofilia/Xenophobia's Distribution by Decade", fontsize=10)

for idx, decade in enumerate([1980, 1990, 2000, 2010]):
    dataset = entropies_matrix[:, (idx * 10):((idx + 1) * 10)]

    classifier = KMeans(n_clusters=3, random_state=42)
    labels = classifier.fit_predict(dataset).tolist()

    entropies_per_cluster = {0: list(), 1: list(), 2: list()}

    for label_idx, mag_id in tqdm(enumerate(sorted(entropies_dict)),
                                  desc='DECADE {} -- ASSIGNING CLUSTER'
                                  .format(decade)):
        entropies = list()

        for year in np.arange(decade, decade + 10):
            entropies.append(entropies_dict[mag_id][str(year)]['entropy'])

        entropies_per_cluster[labels[label_idx]].append(np.mean(entropies))

    axs[idx].boxplot(entropies_per_cluster, labels=['0', '1', '2'],
                     showfliers=False, showmeans=True)
    axs[idx].grid(linestyle='--', color='black', alpha=0.4)
    axs[idx].set_title('From {} to {}'.format(decade, decade + 9), fontsize=8)
    axs[idx].set_xlabel('Cluster', fontsize=6)
    axs[idx].set_ylabel('Xenofilia/Xenophobia', fontsize=6)

fig.savefig('../images/clustering/entropies_distribution_by_decade.pdf',
            format='pdf')
