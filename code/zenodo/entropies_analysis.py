import json
import logging
import numpy as np
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

logging.basicConfig(filename='./logs/entropies_analysis.log',
                    level=logging.INFO, format='%(asctime)s -- %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S')

entropies_dict, years = dict(), list()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

        for MAG_id in creator:
            years += list(creator[MAG_id].keys())

entropies_matrix, years = list(), sorted(list(set(years)))

for MAG_id in tqdm(entropies_dict, desc='BUILDING ENTROPIES MATRIX'):
    row = list()

    for year in years:
        if year in entropies_dict[MAG_id]:
            row.append(entropies_dict[MAG_id][year]['entropy'])
        else:
            row.append(np.nan)

    entropies_matrix.append(row)

entropies_matrix = np.array(entropies_matrix)

means = [np.nanmean(entropies_matrix[:, idx])
         for idx in np.arange(0, entropies_matrix.shape[1])]

for i, row in tqdm(enumerate(entropies_matrix), desc='PREPROCESSING'):
    for j, val in enumerate(np.isnan(row)):
        if val == 1:
            entropies_matrix[i, j] = means[j]

for clusters_number in tqdm(np.arange(2, 11), desc='GRID SEARCH'):
    classifier = KMeans(n_clusters=clusters_number, max_iter=100)
    labels = classifier.fit_predict(entropies_matrix)

    silhouette_avg = silhouette_score(entropies_matrix, labels,
                                      sample_size=100000)

    logging.info("KMEANS' N_CLUSTERS GRID SEARCH -- N_CLUSTERS: {}, "
                 "AVERAGE SILHOUETTE SCORE: {}".format(clusters_number,
                                                       silhouette_avg))
