import json
import logging
import numpy as np
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

logging.basicConfig(filename='kmeans_log.log', level=logging.DEBUG)

entropies_matrix = list()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = list(json.loads(row.strip()).items())[0]

        entropies_row = list()

        for year in np.arange(1980, 2010):
            if str(year) in creator[1]:
                entropies_row.append(creator[1][str(year)]['entropy'])
            else:
                entropies_row.append(np.nan)

        entropies_matrix.append(entropies_row)

entropies_matrix = np.array(entropies_matrix)

means = [np.nanmean(entropies_matrix[:, idx]) for idx in np.arange(0, 30)]

for i, row in tqdm(enumerate(entropies_matrix), desc='PREPROCESSING'):
    for j, val in enumerate(np.isnan(row)):
        if val == 1:
            entropies_matrix[i, j] = means[j]

for clusters_number in np.arange(2, 11):
    classifier = KMeans(n_clusters=clusters_number, max_iter=100)
    labels = classifier.fit_predict(entropies_matrix)

    silhouette_avg = silhouette_score(entropies_matrix, labels)

    logging.info("n_clusters: {}, average silhouette_score: {}"
                 .format(clusters_number, silhouette_avg))
