import json
import logging
import numpy as np
import pandas as pd
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

log_filename = './logs/entropies_deeper_clustering_grid_search.log' \
    if len(sys.argv == 5) else './logs/entropies_deeper_clustering_results.log'

logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO,
                    format='%(message)s')

clustering_dataframe = pd.read_csv(sys.argv[2], dtype={'MAG_id': str,
                                   'cluster': int})
clustering_dataframe = \
    clustering_dataframe[clustering_dataframe.cluster != int(sys.argv[3])]

entropies_dict, total_years = dict(), set()
valid_MAG_ids = sorted(clustering_dataframe['MAG_id'].values.tolist())

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

        for MAG_id in creator:
            for year in creator[MAG_id]:
                total_years.add(year)

entropies_matrix, total_years, local_years = list(), sorted(total_years), set()

for MAG_id in valid_MAG_ids:
    for year in entropies_dict[MAG_id]:
        local_years.add(year)

local_years = sorted(local_years)

for MAG_id in tqdm(valid_MAG_ids, desc='BUILDING ENTROPIES MATRIX'):
    row = list()

    for year in local_years:
        if year in entropies_dict[MAG_id]:
            row.append(entropies_dict[MAG_id][year]['entropy'])
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

if len(sys.argv == 5):
    for iteration in tqdm(np.arange(0, int(sys.argv[4])),
                          desc='GRID SEARCH ITERATION'):
        for clusters_number in tqdm(np.arange(2, 11), desc='GRID SEARCH'):
            classifier = KMeans(n_clusters=clusters_number, max_iter=100)
            labels = classifier.fit_predict(entropies_matrix)

            silhouette_avg = silhouette_score(entropies_matrix, labels)

            logging.info("KMEANS' N_CLUSTERS GRID SEARCH -- N_CLUSTERS: {}, "
                         "AVERAGE SILHOUETTE SCORE: {}".format(clusters_number,
                                                               silhouette_avg))
else:
    classifier = KMeans(n_clusters=2, random_state=42)
    labels = classifier.fit_predict(entropies_matrix)
    centroids = classifier.cluster_centers_

    clusters_infos, dataframe_infos = dict(), [[], [], []]
    clusters_records = [[], [], [], [], []]
    clusters_records_without_mean = [[], [], [], [], []]

    for idx, MAG_id in tqdm(enumerate(sorted(entropies_dict)),
                            desc='ASSIGNING CLUSTERS',
                            total=len(entropies_dict)):
        if labels[idx] not in clusters_infos:
            clusters_infos[labels[idx]] = \
                {'years': set(list(entropies_dict[MAG_id])),
                 'countries': set([entropies_dict[MAG_id][y]['affiliation']
                                  for y in entropies_dict[MAG_id]])}
        else:
            for year in entropies_dict[MAG_id]:
                clusters_infos[labels[idx]]['years'].add(year)
                clusters_infos[labels[idx]]['countries']\
                    .add(entropies_dict[MAG_id][year]['affiliation'])

        dataframe_infos[0].append(MAG_id)
        dataframe_infos[1].append(labels[idx])
        dataframe_infos[2].append(
            np.linalg.norm(entropies_matrix[labels[idx]] -
                           centroids[labels[idx]]))

        clusters_records[labels[idx]].append(np.mean(entropies_matrix[idx]))

    clustering_dataframe = pd.DataFrame({'MAG_id': dataframe_infos[0],
                                         'cluster': dataframe_infos[1],
                                         'DFC': dataframe_infos[2]})
