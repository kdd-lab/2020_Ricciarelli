import json
import logging
import numpy as np
import pandas as pd
import sys

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# UNCOMMENT FOR GRID SEARCHING THE OPTIMAL NUMBER OF CLUSTERS

# logging.basicConfig(filename='./logs/entropies_analysis.log',
#                     level=logging.INFO, format='%(asctime)s -- %(message)s',
#                     datefmt='%d-%m-%y %H:%M:%S')

logging.basicConfig(filename='./logs/entropies_analysis_results.log',
                    filemode='w', level=logging.INFO, format='%(message)s')

entropies_dict, years = dict(), set()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

        for MAG_id in creator:
            for year in creator[MAG_id]:
                years.add(year)

entropies_matrix, years = list(), sorted(years)

for MAG_id in tqdm(sorted(entropies_dict), desc='BUILDING ENTROPIES MATRIX'):
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

for i, row in tqdm(enumerate(entropies_matrix), desc='PREPROCESSING',
                   total=entropies_matrix.shape[0]):
    for j, val in enumerate(np.isnan(row)):
        if val == 1:
            entropies_matrix[i, j] = means[j]

# UNCOMMENT FOR GRID SEARCHING THE OPTIMAL NUMBER OF CLUSTERS

# for clusters_number in tqdm(np.arange(2, 11), desc='GRID SEARCH'):
#     classifier = KMeans(n_clusters=clusters_number, max_iter=100)
#     labels = classifier.fit_predict(entropies_matrix)

#     silhouette_avg = silhouette_score(entropies_matrix, labels,
#                                       sample_size=100000)

#     logging.info("KMEANS' N_CLUSTERS GRID SEARCH -- N_CLUSTERS: {}, "
#                  "AVERAGE SILHOUETTE SCORE: {}".format(clusters_number,
#                                                        silhouette_avg))

classifier = KMeans(n_clusters=5, random_state=42)
labels = classifier.fit_predict(entropies_matrix)
centroids = classifier.cluster_centers_

clusters_infos = dict()
dataframe_infos = [[], [], []]

for idx, MAG_id in tqdm(enumerate(sorted(entropies_dict)),
                        desc='ASSIGNING CLUSTERS', total=len(entropies_dict)):
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
        np.linalg.norm(entropies_matrix[labels[idx]] - centroids[labels[idx]]))

clustering_dataframe = pd.DataFrame({'MAG_id': dataframe_infos[0],
                                     'cluster': dataframe_infos[1],
                                     'DFC': dataframe_infos[2]})

representative_records = list()

for cluster in sorted(clustering_dataframe.cluster.unique()):
    DFCs = \
        clustering_dataframe[clustering_dataframe.cluster == cluster]['DFC']

    record = clustering_dataframe\
        .query('cluster == {} and DFC == {}'.format(cluster, min(DFCs)))\
        .iloc[0]

    representative_records.append(entropies_dict[record.values[0]])

creators_per_cluster = Counter(list(labels))

for cluster in sorted(creators_per_cluster):
    percentage = \
        np.round((creators_per_cluster[cluster] / len(entropies_dict)) * 100,
                 2)

    logging.info('CLUSTER {}:\n\t{}% OF THE RESEARCHERS\n'
                 .format(cluster, percentage))
    logging.info('\tYEARS: {}\n'
                 .format(sorted(clusters_infos[cluster]['years'])))
    logging.info('\tCOUNTRIES: {}\n'
                 .format(sorted(clusters_infos[cluster]['countries'])))
    logging.info('\tREPRESENTATIVE RECORD: {}\n'
                 .format(representative_records[int(cluster)]))
