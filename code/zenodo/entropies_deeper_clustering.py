import json
import logging
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

log_filename = None

if len(sys.argv) == 2:
    log_filename = './logs/entropies_clustering_results.log'
elif len(sys.argv) == 3:
    log_filename = './logs/entropies_clustering_grid_search.log'
elif len(sys.argv) == 4:
    log_filename = './logs/entropies_deeper_clustering_results.log'
else:
    log_filename = './logs/entropies_deeper_clustering_grid_search.log'

logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO,
                    format='%(message)s')

clustering_df, valid_clustering_df, valid_MAG_ids = None, None, None

if len(sys.argv) > 3:
    clustering_df = pd.read_csv(sys.argv[2],
                                dtype={'MAG_id': str, 'cluster': int})
    valid_clustering_df = \
        clustering_df[clustering_df.cluster != int(sys.argv[3])]

    valid_MAG_ids = sorted(valid_clustering_df['MAG_id'].values.tolist())

entropies_dict, total_years = dict(), set()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

        for MAG_id in creator:
            for year in creator[MAG_id]:
                total_years.add(year)

entropies_matrix, total_years, local_years = list(), sorted(total_years), set()

if sys.argv > 3:
    for MAG_id in valid_MAG_ids:
        for year in entropies_dict[MAG_id]:
            local_years.add(year)

    local_years = sorted(local_years)

for MAG_id in tqdm(valid_MAG_ids if sys.argv > 3 else sorted(entropies_dict),
                   desc='BUILDING ENTROPIES MATRIX'):
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

if len(sys.argv) == 3 or len(sys.argv) == 5:
    iterations = int(sys.argv[2]) if len(sys.argv) == 2 else int(sys.argv[4])

    for iteration in tqdm(np.arange(0, iterations),
                          desc='GRID SEARCH ITERATION'):
        for clusters_number in tqdm(np.arange(2, 11), desc='GRID SEARCH'):
            classifier = KMeans(n_clusters=clusters_number, max_iter=100)
            labels = classifier.fit_predict(entropies_matrix)

            silhouette_avg = \
                silhouette_score(entropies_matrix, labels,
                                 sample_size=100000 if len(sys.argv) == 3
                                 else None)

            logging.info("KMEANS' N_CLUSTERS GRID SEARCH -- N_CLUSTERS: {}, "
                         "AVERAGE SILHOUETTE SCORE: {}".format(clusters_number,
                                                               silhouette_avg))
else:
    classifier = KMeans(n_clusters=2, random_state=42)
    labels = classifier.fit_predict(entropies_matrix).tolist()

    if len(sys.argv) > 3:
        for idx, label in enumerate(labels):
            if label == 0:
                labels[idx] = 1
            else:
                labels[idx] = 2

    centroids = classifier.cluster_centers_

    clusters_infos, dataframe_infos = dict(), [[], [], []]
    clusters_records, clusters_records_without_mean = None, None

    if len(sys.argv) == 2:
        clusters_records, clusters_records_without_mean = \
            [[], [], [], [], []], [[], [], [], [], []]
    else:
        clusters_records, clusters_records_without_mean = [[], []], [[], []]

    total = len(entropies_dict) if len(sys.argv) == 2 else len(valid_MAG_ids)

    for idx, MAG_id in tqdm(enumerate(sorted(entropies_dict))
                            if len(sys.argv) == 2 else
                            enumerate(valid_MAG_ids),
                            desc='ASSIGNING CLUSTERS', total=total):
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

        c_idx = labels[idx] if len(sys.argv) == 2 else labels[idx] - 1

        dataframe_infos[0].append(MAG_id)
        dataframe_infos[1].append(labels[idx])
        dataframe_infos[2].append(
            np.linalg.norm(entropies_matrix[labels[idx]] - centroids[c_idx]))

        clusters_records[c_idx].append(np.mean(entropies_matrix[idx]))

    new_clustering_df = pd.DataFrame({'MAG_id': dataframe_infos[0],
                                      'cluster': dataframe_infos[1],
                                      'DFC': dataframe_infos[2]})

    representative_records = list()

    for cluster in sorted(new_clustering_df.cluster.unique()):
        cluster_dataframe = \
            new_clustering_df[new_clustering_df.cluster == cluster]

        records = new_clustering_df\
            .query('cluster == {} and DFC == {}'
                   .format(cluster, min(cluster_dataframe['DFC']))).iloc[0:5]

        for record in records.values:
            representative_records\
                .append({record[0]: entropies_dict[record[0]]})

        for MAG_id in cluster_dataframe['MAG_id']:
            entropies = list()

            for year in entropies_dict[MAG_id]:
                entropies.append(entropies_dict[MAG_id][year]['entropy'])

            c_idx = cluster if len(sys.argv) == 2 else cluster - 1

            clusters_records_without_mean[c_idx].append(np.mean(entropies))

    creators_per_cluster = Counter(labels)

    for cluster in sorted(creators_per_cluster):
        denominator = len(entropies_dict) if len(sys.argv) == 2 else \
            len(valid_MAG_ids)
        percentage = \
            np.round((creators_per_cluster[cluster] / denominator) * 100, 2)

        logging.info('CLUSTER {}:\n\t{} RESEARCHERS, {}% OF THE TOTAL\n'
                     .format(cluster, creators_per_cluster[cluster],
                             percentage))
        logging.info('\tYEARS: {}\n'
                     .format(sorted(clusters_infos[cluster]['years'])))
        logging.info('\tCOUNTRIES: {}\n'
                     .format(sorted(clusters_infos[cluster]['countries'])))
        logging.info('\tREPRESENTATIVE RECORDS:\n')

        step = int(cluster) if len(sys.argv) == 2 else int(cluster) - 1

        for record in representative_records[step * 5:(step * 5) + 5]:
            logging.info('\t\t{}'.format(record))

    # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # axs[0].set_title("Entropies' Distribution per Cluster with Mean Values",
    #                  fontsize=20)
    # axs[0].boxplot(clusters_records, labels=['1', '2'], zorder=2)
    # axs[0].set_xlabel('Cluster', fontsize=14)
    # axs[0].set_ylabel('Silhouette Score', fontsize=14)
    # axs[0].grid(axis='y', linestyle='--', color='black', zorder=1)

    # axs[1].set_title("Entropies' Distribution per Cluster without Mean Values",
    #                  fontsize=20)
    # axs[1].boxplot(clusters_records_without_mean, labels=['1', '2'], zorder=2)
    # axs[1].set_xlabel('Cluster', fontsize=14)
    # axs[1].set_ylabel('Silhouette Score', fontsize=14)
    # axs[1].grid(axis='y', linestyle='--', color='black', zorder=1)

    # fig.tight_layout()
    # fig.savefig('./images/entropies_distribution_per_deeper_cluster.pdf',
    #             format='pdf')
    # plt.close(fig=fig)

    if len(sys.argv) == 2:
        new_clustering_df[['MAG_id', 'cluster']].to_csv(
            '~/mydata/zenodo/clustering/clustering_data.csv', index=False)
    else:
        clustering_df.loc[
            clustering_df.query('MAG_id in {}'.format(valid_MAG_ids)).index,
            'cluster'] = labels
        clustering_df.to_csv(
            '~/mydata/zenodo/clustering/deeper_clustering.csv', index=False)
