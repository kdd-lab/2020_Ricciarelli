import json
import numpy as np
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

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

for idx, decade in enumerate([1980, 1990, 2000, 2010]):
    dataset = entropies_matrix[:, (idx * 10):((idx + 1) * 10)]

    classifier = KMeans(n_clusters=3, random_state=42)
    labels = classifier.fit_predict(dataset).tolist()

    silhouette_avg = silhouette_score(entropies_matrix, labels,
                                      sample_size=100000)

    print('Decade {}-{}: {}'.format(decade, decade + 9, silhouette_avg))
