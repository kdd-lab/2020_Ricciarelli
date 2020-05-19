import json
import numpy as np
import sys

from tqdm import tqdm

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

import ipdb
ipdb.set_trace()

for i, row in tqdm(enumerate(entropies_matrix), desc='PREPROCESSING'):
    for j, val in enumerate(np.isnan(row)):
        if val == 1:
            entropies_matrix[i, j] = means[j]
