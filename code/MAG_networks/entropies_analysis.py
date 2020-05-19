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
                entropies_row.append(creator[1][str(year)])
            else:
                entropies_row.append(np.nan)

        entropies_matrix.append(entropies_row)

entropies_matrix = np.array(entropies_matrix)

for i, row in enumerate(entropies_matrix):
    for j, val in enumerate(np.isnan(row)):
        if val is True:
            entropies_matrix[i, j] = np.nanmean(entropies_matrix[:, j])
