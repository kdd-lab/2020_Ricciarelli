import json
import pandas as pd
import sys

from tqdm import tqdm

cl_df = pd.read_csv(sys.argv[2], dtype={'MAG_id': str, 'cluster': int})
valid_mag_ids = cl_df[cl_df.cluster.isin([1, 2])]['MAG_id'].values

entropies_dict = dict()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        for mag_id in creator:
            if mag_id in valid_mag_ids and len(creator[mag_id]) != 1:
                entropies_dict.update(creator)

print(len(entropies_dict))
