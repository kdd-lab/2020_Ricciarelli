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

        entropies_dict.update(creator)

valid_creators = dict()

for mag_id in valid_mag_ids:
    creator = entropies_dict[mag_id]

    for m_id in creator:
        if len(creator[m_id]) >= 20:
            valid_creators.update(creator)

print(len(valid_creators))
