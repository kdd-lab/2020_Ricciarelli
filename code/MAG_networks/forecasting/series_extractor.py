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

for mag_id in tqdm(valid_mag_ids, desc='SELECTING VALID CREATORS'):
    creator = entropies_dict[mag_id]

    if len(creator) >= 20:
        valid_creators.update({mag_id: creator})

saving_to = sys.argv[1].replace('global_entropies.jsonl',
                                'valid_creators_time_series.jsonl')

with open(saving_to, 'w') as valid_creators_file:
    for mag_id in tqdm(valid_creators, desc='SAVING VALID CREATORS'):
        json.dump({mag_id: valid_creators[mag_id]}, valid_creators_file)
        valid_creators_file.write('\n')
