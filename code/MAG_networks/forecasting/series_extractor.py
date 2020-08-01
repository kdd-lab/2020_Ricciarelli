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

more_than_20_ents = dict()

for mag_id in tqdm(valid_mag_ids,
                   desc='SELECTING SERIES WITH MORE THAN 20 ENTRIES'):
    creator = entropies_dict[mag_id]

    if len(creator) >= 20:
        more_than_20_ents.update({mag_id: creator})

saving_to = sys.argv[1]\
    .replace('/entropies/global_entropies.jsonl',
             '/forecasting/series_with_more_than_20_entries.jsonl')

with open(saving_to, 'w') as more_than_20_ents_file:
    for mag_id in tqdm(more_than_20_ents,
                       desc='SAVING SERIES WITH MORE THAN 20 ENTRIES'):
        json.dump({mag_id: more_than_20_ents[mag_id]}, more_than_20_ents_file)
        more_than_20_ents_file.write('\n')

more_than_1_affiliation = dict()

for mag_id in tqdm(more_than_20_ents,
                   desc='SELECTING SERIES WITH MORE THAN 1 AFFILIATION'):
    creator, countries = entropies_dict[mag_id], set()

    for year in creator:
        countries.add(creator[year]['affiliation'])

    if len(countries) != 1:
        more_than_1_affiliation.update({mag_id: creator})

saving_to = saving_to.replace('series_with_more_than_20_entries.jsonl',
                              'series_with_more_than_1_affiliation.jsonl')

with open(saving_to, 'w') as more_than_1_affiliation_file:
    for mag_id in tqdm(more_than_1_affiliation,
                       desc='SAVING SERIES WITH MORE THAN 1 AFFILIATION'):
        json.dump({mag_id: more_than_1_affiliation[mag_id]},
                  more_than_1_affiliation_file)
        more_than_1_affiliation_file.write('\n')
