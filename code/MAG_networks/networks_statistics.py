import json
import numpy as np
import sys

from tqdm import tqdm

decade = np.arange(int(sys.argv[1]), int(sys.argv[1]) + 10)

authors_affiliations = dict()

import ipdb
ipdb.set_trace()

with open(sys.argv[2], 'r') as affiliations:
    for affiliation in tqdm(affiliations, desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a = json.loads(affiliation.strip())

            for mag_id in a:
                for aff_id in a[mag_id]:
                    if a[mag_id][aff_id]['from'] not in decade:
                        del a[mag_id][aff_id]

            if len(a[mag_id]) != 0:
                authors_affiliations.update(a)

affiliations_countries = dict()

with open(sys.argv[3], 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries.update(a)
