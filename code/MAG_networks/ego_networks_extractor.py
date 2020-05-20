import gzip
import json
import numpy as np
import os
import sys

from collections import Counter
from tqdm import tqdm

affiliations_from_file = dict()

file_name = sys.argv[1] + 'authors_affiliation.json'

with open(file_name, 'r') as authors_affiliations_file:
    for affiliation in tqdm(authors_affiliations_file,
                            desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a = list(json.loads(affiliation.strip()).items())[0]

            to_add = dict()
            to_add['affiliations'] = a[1]

            affiliations_from_file[a[0]] = to_add

affiliations_countries = dict()

file_name = sys.argv[1] + 'affiliations_geo.txt'

with open(file_name, 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries[a[0]] = a[1]

affiliations = dict()

for year in np.arange(int(sys.argv[2]), int(sys.argv[2]) + 10):
    file_name = '{}/{}/{}.gz'.format(sys.argv[1], year, year)
    ego_nets = sys.argv[1] + str(year) + '/ego_networks'

    if not os.path.isdir(ego_nets):
        os.mkdir(ego_nets)

    with gzip.open(file_name, 'r') as es_list:
        for edge in tqdm(es_list,
                         desc='YEAR {}: READING EDGES LIST'.format(year)):
            e = edge.decode().strip().split(',')

            for node in [e[0], e[1]]:
                if node not in affiliations:
                    affiliation = list()

                    if node in affiliations_from_file:
                        for a_id in \
                         affiliations_from_file[node]['affiliations']:
                            _from = affiliations_from_file
                            [node]['affiliations'][a_id]['from']
                            _to = affiliations_from_file
                            [node]['affiliations'][a_id]['to'] + 1

                            years_range = np.arange(_from, _to)

                            if year in years_range and \
                               a_id in affiliations_countries:
                                affiliation.append(
                                    affiliations_countries[a_id])

                    if len(affiliation) != 0:
                        affiliation = Counter(affiliation)

                        affiliations[node] = affiliation.most_common(1)[0][0]

            if e[0] in affiliations and e[1] in affiliations:
                with open(ego_nets + '/' + e[0], 'a+') as ego_net_file:
                    to_write = '{},{},{},{}'.format(e[0],
                                                    affiliations[e[0]], e[1],
                                                    affiliations[e[1]])

                    ego_net_file.write(to_write + '\n')

                with open(ego_nets + '/' + e[1], 'a+') as ego_net_file:
                    to_write = '{},{},{},{}'.format(e[1],
                                                    affiliations[e[1]], e[0],
                                                    affiliations[e[0]])

                    ego_net_file.write(to_write + '\n')
