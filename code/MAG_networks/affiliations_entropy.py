import gzip
import igraph as ig
import json
import numpy as np
import sys

from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm

authors_affiliations = dict()

file_name = sys.argv[1] + '/' + 'authors_affiliation.json'

with open(file_name, 'r') as authors_affiliations_file:
    for affiliation in tqdm(authors_affiliations_file,
                            desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a = list(json.loads(affiliation.strip()).items())[0]

            to_add = dict()
            to_add['affiliations'] = a[1]
            to_add['valid'] = False

            authors_affiliations[a[0]] = to_add

affiliations_countries = dict()

file_name = sys.argv[1] + '/' + 'affiliations_geo.txt'

with open(file_name, 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries[a[0]] = a[1]

entropies = dict()

for year in np.arange(int(sys.argv[2]), int(sys.argv[2]) + 10):
    g = ig.Graph()

    file_name = '{}/{}/{}.gz'.format(sys.argv[1], year, year)

    nodes_list, edges_list, weights_list = list(), list(), list()

    with gzip.open(file_name, 'r') as es_list:
        for edge in tqdm(es_list,
                         desc='YEAR {}: READING EDGES LIST'.format(year)):
            e = edge.decode().strip().split(',')

            nodes_list.append(e[0])
            nodes_list.append(e[1])
            edges_list.append((e[0], e[1]))
            weights_list.append(int(e[2]))

    nodes_list = list(set(nodes_list))

    for node in tqdm(nodes_list, desc='YEAR {}: ADDING NODES'.format(year)):
        affiliation = list()

        if node in authors_affiliations:
            for a_id in authors_affiliations[node]['affiliations']:
                _from = \
                    authors_affiliations[node]['affiliations'][a_id]['from']
                _to = \
                    authors_affiliations[node]['affiliations'][a_id]['to'] + 1

                years_range = np.arange(_from, _to)

                if year in years_range and a_id in affiliations_countries:
                    affiliation.append(affiliations_countries[a_id])

            if len(affiliation) != 0:
                affiliation = Counter(affiliation)

                g.add_vertex(node,
                             affiliation=affiliation.most_common(1)[0][0])

                authors_affiliations[node]['valid'] = True

    valid_edges, valid_weights = list(), list()

    for idx, val in tqdm(enumerate(edges_list),
                         desc='YEAR {}: VALIDATING EDGES'.format(year)):
        if val[0] in authors_affiliations and val[1] in authors_affiliations:
            if authors_affiliations[val[0]]['valid'] and \
               authors_affiliations[val[1]]['valid']:
                valid_edges.append(edges_list[idx])
                valid_weights.append(weights_list[idx])

    g.add_edges(valid_edges)
    g.es['weight'] = valid_weights

    for node in tqdm(g.vs, desc='YEAR {}: COMPUTING ENTROPIES'.format(year)):
        if len(g.neighborhood(node['name'], mindist=1)) != 0:
            ego_net = g.induced_subgraph(g.neighborhood(node['name'],
                                                        mindist=1))

            ego_country = node['affiliation']

            affiliations_types = list()
            monst_common_class = None

            for neighbor in ego_net.vs:
                neighbor_country = neighbor['affiliation']

                if neighbor_country == ego_country:
                    affiliations_types.append('same')
                else:
                    affiliations_types.append('different')

            affiliations_types = Counter(affiliations_types)
            monst_common_class = affiliations_types.most_common(1)[0][0]
            affiliations_types = \
                [affiliations_types[k] / sum(affiliations_types.values())
                 for k in affiliations_types]

            to_add = dict()

            if monst_common_class == 'same':
                to_add['entropy'] = \
                    np.round(entropy(affiliations_types, base=2), 2)
            else:
                to_add['entropy'] = \
                    np.round(entropy(affiliations_types, base=2), 2) * -1

            to_add['class'] = monst_common_class
            to_add['affiliation'] = ego_country

            if node['name'] not in entropies:
                entropies[node['name']] = {int(year): to_add}
            else:
                entropies[node['name']][int(year)] = to_add

    for node in nodes_list:
        if node in authors_affiliations:
            authors_affiliations[node]['valid'] = False

to_save = '{}entropies_{}_{}.jsonl'.format(sys.argv[1], sys.argv[2],
                                           int(sys.argv[2]) + 9)

with open(to_save, 'w') as entropies_file:
    for creator in tqdm(entropies.items(), desc='WRITING ENTROPIES JSONL'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, entropies_file)
        entropies_file.write('\n')
