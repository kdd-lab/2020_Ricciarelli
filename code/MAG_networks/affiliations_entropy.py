import gzip
import igraph as ig
import json
import numpy as np
import sys

from collections import Counter
from scipy.stats import entropy
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
            to_add['valid'] = False

            affiliations_from_file[a[0]] = to_add

affiliations_countries = dict()

file_name = sys.argv[1] + 'affiliations_geo.txt'

with open(file_name, 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries[a[0]] = a[1]

entropies, affiliations = dict(), dict()

for year in np.arange(int(sys.argv[2]), int(sys.argv[2]) + 10):
    file_name = '{}/{}/{}.gz'.format(sys.argv[1], year, year)

    nodes_list, edges_list = list(), list()

    with gzip.open(file_name, 'r') as es_list:
        for edge in tqdm(es_list,
                         desc='YEAR {}: READING EDGES LIST'.format(year)):
            e = edge.decode().strip().split(',')

            nodes_list.append(e[0])
            nodes_list.append(e[1])
            edges_list.append((e[0], e[1]))

    nodes_list = list(set(nodes_list))

    for node in tqdm(nodes_list, desc='YEAR {}: ADDING NODES'.format(year)):
        g = ig.Graph()

        founded_nodes, founded_edges = list(), list()
        founded_nodes.append(node)

        for edge in edges_list:
            if node in edge:
                founded_edges.append(edge)

                if edge[0] != node:
                    founded_nodes.append(edge[0])
                else:
                    founded_nodes.append(edge[1])

        for f_node in founded_nodes:
            if f_node in affiliations:
                g.add_vertex(node, affiliation=affiliations[f_node])
            elif f_node in affiliations_from_file:
                for a_id in affiliations_from_file[f_node]['affiliations']:
                    _from = \
                        affiliations_from_file[f_node]['affiliations'][a_id]['from']
                    _to = \
                        affiliations_from_file[f_node]['affiliations'][a_id]['to'] + 1

                    years_range = np.arange(_from, _to)

                    if year in years_range and a_id in affiliations_countries:
                        affiliation.append(affiliations_countries[a_id])

                if len(affiliation) != 0:
                    affiliation = Counter(affiliation)

                    affiliations[f_node] = affiliation.most_common(1)[0][0]

                    g.add_vertex(f_node, affiliation=affiliations[f_node])

                    affiliations_from_file[f_node]['valid'] = True

        valid_edges, valid_weights = list(), list()

        for idx, val in enumerate(founded_edges):
            if val[0] in affiliations and val[1] in affiliations:
                if affiliations_from_file[val[0]]['valid'] and \
                   affiliations_from_file[val[1]]['valid']:
                    valid_edges.append(edges_list[idx])

        g.add_edges(valid_edges)

        if len(g.neighborhood(node['name'], mindist=1)) != 0:
            ego_net = g.induced_subgraph(g.neighborhood(node, mindist=1))

            ego_country = affiliations[node]

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

            if node not in entropies:
                entropies[node] = {int(year): to_add}
            else:
                entropies[node][int(year)] = to_add

        for f_node in founded_nodes:
            if f_node in affiliations_from_file:
                affiliations_from_file[f_node]['valid'] = False

to_save = '{}entropies_{}_{}.jsonl'.format(sys.argv[1], sys.argv[2],
                                           int(sys.argv[2]) + 9)

with open(to_save, 'w') as entropies_file:
    for creator in tqdm(entropies.items(), desc='WRITING ENTROPIES JSONL'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, entropies_file)
        entropies_file.write('\n')
