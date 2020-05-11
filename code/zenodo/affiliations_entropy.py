import igraph as ig
import json
import numpy as np
import os
import sys

from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm

frameworks_by_year = dict()

for framework in os.listdir(sys.argv[1]):
    years = [y for y in os.listdir(sys.argv[1] + framework)
             if os.path.isdir(sys.argv[1] + framework + '/' + y)]

    for year in years:
        if year not in frameworks_by_year:
            frameworks_by_year[year] = [framework]
        else:
            frameworks_by_year[year].append(framework)

for year in frameworks_by_year:
    entropies = dict()
    g = ig.Graph()

    nodes = dict()
    edges_list, weights_list, frameworks_list = list(), list(), list()

    for framework in frameworks_by_year[year]:
        file_name = sys.argv[1] + framework + '/' + year + '/' + \
            'node_list.jsonl'

        with open(file_name, 'r') as node_list:
            for node in tqdm(node_list,
                             desc='YEAR {}: IMPORTING FRAMEWORK {} NODES'
                             .format(year, framework)):
                n = json.loads(node.strip())
                n = [i for i in n.items()][0]

                if n[0] not in nodes:
                    nodes[n[0]] = [n[1]]
                else:
                    nodes[n[0]].append(n[1])

        file_name = sys.argv[1] + framework + '/' + year + '/' + \
            'edge_list.jsonl'

        with open(file_name, 'r') as edge_list:
            for edge in tqdm(edge_list,
                             desc='YEAR {}: IMPORTING FRAMEWORK {} EDGES'
                             .format(year, framework)):
                e = json.loads(edge.strip())

                for node_1 in e:
                    for node_2 in e[node_1]:
                        edges_list.append((node_1, node_2))
                        weights_list.append(len(e[node_1][node_2]))
                        frameworks_list.append(framework)

    for node in tqdm(nodes, desc='YEAR {}: ADDING NODES'.format(year)):
        affiliation = Counter(nodes[node])

        g.add_vertex(node, affiliation=affiliation.most_common(1)[0][0])

    g.add_edges(edges_list)
    g.es['weight'] = weights_list
    g.es['framework'] = frameworks_list

    import ipdb
    ipdb.set_trace()

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

            entropies[node['name']] = to_add

# dirs = [d for d in os.listdir(sys.argv[1])
#         if os.path.isdir(sys.argv[1] + d)]

# for year in sorted(dirs):
#     entropies = dict()
#     g = ig.Graph()

#     if os.path.exists(sys.argv[1] + year + '/' + 'node_list.jsonl') and \
#        os.path.exists(sys.argv[1] + year + '/' + 'edge_list.jsonl'):
#         file_name = sys.argv[1] + year + '/' + 'node_list.jsonl'

#         with open(file_name, 'r') as node_list:
#             for node in tqdm(node_list,
#                              desc='YEAR {}: READING NODE LIST'.format(year)):
#                 n = json.loads(node.strip())
#                 n = [i for i in n.items()][0]

#                 g.add_vertex(n[0], affiliation=n[1])

#         edges_list, weights_list = list(), list()

#         file_name = sys.argv[1] + year + '/' + 'edge_list.jsonl'

#         with open(file_name, 'r') as edge_list:
#             for edge in tqdm(edge_list,
#                              desc='YEAR {}: READING EDGE LIST'.format(year)):
#                 e = json.loads(edge.strip())

#                 for node_1 in e:
#                     for node_2 in e[node_1]:
#                         edges_list.append((node_1, node_2))
#                         weights_list.append(len(e[node_1][node_2]))

#         g.add_edges(edges_list)
#         g.es['weight'] = weights_list

#         for node in tqdm(g.vs,
#                          desc='YEAR {}: COMPUTING ENTROPIES'.format(year)):
#             if len(g.neighborhood(node['name'], mindist=1)) != 0:
#                 ego_net = g.induced_subgraph(g.neighborhood(node['name'],
#                                              mindist=1))

#                 ego_country = node['affiliation']

#                 affiliations_types = list()
#                 monst_common_class = None

#                 for neighbor in ego_net.vs:
#                     neighbor_country = neighbor['affiliation']

#                     if neighbor_country == ego_country:
#                         affiliations_types.append('same')
#                     else:
#                         affiliations_types.append('different')

#                 affiliations_types = Counter(affiliations_types)
#                 monst_common_class = affiliations_types.most_common(1)[0][0]
#                 affiliations_types = \
#                     [affiliations_types[k] / sum(affiliations_types.values())
#                      for k in affiliations_types]

#                 to_add = dict()

#                 if monst_common_class == 'same':
#                     to_add['entropy'] = \
#                         np.round(entropy(affiliations_types, base=2), 2)
#                 else:
#                     to_add['entropy'] = \
#                         np.round(entropy(affiliations_types, base=2), 2) * -1

#                 to_add['class'] = monst_common_class
#                 to_add['affiliation'] = ego_country

#                 entropies[node['name']] = to_add

#         to_save = sys.argv[1] + year + '/entropies.jsonl'

#         with open(to_save, 'w') as entropies_file:
#             for creator in tqdm(entropies.items(),
#                                 desc='YEAR {}: WRITING ENTROPIES JSONL'
#                                 .format(year)):
#                 to_write = dict()
#                 to_write[creator[0]] = creator[1]

#                 json.dump(to_write, entropies_file)
#                 entropies_file.write('\n')
