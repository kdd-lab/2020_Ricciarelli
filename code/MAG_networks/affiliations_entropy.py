import igraph as ig
import json
import numpy as np
import os
import sys

from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm

entropies = dict()

for year in np.arange(int(sys.argv[2]), int(sys.argv[2]) + 10):
    dir_name = '{}{}/ego_networks/'.format(sys.argv[1], year)

    for ego_n in tqdm(os.listdir(dir_name),
                      desc='PROCESSING YEAR {}'.format(year)):
        g = ig.Graph()

        ego, ego_country, added_ego = None, None, False

        edges = list()

        with open(dir_name + ego_n, 'r') as ego_n_file:
            for edge in ego_n_file:
                e = edge.strip().split(',')

                if not added_ego:
                    ego, ego_country, added_ego = e[0], e[1], True

                    g.add_vertex(e[0], affiliation=e[1])

                g.add_vertex(e[2], affiliation=e[3])

                edges.append((e[0], e[2]))

        g.add_edges(edges)

        if len(g.neighborhood(ego, mindist=1)) != 0:
            ego_net = g.induced_subgraph(g.neighborhood(ego, mindist=1))

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

            if ego not in entropies:
                entropies[ego] = {int(year): to_add}
            else:
                entropies[ego][int(year)] = to_add

to_save = '{}entropies/entropies_{}_{}.jsonl'.format(sys.argv[1], sys.argv[2],
                                                     int(sys.argv[2]) + 9)

with open(to_save, 'w') as entropies_file:
    for creator in tqdm(entropies.items(), desc='WRITING ENTROPIES JSONL'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, entropies_file)
        entropies_file.write('\n')
