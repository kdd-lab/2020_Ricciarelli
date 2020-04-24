import igraph as ig
import json
import numpy as np
import pandas as pd
import sys

from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm

creators, entropies = dict(), dict()

with open(sys.argv[1], 'r') as node_list:
    for creator in tqdm(node_list, desc='READING NODE LIST'):
        c = json.loads(creator.strip())

        creators[list(c.keys())[0]] = list(c.values())[0]

edges_x, edges_y, years, project_titles, weights = list(), \
    list(), list(), list(), list()

with open(sys.argv[2], 'r') as edge_list:
    for edge in tqdm(edge_list, desc='READING EDGE LIST'):
        params = json.loads(edge.strip())
        edge_x = list(params.keys())[0]

        for edge_y in params[edge_x]:
            for year in params[edge_x][edge_y]:
                edges_x.append(edge_x)
                edges_y.append(edge_y)
                years.append(year)
                project_titles.append(params[edge_x][edge_y][year])
                weights.append(len(params[edge_x][edge_y][year]))

df = pd.DataFrame({'X': edges_x, 'Y': edges_y, 'Year': years,
                  'Infos': project_titles, 'Weight': weights})

for year in sorted(df.Year.unique()):
    subgraph = df[df.Year == year]
    creators_by_year = set(subgraph.X.values).union(set(subgraph.Y.values))

    g = ig.Graph()

    for n in tqdm(creators_by_year,
                  desc='YEAR {}: BUILDING NETWORK'.format(year)):
        g.add_vertex(n, affiliations=creators[n])

    edges = list()
    weights = list()

    for row in subgraph.itertuples():
        edges.append((row[1], row[2]))
        weights.append(int(row[5]))

    g.add_edges(edges)
    g.es['weight'] = weights

    for node in tqdm(g.vs, desc='YEAR {}: COMPUTING ENTROPIES'.format(year)):
        if len(g.neighborhood(node['name'], mindist=1)) != 0:
            ego_net = g.induced_subgraph(g.neighborhood(node['name'],
                                                        mindist=1))

            ego_country = None

            if type(node['affiliations'][year]) == str:
                ego_country = node['affiliations'][year]
            else:
                ego_country = Counter(node['affiliations'][year])
                ego_country = ego_country.most_common(1)[0][0]

            affiliations_types = list()
            monst_common_class = None

            for neighbor in ego_net.vs:
                if year in neighbor['affiliations']:
                    neighbor_country = None

                    if type(neighbor['affiliations'][year]) == str:
                        neighbor_country = neighbor['affiliations'][year]
                    else:
                        neighbor_country = \
                            Counter(neighbor['affiliations'][year])
                        neighbor_country = \
                            neighbor_country.most_common(1)[0][0]

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
            to_add['entropy'] = np.round(entropy(affiliations_types), 2) \
                if monst_common_class == 'same' else \
                np.round(entropy(affiliations_types), 2) * -1
            to_add['class'] = monst_common_class

            if node['name'] not in entropies:
                entropies[node['name']] = {year: to_add}
            else:
                entropies[node['name']][year] = to_add

framework = sys.argv[1].split('/')[-1].split('_')[0]
path = '../../datasets/zenodo/{}/zenodo_{}_entropies.jsonl'.format(framework,
                                                                   framework)

with open(path, 'w') as entropies_file:
    for creator in tqdm(entropies.items(), desc='WRITING ENTROPIES JSONL'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, entropies_file)
        entropies_file.write('\n')
