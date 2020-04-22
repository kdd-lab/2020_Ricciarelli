import igraph as ig
import json
import numpy as np
import pandas as pd
import sys

from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm

creators, international_creators, entropies = dict(), list(), dict()

with open(sys.argv[1], 'r') as node_list:
    for creator in tqdm(node_list, desc='READING NODE LIST'):
        c = json.loads(creator.strip())

        creators[list(c.keys())[0]] = list(c.values())[0]

        if len(list(c.values())[0].keys()) > 1:
            international_creators.append(list(c.keys())[0])

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
    international_creators_by_year = creators_by_year. \
        intersection(set(international_creators))
    probs = list()

    if len(international_creators_by_year) > 0:
        g = ig.Graph()

        for n in tqdm(creators_by_year,
                      desc='YEAR {}: ADDING NODES'.format(year)):
            g.add_vertex(n, affiliations=creators[n])

        edges = list()
        weights = list()

        for row in subgraph.itertuples():
            edges.append((row[1], row[2]))
            weights.append(int(row[5]))

        g.add_edges(edges)
        g.es['weight'] = weights

        for creator in tqdm(international_creators_by_year,
                            desc='YEAR {}: COMPUTING ENTROPIES'.format(year)):
            ego_net = g.induced_subgraph(g.neighborhood(creator))

            for neighbor in ego_net.neighborhood(creator, mindist=1):
                affiliations = ego_net.vs.find(neighbor)['affiliations']

                for country in affiliations:
                    if year in affiliations[country]:
                        probs.append(country)

            probs = Counter(probs)
            probs = [probs[cn] / sum(probs.values()) for cn in probs]

            if creator not in entropies:
                entropies[creator] = {year: np.round(entropy(probs), 2)}
            else:
                entropies[creator][year] = np.round(entropy(probs), 2)

framework = sys.argv[1].split('/')[-1].split('_')[0]
path = '../../datasets/zenodo/{}/zenodo_{}_entropies.jsonl'.format(framework,
                                                                   framework)

with open(path, 'w') as entropies_file:
    for creator in tqdm(entropies.items(), desc='WRITING ENTROPIES JSONL'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, entropies_file)
        entropies_file.write('\n')
