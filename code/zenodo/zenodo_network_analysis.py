import orjson
import networkx as nx
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm

edges_x, edges_y, years, paper_project_titles, weights = list(), \
    list(), list(), list(), list()

with open(sys.argv[1], 'r') as edge_list:
    for edge in tqdm(edge_list, desc='READING EDGE LIST'):
        params = orjson.loads(edge.strip())
        edge_x = list(params.keys())[0]

        for edge_y in params[edge_x]:
            for year in params[edge_x][edge_y]:
                edges_x.append(edge_x)
                edges_y.append(edge_y)
                years.append(year)
                paper_project_titles.append(params[edge_x][edge_y][year])
                weights.append(len(params[edge_x][edge_y][year]))

df = pd.DataFrame({'X': edges_x, 'Y': edges_y, 'Year': years,
                  'Infos': paper_project_titles, 'Weight': weights})

g = nx.Graph()

ys, nodes, edges, densities, avg_cc, transitivities, diameters, rads = \
    list(), list(), list(), list(), list(), list(), list(), list()

for year in sorted(df.Year.unique()):
    subgraph = df[df.Year == year]

    for row in tqdm(subgraph.itertuples(),
                    desc='YEAR {}: BUILDING THE NETWORK'.format(year)):
        g.add_edge(row[1], row[2], weight=int(row[5]))

    nodes_number = g.number_of_nodes()
    edges_number = g.number_of_edges()
    density = nx.density(g)

    avg_clustering_coefficient = list()
    transitivity = list()
    diameter, radius = list(), list()

    connected_components = nx.connected_components(g)

    for cc in tqdm(connected_components,
                   desc='YEAR {}: ANALYZING CONNECTED COMPONENTS'
                   .format(year)):
        if len(cc) > 2 and len(cc) < 51:
            subg = g.subgraph(cc)

            avg_clustering_coefficient.append(
                nx.average_clustering(subg, weight='weight'))
            transitivity.append(nx.transitivity(subg))
            diameter.append(nx.diameter(subg))
            radius.append(nx.radius(subg))

    g.clear()

    ys.append(year)
    nodes.append(nodes_number)
    edges.append(edges_number)
    densities.append(density)
    avg_cc.append(np.mean(avg_clustering_coefficient))
    transitivities.append(np.mean(transitivity))
    diameters.append(np.mean(diameter))
    rads.append(np.mean(radius))

to_save = '../datasets/zenodo/zenodo_{}_networks_statistics.csv' \
    .format(sys.argv[1].split('/')[-1].split('_')[0])

pd.DataFrame({'Year': ys, 'Nodes': nodes, 'Edges': edges,
              'Density': densities, 'Avg Clustering Coefficient': avg_cc,
              'Transitivity': transitivities, 'Diameter': diameters,
              'Radius': rads}).to_csv(to_save, index=False)
