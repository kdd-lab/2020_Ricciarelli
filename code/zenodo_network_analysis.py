import networkx as nx
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm

edges_x, edges_y, years, weights = list(), list(), list(), list()

with open(sys.argv[1], 'r') as edge_list:
    for edge in tqdm(edge_list, desc='READING EDGE LIST'):
        params = edge.strip().split(' ')

        edges_x.append(params[0])
        edges_y.append(params[1])
        years.append(params[2])
        weights.append(params[3])

df = pd.DataFrame({'X': edges_x, 'Y': edges_y, 'Year': years,
                  'Weight': weights})

net_2010 = df[df.Year == '2010']
g = nx.Graph()

for row in tqdm(net_2010.itertuples(), desc='BUILDING THE NETWORK'):
    g.add_edge(row[1], row[2], year=row[3], weight=int(row[4]))

nodes_number = g.number_of_nodes()
edges_number = g.number_of_edges()
density = nx.density(g)

avg_clustering_coefficient = list()
transitivity = list()
diameter, radius = list(), list()

connected_components = list(nx.connected_components(g))
connected_components = [cc for cc in connected_components if len(cc) > 2]

for cc in tqdm(connected_components, desc='ANALYZING CONNECTED COMPONENTS'):
    subgraph = g.subgraph(cc)

    avg_clustering_coefficient.append(
        nx.average_clustering(subgraph, weight='weight'))
    transitivity.append(nx.transitivity(subgraph))
    diameter.append(nx.diameter(subgraph))
    radius.append(nx.radius(subgraph))

print(nodes_number, edges_number, density, np.mean(avg_clustering_coefficient),
      np.mean(transitivity), np.mean(diameter), np.mean(radius))
