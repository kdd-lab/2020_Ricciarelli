import igraph as ig
import json
import numpy as np
import os
import pandas as pd
import sys

from tqdm import tqdm

years, nodes, edges, densities, avg_cc, transitivities, diameters, rads = \
    list(), list(), list(), list(), list(), list(), list(), list()

dirs = [d for d in os.listdir(sys.argv[1]) if d != '.DS_Store']

for year in sorted(dirs):
    g = ig.Graph()

    with open(sys.argv[1] + year + '/' + 'node_list.jsonl', 'r') as node_list:
        for node in tqdm(node_list,
                         desc='YEAR {}: READING NODE LIST'.format(year)):
            n = json.loads(node.strip())
            n = [i for i in n.items()][0]

            g.add_vertex(n[0], affiliation=n[1])

    with open(sys.argv[1] + year + '/' + 'edge_list.jsonl', 'r') as edge_list:
        for edge in tqdm(edge_list,
                         desc='YEAR {}: READING EDGE LIST'.format(year)):
            e = json.loads(edge.strip())

            for node_1 in e:
                for node_2 in e[node_1]:
                    g.add_edge(node_1, node_2, weight=len(e[node_1][node_2]))

    nodes_number = len(g.vs)
    edges_number = len(g.es)
    density = np.round(g.density(), 2)

    avg_clustering_coefficient = list()
    transitivity = list()
    diameter, radius = list(), list()

    for conn_component in tqdm(g.components(),
                               desc='YEAR {}: ANALYZING CONNECTED COMPONENTS'
                               .format(year)):
        subgraph = g.induced_subgraph(conn_component)

        avg_clustering_coefficient\
            .append(subgraph.transitivity_avglocal_undirected(mode='zero'))
        transitivity.append(subgraph.transitivity_undirected(mode='zero'))
        diameter.append(subgraph.diameter(directed='False'))
        radius.append(subgraph.radius())

    years.append(year)
    nodes.append(nodes_number)
    edges.append(edges_number)
    densities.append(density)
    avg_cc.append(np.round(np.mean(avg_clustering_coefficient), 2))
    transitivities.append(np.round(np.mean(transitivity), 2))
    diameters.append(np.round(np.mean(diameter), 2))
    rads.append(np.round(np.mean(radius), 2))

to_save = sys.argv[1] + 'networks_statistics.csv'

pd.DataFrame({'Year': years, 'Nodes': nodes, 'Edges': edges,
              'Density': densities, 'Avg Clustering Coefficient': avg_cc,
              'Transitivity': transitivities, 'Diameter': diameters,
              'Radius': rads}).to_csv(to_save, index=False)
