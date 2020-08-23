import gzip
import igraph as ig
import json
import numpy as np
import pandas as pd
import sys

from collections import Counter, defaultdict
from tqdm import tqdm

decade = np.arange(int(sys.argv[1]), int(sys.argv[1]) + 10)

authors_affiliations = dict()

file_name = '/home/ricciarelli/mydata/MAG_networks/authors_affiliation.json'

with open(file_name, 'r') as affiliations:
    for affiliation in tqdm(affiliations, desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a, to_delete = json.loads(affiliation.strip()), list()

            authors_affiliations.update(a)
            #for mag_id in a:
            #    for aff_id in a[mag_id]:
            #        if a[mag_id][aff_id]['from'] not in decade:
            #            to_delete.append(aff_id)

            #for mag_id in a:
            #    for aff_id in to_delete:
            #        del a[mag_id][aff_id]

            #if len(a[mag_id]) != 0:
            #    authors_affiliations.update(a)

affiliations_countries = dict()

file_name = '/home/ricciarelli/mydata/MAG_networks/affiliations_geo.txt'

with open(file_name, 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries.update({a[0]: a[1]})

statistics = defaultdict(list)

for year in decade:
    g = ig.Graph()

    file_name = '/home/ricciarelli/mydata/MAG_networks/{}/{}.gz'\
        .format(year, year)

    nodes_list, edges_list, weights_list = set(), list(), list()

    with gzip.open(file_name, 'r') as es_list:
        for edge in tqdm(es_list,
                         desc='YEAR {}: READING EDGES LIST'.format(year)):
            e = edge.decode().strip().split(',')

            nodes_list.add(e[0])
            nodes_list.add(e[1])
            edges_list.append((e[0], e[1]))
            weights_list.append(int(e[2]))

    valid_nodes = dict()

    for node in tqdm(nodes_list, desc='YEAR {}: ADDING NODES'.format(year)):
        valid_nodes[node] = False

        affiliation = list()

        if node in authors_affiliations:
            for aff_id in authors_affiliations[node]:
                _from = authors_affiliations[node][aff_id]['from']
                _to = authors_affiliations[node][aff_id]['to']

                years_range = np.arange(_from, _to + 1)

                if year in years_range and aff_id in affiliations_countries:
                    affiliation.append(affiliations_countries[aff_id])

            if len(affiliation) != 0:
                affiliation = Counter(affiliation)

                g.add_vertex(node,
                             affiliation=affiliation.most_common(1)[0][0])
                valid_nodes[node] = True

    valid_edges, valid_weights = list(), list()

    for idx, edge in tqdm(enumerate(edges_list),
                          desc='YEAR {}: VALIDATING EDGES'.format(year)):
        if edge[0] in authors_affiliations and edge[1] in authors_affiliations:
            if valid_nodes[edge[0]] and valid_nodes[edge[1]]:
                valid_edges.append(edges_list[idx])
                valid_weights.append(weights_list[idx])

    g.add_edges(valid_edges)
    g.es['weight'] = valid_weights

    statistics['Nodes'].append(len(g.vs))
    statistics['Edges'].append(len(g.es))
    statistics['Density'].append(g.density())

    avg_clustering_coefficient = list()
    transitivity = list()
    diameter, radius = list(), list()

    for conn_component in \
        tqdm(g.components(),
             desc='YEAR {}: ANALYZING CONNECTED COMPONENTS'.format(year)):
        subgraph = g.induced_subgraph(conn_component)

        avg_clustering_coefficient\
            .append(subgraph.transitivity_avglocal_undirected(mode='zero'))
        transitivity.append(subgraph.transitivity_undirected(mode='zero'))
        diameter.append(subgraph.diameter(directed='False'))
        radius.append(subgraph.radius())

    statistics['Average CC'].append(
        np.round(np.mean(avg_clustering_coefficient), 2))
    statistics['Transitivity'].append(np.round(np.mean(transitivity), 2))
    statistics['Diameter'].append(np.round(np.mean(diameter), 2))
    statistics['Radius'].append(np.round(np.mean(radius), 2))

pd.DataFrame(data=statistics, index=decade)\
    .to_csv('/home/ricciarelli/mydata/MAG_networks/networks_statistics/'
            '{}_{}_statistics.csv'.format(decade[0], decade[-1]))
