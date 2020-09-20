import gzip
import igraph as ig
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import Counter, defaultdict
from scipy.optimize import curve_fit
from tqdm import tqdm


def power_law(x, b):
    return x**(-b)


matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8

decade = np.arange(int(sys.argv[1]), int(sys.argv[1]) + 10)

authors_affiliations = dict()

file_name = '/home/ricciarelli/mydata/MAG_networks/authors_affiliation.json'

with open(file_name, 'r') as affiliations:
    for affiliation in tqdm(affiliations, desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a, to_delete = json.loads(affiliation.strip()), list()

            authors_affiliations.update(a)

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

    if len(sys.argv) == 2 or (len(sys.argv) > 2 and sys.argv[-1] == str(year)):
        with gzip.open(file_name, 'r') as es_list:
            for edge in tqdm(es_list,
                             desc='YEAR {}: READING EDGES LIST'.format(year)):
                e = edge.decode().strip().split(',')

                nodes_list.add(e[0])
                nodes_list.add(e[1])
                edges_list.append((e[0], e[1]))
                weights_list.append(int(e[2]))

        valid_nodes = dict()

        for node in tqdm(nodes_list, desc='YEAR {}: ADDING NODES'
                         .format(year)):
            valid_nodes[node] = False

            affiliation = list()

            if node in authors_affiliations:
                for aff_id in authors_affiliations[node]:
                    _from = authors_affiliations[node][aff_id]['from']
                    _to = authors_affiliations[node][aff_id]['to']

                    years_range = np.arange(_from, _to + 1)

                    if year in years_range and \
                            aff_id in affiliations_countries:
                        affiliation.append(affiliations_countries[aff_id])

                if len(affiliation) != 0:
                    affiliation = Counter(affiliation)

                    g.add_vertex(node,
                                 affiliation=affiliation.most_common(1)[0][0])
                    valid_nodes[node] = True

        valid_edges, valid_weights = list(), list()

        for idx, edge in tqdm(enumerate(edges_list),
                              desc='YEAR {}: VALIDATING EDGES'.format(year)):
            if edge[0] in authors_affiliations and \
                    edge[1] in authors_affiliations:
                if valid_nodes[edge[0]] and valid_nodes[edge[1]]:
                    valid_edges.append(edges_list[idx])
                    valid_weights.append(weights_list[idx])

        g.add_edges(valid_edges)
        g.es['weight'] = valid_weights

    if len(sys.argv) == 2:
        statistics['Nodes'].append(len(g.vs))
        statistics['Edges'].append(len(g.es))
        statistics['Density'].append(g.density())
    else:
        if sys.argv[-1] == str(year):
            degrees = Counter(g.degree())

            if 0 in degrees:
                del degrees[0]

            probabilities = list()

            for degree in sorted(degrees):
                probabilities.append(degrees[degree] / len(g.vs))

            pars, cov = curve_fit(f=power_law, xdata=sorted(degrees),
                                  ydata=probabilities)
            stdevs = np.sqrt(np.diag(cov))

            fig, ax = plt.subplots()
            ax.scatter(sorted(degrees), probabilities, color='steelblue',
                       edgecolor='steelblue', alpha=0.7)
            ax.plot(sorted(degrees), power_law(sorted(degrees), pars[0]),
                    linestyle='--', color='black', alpha=0.8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(r'Probability Density Distribution'
                         r' - Year {} - $\gamma = {} \pm {}$'
                         .format(year, np.round(pars[0], 2),
                                 np.round(stdevs[0], 2)))
            ax.set_xlabel(r'$k$')
            ax.set_ylabel(r'$P_k$')
            ax.tick_params(axis='both', which='major')
            ax.set_ylim(1e-5, 2)
            fig.savefig('./images/degree_distribution_{}.pdf'.format(year),
                        format='pdf')
            plt.close(fig=fig)

    # avg_clustering_coefficient = list()
    # transitivity = list()
    # diameter, radius = list(), list()

    # for conn_component in \
    #     tqdm(g.components(),
    #          desc='YEAR {}: ANALYZING CONNECTED COMPONENTS'.format(year)):
    #     subgraph = g.induced_subgraph(conn_component)

    #     avg_clustering_coefficient\
    #         .append(subgraph.transitivity_avglocal_undirected(mode='zero'))
    #     transitivity.append(subgraph.transitivity_undirected(mode='zero'))
    #     diameter.append(subgraph.diameter(directed='False'))
    #     radius.append(subgraph.radius())

    # statistics['Average CC'].append(
    #     np.round(np.mean(avg_clustering_coefficient), 2))
    # statistics['Transitivity'].append(np.round(np.mean(transitivity), 2))
    # statistics['Diameter'].append(np.round(np.mean(diameter), 2))
    # statistics['Radius'].append(np.round(np.mean(radius), 2))

if len(sys.argv) == 2:
    pd.DataFrame(data=statistics, index=decade)\
        .to_csv('/home/ricciarelli/mydata/MAG_networks/networks_statistics/'
                '{}_{}_statistics.csv'.format(decade[0], decade[-1]))
