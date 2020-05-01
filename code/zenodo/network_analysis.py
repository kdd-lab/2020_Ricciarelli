import igraph as ig
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import warnings

from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore")

years, nodes, edges, densities, avg_cc, transitivities, diameters, rads = \
    list(), list(), list(), list(), list(), list(), list(), list()

dirs = [d for d in os.listdir(sys.argv[1])
        if os.path.isdir(sys.argv[1] + d)]

for year in sorted(dirs):
    g = ig.Graph()

    if os.path.exists(sys.argv[1] + year + '/' + 'node_list.jsonl') and \
       os.path.exists(sys.argv[1] + year + '/' + 'edge_list.jsonl'):
        file_name = sys.argv[1] + year + '/' + 'node_list.jsonl'

        with open(file_name, 'r') as node_list:
            for node in tqdm(node_list,
                             desc='YEAR {}: READING NODE LIST'.format(year)):
                n = json.loads(node.strip())
                n = [i for i in n.items()][0]

                g.add_vertex(n[0], affiliation=n[1])

        edges_list, weights_list = list(), list()

        file_name = sys.argv[1] + year + '/' + 'edge_list.jsonl'

        with open(file_name, 'r') as edge_list:
            for edge in tqdm(edge_list,
                             desc='YEAR {}: READING EDGE LIST'.format(year)):
                e = json.loads(edge.strip())

                for node_1 in e:
                    for node_2 in e[node_1]:
                        edges_list.append((node_1, node_2))
                        weights_list.append(len(e[node_1][node_2]))

        g.add_edges(edges_list)
        g.es['weight'] = weights_list

        nodes_number = len(g.vs)
        edges_number = len(g.es)
        density = g.density()

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

        years.append(year)
        nodes.append(nodes_number)
        edges.append(edges_number)
        densities.append(density)
        avg_cc.append(np.round(np.mean(avg_clustering_coefficient), 2))
        transitivities.append(np.round(np.mean(transitivity), 2))
        diameters.append(np.round(np.mean(diameter), 2))
        rads.append(np.round(np.mean(radius), 2))

        degrees = sorted(g.degree())

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        fig.suptitle('Degree Distribution & Probability Density, Year ' +
                     str(year), fontsize=20)

        axs[0].hist(degrees, log=True, zorder=2, color='#3296dc',
                    edgecolor='#1f77b4')
        axs[0].set_title('Degree Distribution Histogram', fontsize=14)
        axs[0].set_xlabel(r'$k$', fontsize=14)
        axs[0].set_ylabel(r'$N_k$', fontsize=14)
        axs[0].grid(axis='y', linestyle='--', color='black', zorder=1)
        axs[0].text(0.65, 0.75, 'mean = {}\nmax = {}\nmin = {}'
                    .format(round(np.mean(degrees), 2), max(degrees),
                            min(degrees)),
                    bbox=dict(facecolor='white', edgecolor='black'),
                    transform=axs[0].transAxes)

        c, countdict_pdf = Counter(degrees), dict()

        for deg in np.arange(min(degrees), max(degrees) + 1):
            countdict_pdf[deg] = (c[deg] / len(degrees)) if deg in c.keys() \
                else 0.

        axs[1].scatter(list(countdict_pdf.keys()),
                       list(countdict_pdf.values()), zorder=2, alpha=0.7,
                       color='#3296dc', edgecolor='#1f77b4')
        axs[1].set_title('Probability Density Distribution', fontsize=14)
        axs[1].set_xscale('log')
        axs[1].set_xlim(min(countdict_pdf.keys()),
                        max(countdict_pdf.keys()) + 50)
        axs[1].set_yscale('log')
        axs[1].set_ylim(1e-7, 1.)
        axs[1].set_xlabel(r'$k$', fontsize=14)
        axs[1].set_ylabel(r'$p_k$', fontsize=14)
        axs[1].grid(axis='y', linestyle='--', color='black', zorder=1)

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.savefig(sys.argv[1] + year + '/'
                    'degree_distribution_probability_distribution.pdf',
                    format='pdf')
        plt.close(fig=fig)

        weights = sorted(weights_list)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        fig.suptitle('Weights Distribution & Probability Density, Year '
                     + str(year), fontsize=20)

        axs[0].hist(weights, log=True, zorder=2, color='#3296dc',
                    edgecolor='#1f77b4')
        axs[0].set_title('Weight Distribution Histogram', fontsize=14)
        axs[0].set_xlabel(r'$w$', fontsize=14)
        axs[0].set_ylabel(r'$N_w$', fontsize=14)
        axs[0].grid(axis='y', linestyle='--', color='black', zorder=1)
        axs[0].text(0.65, 0.75, 'mean = {}\nmax = {}\nmin = {}'
                    .format(round(np.mean(weights), 2), max(weights),
                            min(weights)),
                    bbox=dict(facecolor='white', edgecolor='black'),
                    transform=axs[0].transAxes)

        c, countdict_pdf = Counter(weights), dict()

        for weight in np.arange(min(weights), max(weights) + 1):
            countdict_pdf[weight] = (c[weight] / len(weights)) \
                if weight in c.keys() else 0.

        axs[1].scatter(list(countdict_pdf.keys()),
                       list(countdict_pdf.values()), zorder=2, alpha=0.7,
                       color='#3296dc', edgecolor='#1f77b4')
        axs[1].set_title('Probability Density Distribution', fontsize=14)
        axs[1].set_xscale('log')
        axs[1].set_xlim(min(countdict_pdf.keys()),
                        max(countdict_pdf.keys()) + 50)
        axs[1].set_yscale('log')
        axs[1].set_ylim(1e-7, 1.)
        axs[1].set_xlabel(r'$w$', fontsize=14)
        axs[1].set_ylabel(r'$p_w$', fontsize=14)
        axs[1].grid(axis='y', linestyle='--', color='black', zorder=1)

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.savefig(sys.argv[1] + year + '/'
                    'weight_distribution_probability_distribution_{}.pdf'
                    .format(years[0], year), format='pdf')
        plt.close(fig=fig)

to_save = sys.argv[1] + 'networks_statistics.csv'

pd.DataFrame({'Year': years, 'Nodes': nodes, 'Edges': edges,
              'Density': densities, 'Avg Clustering Coefficient': avg_cc,
              'Transitivity': transitivities, 'Diameter': diameters,
              'Radius': rads}).to_csv(to_save, index=False)
