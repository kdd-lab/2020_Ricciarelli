import gzip
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import Counter
from tqdm import tqdm

nets = '../../datasets/coauthors_networks/network_weight/'


years = np.arange(int(sys.argv[1]), int(sys.argv[2]))

nodes, edges, densities, avg_cc, transitivities, diameters, rads = \
    list(), list(), list(), list(), list(), list(), list()

for year in years:
    print('\nPROCESSING YEAR ' + str(year))

    g = ig.Graph()

    ns, es, ws = list(), list(), list()

    with gzip.open(nets + str(year) + '.gz', 'r') as edge_list:
        for record in tqdm(edge_list, desc='READING FILE'):
            params = record.decode('utf-8').strip().split(',')

            ns.append(params[0])
            ns.append(params[1])
            es.append((params[0], params[1]))
            ws.append(int(params[2]))

    ns = set(ns)

    for n in tqdm(ns, desc='ADDING NODES'):
        g.add_vertex(n)

    g.add_edges(es)
    g.es['weight'] = ws

    nodes_number = g.vcount()
    edges_number = g.ecount()
    density = g.density()

    if sys.argv[3] == 'Y':
        avg_clustering_coefficient = list()
        transitivity = list()
        diameter, radius = list(), list()

        connected_components = g.components()

        for cc in tqdm(connected_components,
                       desc='ANALYZING CONNECTED COMPONENTS'):
            sub = g.induced_subgraph(cc)

            diameter.append(sub.diameter())
            radius.append(sub.radius())
            avg_clustering_coefficient.append(
                sub.transitivity_undirected())
            transitivity.append(
                sub.transitivity_avglocal_undirected(mode='zero',
                                                     weights='weight'))

        avg_clustering_coefficient = np.mean(avg_clustering_coefficient)
        transitivity = np.mean(transitivity)
        diameter = np.mean(diameter)
        radius = np.mean(radius)

        nodes.append(np.format_float_scientific(nodes_number, 2))
        edges.append(np.format_float_scientific(edges_number, 2))
        densities.append(np.format_float_scientific(density, 2))
        avg_cc.append(
            np.format_float_scientific(avg_clustering_coefficient, 2))
        transitivities.append(np.format_float_scientific(transitivity, 2))
        diameters.append(np.format_float_scientific(diameter, 2))
        rads.append(np.format_float_scientific(radius, 2))

    if sys.argv[4] == 'Y':
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
        fig.savefig('../../datasets/weighted_coauthors_networks/{}/'
                    'degree_distribution_probability_distribution_{}.pdf'
                    .format(years[0], year), format='pdf')
        plt.close(fig=fig)

        ws = sorted(ws)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        fig.suptitle('Weights Distribution & Probability Density, Year '
                     + str(year), fontsize=20)

        axs[0].hist(ws, log=True, zorder=2, color='#3296dc',
                    edgecolor='#1f77b4')
        axs[0].set_title('Weight Distribution Histogram', fontsize=14)
        axs[0].set_xlabel(r'$w$', fontsize=14)
        axs[0].set_ylabel(r'$N_w$', fontsize=14)
        axs[0].grid(axis='y', linestyle='--', color='black', zorder=1)
        axs[0].text(0.65, 0.75, 'mean = {}\nmax = {}\nmin = {}'
                    .format(round(np.mean(ws), 2), max(ws),
                            min(ws)),
                    bbox=dict(facecolor='white', edgecolor='black'),
                    transform=axs[0].transAxes)

        c, countdict_pdf = Counter(ws), dict()

        for weight in np.arange(min(ws), max(ws) + 1):
            countdict_pdf[weight] = (c[weight] / len(ws)) \
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
        fig.savefig('../../datasets/weighted_coauthors_networks/{}/weight_'
                    'distribution_probability_distribution_{}.pdf'
                    .format(years[0], year), format='pdf')
        plt.close(fig=fig)

if sys.argv[3] == 'Y':
    statistics = pd.DataFrame({'Year': years, 'Nodes': nodes, 'Edges': edges,
                               'Density': densities,
                               'Average Clustering Coefficient': avg_cc,
                               'Transitivity': transitivities,
                               'Diameter': diameters, 'Radius': rads})
    statistics.set_index('Year', inplace=True)
    statistics.to_csv(
        '../../datasets/weighted_coauthors_networks/{}/statistics_'
        '{}_{}.csv'.format(years[0], years[0], years[-1]))
