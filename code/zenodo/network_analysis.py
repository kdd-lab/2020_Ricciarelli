import igraph as ig
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        left_y_lim, right_x_lim = None, None

        for deg in np.arange(0, max(degrees) + 1):
            countdict_pdf[deg] = (c[deg] / len(degrees)) if deg in c.keys() \
                else 0.

        min_prob = [i for i in sorted(countdict_pdf.values()) if i != 0.0][0]
        max_degree = max(countdict_pdf.keys())
        es_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        es_2 = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

        for i, v in enumerate(es_1):
            if min_prob < v:
                left_y_lim = es_1[i - 1]
                break

        for i, v in enumerate(es_2):
            if max_degree < v:
                right_x_lim = es_2[i]
                break

        axs[1].scatter(list(countdict_pdf.keys()),
                       list(countdict_pdf.values()), zorder=2, alpha=0.7,
                       color='#3296dc', edgecolor='#1f77b4')
        axs[1].set_title('Probability Density Distribution', fontsize=14)
        axs[1].set_xscale('log')
        axs[1].set_xlim(0.5, right_x_lim)
        axs[1].set_yscale('log')
        axs[1].set_ylim(left_y_lim, 1.)
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
        left_y_lim, right_x_lim = None, None

        for weight in np.arange(min(weights), max(weights) + 1):
            countdict_pdf[weight] = (c[weight] / len(weights)) \
                if weight in c.keys() else 0.

        min_prob = [i for i in sorted(countdict_pdf.values()) if i != 0.0][0]
        max_degree = max(countdict_pdf.keys())

        for i, v in enumerate(es_1):
            if min_prob < v:
                left_y_lim = es_1[i - 1]
                break

        for i, v in enumerate(es_2):
            if max_degree < v:
                right_x_lim = es_2[i]
                break

        axs[1].scatter(list(countdict_pdf.keys()),
                       list(countdict_pdf.values()), zorder=2, alpha=0.7,
                       color='#3296dc', edgecolor='#1f77b4')
        axs[1].set_title('Probability Density Distribution', fontsize=14)
        axs[1].set_xscale('log')
        axs[1].set_xlim(0.5, right_x_lim)
        axs[1].set_yscale('log')
        axs[1].set_ylim(left_y_lim, 1.)
        axs[1].set_xlabel(r'$w$', fontsize=14)
        axs[1].set_ylabel(r'$p_w$', fontsize=14)
        axs[1].grid(axis='y', linestyle='--', color='black', zorder=1)

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.savefig(sys.argv[1] + year + '/'
                    'weight_distribution_probability_distribution.pdf',
                    format='pdf')
        plt.close(fig=fig)

        betweenness = g.betweenness(directed=False)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        fig.suptitle("Node's Betweenness Distribution, Year "
                     + str(year), fontsize=20)

        axs.hist(betweenness, log=True, zorder=2, color='#3296dc',
                 edgecolor='#1f77b4')
        axs.set_xlabel(r'$g$', fontsize=14)
        axs.set_ylabel(r'$N_g$', fontsize=14)
        axs.grid(axis='y', linestyle='--', color='black', zorder=1)
        axs.text(1.05, 0.9, 'mean = {}\nmax = {}\nmin = {}'
                 .format(np.format_float_scientific(np.mean(betweenness),
                                                    precision=2),
                         np.format_float_scientific(max(betweenness),
                                                    precision=2),
                         np.format_float_scientific(min(betweenness),
                                                    precision=2)),
                 bbox=dict(facecolor='white', edgecolor='black'),
                 transform=axs.transAxes, fontsize=14)

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.savefig(sys.argv[1] + year + '/'
                    'node_betweenness_distribution.pdf', format='pdf')
        plt.close(fig=fig)

        closeness = g.closeness()

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        fig.suptitle("Node's Closeness Distribution, Year "
                     + str(year), fontsize=20)

        axs.hist(closeness, log=True, zorder=2, color='#3296dc',
                 edgecolor='#1f77b4')
        axs.set_xlabel(r'$C$', fontsize=14)
        axs.set_ylabel(r'$N_C$', fontsize=14)
        axs.grid(axis='y', linestyle='--', color='black', zorder=1)
        axs.text(1.05, 0.9, 'mean = {}\nmax = {}\nmin = {}'
                 .format(np.format_float_scientific(np.mean(closeness),
                                                    precision=2),
                         np.format_float_scientific(max(closeness),
                                                    precision=2),
                         np.format_float_scientific(min(closeness),
                                                    precision=2)),
                 bbox=dict(facecolor='white', edgecolor='black'),
                 transform=axs.transAxes, fontsize=14)

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.savefig(sys.argv[1] + year + '/'
                    'node_closeness_distribution.pdf', format='pdf')
        plt.close(fig=fig)

to_save = sys.argv[1] + 'networks_statistics.csv'

df = pd.DataFrame({'Year': years, 'Nodes': nodes, 'Edges': edges,
                   'Density': densities, 'Avg Clustering Coefficient': avg_cc,
                   'Transitivity': transitivities, 'Diameter': diameters,
                   'Radius': rads})
pos_list = [i * 2 for i in np.arange(0, len(df['Year'].values))]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

fig.suptitle('Nodes and Edges growth by Year', fontsize=20)

axs.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
axs.xaxis.set_major_formatter(ticker.FixedFormatter((df['Year'].values)))
df.plot(kind='line', x='Year', y='Nodes', grid=True, logy=True,
        color='#3296dc', linewidth=2, ax=axs)
df.plot(kind='line', x='Year', y='Edges', grid=True, logy=True,
        color='#dc3241', linewidth=2, ax=axs)
axs.set_xlabel('Year', fontsize=14)

fig.savefig(sys.argv[1] + '/nodes_and_edges_growth_by_year.pdf', format='pdf')
plt.close(fig=fig)

df.to_csv(to_save, index=False)
