import gzip
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

nodes, edges, densities, avg_cc, transitivities, diameters, rads = \
    list(), list(), list(), list(), list(), list(), list()

betweennesses, closenesses = list(), list()

years = [d for d in os.listdir(sys.argv[1]) if os.path.isdir(sys.argv[1] + d)]

authors_affiliations = dict()

file_name = sys.argv[1] + '/' + 'authors_affiliation.json'

with open(file_name, 'r') as authors_affiliations_file:
    for affiliation in tqdm(authors_affiliations_file,
                            desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a = list(json.loads(affiliation.strip()).items())[0]

            a[1]['valid'] = False
            authors_affiliations[a[0]] = a[1]

affiliations_countries = dict()

file_name = sys.argv[1] + '/' + 'affiliations_geo.txt'

with open(file_name, 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries[a[0]] = a[1]

for year in sorted(years):
    g = ig.Graph()

    file_name = sys.argv[1] + year + '/' + year + '.gz'

    nodes_list, edges_list, weights_list = list(), list(), list()

    with gzip.open(file_name, 'r') as es_list:
        for edge in tqdm(es_list,
                         desc='YEAR {}: READING EDGES LIST'.format(year)):
            e = edge.decode().strip().split(',')

            nodes_list.append(e[0])
            nodes_list.append(e[1])
            edges_list.append((e[0], e[1]))
            weights_list.append(int(e[2]))

    nodes_list = list(set(nodes_list))

    for node in tqdm(nodes_list, desc='YEAR {}: ADDING NODES'.format(year)):
        affiliation = list()

        for affiliation_id in authors_affiliations[node]:
            years_range = np\
                .arange(authors_affiliations[node][affiliation_id]['from'],
                        authors_affiliations[node][affiliation_id]['to'] + 1)

            if int(year) in years_range and \
               affiliation_id in affiliations_countries:
                affiliation.append(affiliations_countries[affiliation_id])

        if len(affiliation) != 0:
            affiliation = Counter(affiliation)

            g.add_vertex(node, affiliation=affiliation.most_common(1)[0][0])

            authors_affiliations[node]['valid'] = True

    valid_edges, valid_weights = list(), list()

    for idx, val in tqdm(enumerate(edges_list),
                         desc='YEAR {}: VALIDATING EDGES'.format(year)):
        if authors_affiliations[val[0]]['valid'] and \
           authors_affiliations[val[1]]['valid']:
            valid_edges.append(edges_list[idx])
            valid_weights.append(weights_list[idx])

    g.add_edges(valid_edges)
    g.es['weight'] = valid_weights

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

    betweennesses.append(g.betweenness(directed=False))

    closenesses.append(g.closeness())

    for node in nodes_list:
        authors_affiliations[node]['valid'] = False

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

fig.suptitle("Node's Betweenness Distribution", fontsize=20)

axs.boxplot(betweennesses, labels=sorted(years), zorder=2)
axs.set_xlabel('Year', fontsize=14)
axs.grid(axis='y', linestyle='--', color='black', zorder=1)

fig.tight_layout(rect=[0, 0.03, 1, 0.90])
fig.savefig(sys.argv[1] + 'betweenness_distribution.pdf', format='pdf')
plt.close(fig=fig)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

fig.suptitle("Node's Closeness Distribution", fontsize=20)

axs.boxplot(closenesses, labels=sorted(years), zorder=2)
axs.set_xlabel('Year', fontsize=14)
axs.grid(axis='y', linestyle='--', color='black', zorder=1)

fig.tight_layout(rect=[0, 0.03, 1, 0.90])
fig.savefig(sys.argv[1] + 'closeness_distribution.pdf', format='pdf')
plt.close(fig=fig)

to_save = sys.argv[1] + 'networks_statistics.csv'

df = pd.DataFrame({'Year': sorted(years), 'Nodes': nodes, 'Edges': edges,
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

fig.savefig(sys.argv[1] + 'nodes_and_edges_growth_by_year.pdf', format='pdf')
plt.close(fig=fig)

df.to_csv(to_save, index=False)
