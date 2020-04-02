import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd

from collections import Counter

path_to_nets = '../datasets/coauthors_networks/network_weight/'

g, years = None, np.arange(2010, 2020)

nodes, edges, densities, avg_cc, transitivities, eccen, diameters, rads = \
    list(), list(), list(), list(), list(), list(), list(), list()

if not os.path.isdir('../datasets/weighted_coauthors_networks/{}'
                     .format(years[0])):
    os.mkdir('../datasets/weighted_coauthors_networks/{}'.format(years[0]))

for year in years:
    print('PROCESSING YEAR ' + str(year))

    with open(path_to_nets + str(year), 'r') as edge_list:
        g = nx.read_weighted_edgelist(edge_list, delimiter=',',
                                      create_using=nx.Graph, nodetype=str)

    nodes_number = g.number_of_nodes()
    edges_number = g.number_of_edges()
    density = nx.density(g)
    avg_clustering_coefficient = nx.average_clustering(g, weight='weight')
    transitivity = nx.transitivity(g)
    eccentricity, diameter, radius = None, None, None

    try:
        eccentricity = nx.eccentricity(g)
        diameter = nx.diameter(g, e=eccentricity)
        radius = nx.radius(g, e=eccentricity)
    except Exception as e:
        eccentricity, diameter, radius = np.Inf, np.Inf, np.Inf

    nodes.append(np.format_float_scientific(nodes_number, 2))
    edges.append(np.format_float_scientific(edges_number, 2))
    densities.append(np.format_float_scientific(density, 2))
    avg_cc.append(np.format_float_scientific(avg_clustering_coefficient, 2))
    transitivities.append(np.format_float_scientific(transitivity, 2))
    eccen.append(eccentricity)
    diameters.append(diameter)
    rads.append(radius)

    degrees = sorted([d for n, d in g.degree(weight='weight')])

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
        countdict_pdf[deg] = (c[deg] / len(degrees)) if deg in c.keys() else 0.

    axs[1].scatter(list(countdict_pdf.keys()), list(countdict_pdf.values()),
                   zorder=2, alpha=0.7, color='#3296dc', edgecolor='#1f77b4')
    axs[1].set_title('Probability Density Distribution', fontsize=14)
    axs[1].set_xscale('log')
    axs[1].set_xlim(min(countdict_pdf.keys()), max(countdict_pdf.keys()) + 50)
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-7, 1.)
    axs[1].set_xlabel(r'$k$', fontsize=14)
    axs[1].set_ylabel(r'$p_k$', fontsize=14)
    axs[1].grid(axis='y', linestyle='--', color='black', zorder=1)

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig('../datasets/weighted_coauthors_networks/{}/'
                'degree_distribution_probability_distribution_{}.pdf'
                .format(years[0], year), format='pdf')
    plt.close(fig=fig)

    weights = sorted([edge[-1]['weight'] for edge in g.edges(data=True)])

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

    axs[1].scatter(list(countdict_pdf.keys()), list(countdict_pdf.values()),
                   zorder=2, alpha=0.7, color='#3296dc', edgecolor='#1f77b4')
    axs[1].set_title('Probability Density Distribution', fontsize=14)
    axs[1].set_xscale('log')
    axs[1].set_xlim(min(countdict_pdf.keys()), max(countdict_pdf.keys()) + 50)
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-7, 1.)
    axs[1].set_xlabel(r'$w$', fontsize=14)
    axs[1].set_ylabel(r'$p_w$', fontsize=14)
    axs[1].grid(axis='y', linestyle='--', color='black', zorder=1)

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig('../datasets/weighted_coauthors_networks/{}/weight_'
                'distribution_probability_distribution_{}.pdf'
                .format(years[0], year), format='pdf')
    plt.close(fig=fig)

    degree_centrality = sorted([v for k, v in nx.degree_centrality(g).items()])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    fig.suptitle('Degree Centrality Distribution, Year ' + str(year),
                 fontsize=20)

    axs.hist(degree_centrality, log=True, zorder=2, color='#3296dc',
             edgecolor='#1f77b4')
    axs.set_xlabel(r'$C_D$', fontsize=14)
    axs.set_ylabel(r'$N_{C_D}$', fontsize=14)
    axs.grid(axis='y', linestyle='--', color='black', zorder=1)
    axs.text(0.65, 0.75, 'mean = {}\nmax = {}\nmin = {}'
             .format(np.format_float_scientific(np.mean(degree_centrality), 2),
                     np.format_float_scientific(max(degree_centrality), 2),
                     np.format_float_scientific(min(degree_centrality), 2)),
             bbox=dict(facecolor='white', edgecolor='black'),
             transform=axs.transAxes)

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig('../datasets/weighted_coauthors_networks/{}/degree_centrality'
                '_distribution_{}.pdf'.format(years[0], year), format='pdf')
    plt.close(fig=fig)

    closeness_centrality = sorted(
        [v for k, v in nx.closeness_centrality(g).items()])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    fig.suptitle('Closeness Centrality Distribution, Year ' + str(year),
                 fontsize=20)

    axs.hist(closeness_centrality, log=True, zorder=2, color='#3296dc',
             edgecolor='#1f77b4')
    axs.set_xlabel(r'$C$', fontsize=14)
    axs.set_ylabel(r'$N_{C}$', fontsize=14)
    axs.grid(axis='y', linestyle='--', color='black', zorder=1)
    axs.text(0.65, 0.75, 'mean = {}\nmax = {}\nmin = {}'
             .format(np.format_float_scientific(np.mean(closeness_centrality),
                                                2),
                     np.format_float_scientific(max(closeness_centrality), 2),
                     np.format_float_scientific(min(closeness_centrality), 2)),
             bbox=dict(facecolor='white', edgecolor='black'),
             transform=axs.transAxes)

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig('../datasets/weighted_coauthors_networks/{}/'
                'closeness_centrality_distribution_{}.pdf'
                .format(years[0], year), format='pdf')
    plt.close(fig=fig)

    betweenness_centrality = sorted(
        [v for k, v in nx.betweenness_centrality(g, k=10, weight='weight').
            items()])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    fig.suptitle('Betweenness Centrality Distribution, Year ' + str(year),
                 fontsize=20)

    axs.hist(betweenness_centrality, log=True, zorder=2, color='#3296dc',
             edgecolor='#1f77b4')
    axs.set_xlabel(r'$B$', fontsize=14)
    axs.set_ylabel(r'$N_{B}$', fontsize=14)
    axs.grid(axis='y', linestyle='--', color='black', zorder=1)
    axs.text(0.65, 0.75, 'mean = {}\nmax = {}\nmin = {}'
             .format(np.format_float_scientific(np.mean(betweenness_centrality)
                                                , 2),
                     np.format_float_scientific(max(betweenness_centrality), 2)
                     , np.format_float_scientific(min(betweenness_centrality),
                                                  2)),
             bbox=dict(facecolor='white', edgecolor='black'),
             transform=axs.transAxes)

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig('../datasets/weighted_coauthors_networks/{}/'
                'betweenness_centrality_distribution_{}.pdf'
                .format(years[0], year), format='pdf')
    plt.close(fig=fig)

statistics = pd.DataFrame({'Year': years, 'Nodes': nodes, 'Edges': edges,
                           'Density': densities,
                           'Average Clustering Coefficient': avg_cc,
                           'Transitivity': transitivities,
                           'Eccentricity': eccen, 'Diameter': diameters,
                           'Radius': rads})
statistics.set_index('Year', inplace=True)
statistics.to_csv('../datasets/weighted_coauthors_networks/{}/statistics_'
                  '{}_{}.csv'.format(years[0], years[0], years[-1]))
