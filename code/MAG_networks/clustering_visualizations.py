import json
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from tqdm import tqdm

if sys.argv[1] == 'lineplot':
    f = open(sys.argv[2], 'r').readlines()

    data = defaultdict(list)

    for line in f:
        ln = [chunk.split(': ')[-1] for chunk in line.strip().split(', ')]

        data[int(ln[0])].append(float(ln[1]))

    for cluster in data:
        data[cluster] = {'mean': np.mean(data[cluster]),
                         'std': np.std(data[cluster])}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(np.arange(0, 9), [data[k]['mean'] for k in sorted(data)],
            linewidth=2, marker='o', markersize=5, color='steelblue')
    ax.fill_between(np.arange(0, 9),
                    [data[k]['mean'] - data[k]['std'] for k in sorted(data)],
                    [data[k]['mean'] + data[k]['std'] for k in sorted(data)],
                    color='steelblue', alpha=0.3)
    ax.grid(linestyle='--', color='black', alpha=0.4)
    ax.set_xticks(np.arange(0, 9))
    ax.set_xticklabels(['2', '3', '4', '5', '6', '7', '8', '9', '10'])
    ax.set_title('KMeans Grid Search on a Deeper Level'
                 if 'deeper' in sys.argv[2] else 'KMeans Grid Search',
                 fontsize=20)
    ax.set_xlabel('Cluster', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)

    fig.savefig('./images/kmeans_deeper_grid_search.pdf' if 'deeper'
                in sys.argv[2] else './images/kmeans_grid_search.pdf',
                format='pdf', bbox_inches='tight')

    plt.close(fig)
elif sys.argv[1] == 'yearplot':
    log_file = open('./logs/entropies_deeper_clustering_results.log', 'r')
    log_file = log_file.readlines()

    years_c1 = log_file[5].split(': ')[-1].strip().replace('[', '')\
        .replace(']', '').replace("'", '').split(', ')
    years_c2 = log_file[19].split(': ')[-1].strip().replace('[', '')\
        .replace(']', '').replace("'", '').split(', ')
    years_in_c1, years_in_c2 = list(), list()

    for year in np.arange(1980, 2020):
        if str(year) in years_c1:
            if str(year) in years_c2:
                years_in_c2.append(-1.)
            else:
                years_in_c2.append(0.)
            years_in_c1.append(1.)
        elif str(year) in years_c2:
            years_in_c1.append(0.)
            years_in_c2.append(-1.)
        else:
            years_in_c1.append(0.)
            years_in_c2.append(0.)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(np.arange(0, 40), years_in_c1, linewidth=2, color='#46B4AF',
            label='Cluster 1')
    ax.plot(np.arange(0, 40), years_in_c2, linewidth=2, color='#b4464b',
            label='Cluster 2')
    ax.set_xticks(np.arange(0, 40))
    ax.set_xticklabels([str(year) for year in np.arange(1980, 2020)],
                       rotation='vertical')
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_yticklabels(['Yes', 'No', 'Yes'])
    ax.set_title('Years represented in each Cluster', fontsize=20)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Is represented?', fontsize=14)
    ax.legend()

    fig.savefig('./images/clustering/years_per_cluster.pdf',
                format='pdf', bbox_inches='tight')
else:
    entropies_dict = dict()

    with open(sys.argv[2], 'r') as entropies_file:
        for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
            creator = json.loads(row)

            entropies_dict.update(creator)

    cl_df = pd.read_csv(sys.argv[3],
                        dtype={'MAG_id': str, 'cluster': int})

    if sys.argv[1] == 'boxplot':
        entropies_per_cluster = [[] for cluster in cl_df.cluster.unique()]

        for cluster in sorted(cl_df.cluster.unique()):
            for MAG_id in cl_df[cl_df.cluster == cluster]['MAG_id']:
                entropies = list()

                for year in entropies_dict[MAG_id]:
                    entropies.append(entropies_dict[MAG_id][year]['entropy'])

                entropies_per_cluster[cluster].append(np.mean(entropies))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.boxplot(entropies_per_cluster,
                   labels=[str(c) for c in sorted(cl_df.cluster.unique())],
                   showfliers=False, showmeans=True)
        ax.grid(linestyle='--', color='black', alpha=0.4)
        ax.set_title("Entropies' Distribution per Clusteron a Deeper Level"
                     if 'deeper' in sys.argv[3] else
                     "Entropies' Distribution per Cluster", fontsize=20)
        ax.set_xlabel('Cluster', fontsize=14)
        ax.set_ylabel('Silhouette Score', fontsize=14)

        fig.savefig('./images/entropies_distribution_deeper.pdf' if 'deeper'
                    in sys.argv[3] else './images/entropies_distribution.pdf',
                    format='pdf', bbox_inches='tight')
    elif sys.argv[1] == 'geoplot':
        for cluster in [1, 2]:
            entropies_per_country = dict()
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world = world[world.name != "Antarctica"]

            for MAG_id in cl_df[cl_df.cluster == cluster]['MAG_id']:
                for year in entropies_dict[MAG_id]:
                    country = entropies_dict[MAG_id][year]['affiliation']
                    entropy = entropies_dict[MAG_id][year]['entropy']

                    if country == 'United States':
                        country = 'United States of America'
                    elif country == 'Korea':
                        country = 'South Korea'
                    elif country == 'Russian Federation':
                        country = 'Russia'
                    elif country == 'Dominican Republic':
                        country = 'Dominican Rep.'

                    if country not in entropies_per_country:
                        entropies_per_country[country] = defaultdict(list)

                    entropies_per_country[country][year].append(entropy)

            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax_index = 0

            fig.suptitle('Mean Entropy per Country over the Decades '
                         '- Cluster {}'.format(cluster), fontsize=10)

            for decade in [1980, 1990, 2000, 2010]:
                entropies_by_decade = defaultdict(list)

                import ipdb
                ipdb.set_trace()

                for country in entropies_per_country:
                    for year in np.arange(decade, decade + 10):
                        entropies_by_decade[country] \
                            .append(np.mean(entropies_per_country[country]
                                            [str(year)]))

                for country in entropies_by_decade:
                    entropies_by_decade[country] = \
                        np.mean(entropies_by_decade[country])

                world['entropy'] = world['name']\
                    .map(dict(entropies_by_decade))

                world.plot(column='entropy', ax=ax[ax_index], legend=True,
                           cmap='RdYlGn',
                           legend_kwds={'label': "Entropy",
                                        'orientation': "horizontal",
                                        'shrink': 0.3},
                           vmin=-1.0, vmax=1.0,
                           missing_kwds={'color': 'lightgrey'},
                           edgecolor='black', linewidth=0.1)
                ax[ax_index].set_title("From {} to {}"
                                       .format(decade, decade + 9),
                                       fontsize=8)
                ax[ax_index].axes.xaxis.set_visible(False)
                ax[ax_index].axes.yaxis.set_visible(False)

                world.drop(['entropy'], axis=1, inplace=True)
                ax_index += 1

            save_n = \
                './images/clustering/mean_entropy_per_decade_cluster_{}.pdf'\
                .format(cluster)

            fig.savefig(save_n, format='pdf', bbox_inches='tight')

            plt.close(fig)

            for country in entropies_per_country:
                entropies = [entropies_per_country[country][year] for year
                             in entropies_per_country[country]]
                entropies_per_country[country] = np.mean(entropies)

            world['entropy'] = world['name'].map(dict(entropies_per_country))

            fig, ax = plt.subplots()
            world.plot(column='entropy', ax=ax, legend=True, cmap='RdYlGn',
                       legend_kwds={'label': "Entropy",
                                    'orientation': "horizontal",
                                    'shrink': 0.3}, vmin=-1.0,
                       vmax=1.0, missing_kwds={'color': 'lightgrey'},
                       edgecolor='black', linewidth=0.1, figsize=(10, 10))
            ax.set_title("Mean Entropy per Country - Cluster {}"
                         .format(cluster), fontsize=10)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            save_n = \
                './images/clustering/mean_entropy_per_country_cluster_{}.pdf'\
                .format(cluster)

            fig.savefig(save_n, format='pdf', bbox_inches='tight')
