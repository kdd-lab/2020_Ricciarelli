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
            entropies_per_country = defaultdict(list)
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

                    entropies_per_country[country].append(entropy)

            for country in entropies_per_country:
                entropies_per_country[country] = \
                    np.mean(entropies_per_country[country])

            world['entropy'] = world['name'].map(dict(entropies_per_country))

            fig, ax = plt.subplots()

            world.plot(column='entropy', ax=ax, legend=True, cmap='RdYlGn',
                       legend_kwds={'label': "Entropy by Country",
                                    'orientation': "horizontal"}, vmin=-1.0,
                       vmax=1.0, missing_kwds={'color': 'lightgrey'},
                       figsize=(10, 10))
            ax.set_title("Entropies' Distribution per Country - Cluster {}"
                         .format(cluster), fontsize=10)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            save_n = \
                './images/entropies_distribution_country_cluster_{}.pdf'\
                .format(cluster)

            fig.savefig(save_n, format='pdf', bbox_inches='tight')
