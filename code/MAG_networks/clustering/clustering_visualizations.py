import json
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings

from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8

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

    fig, ax = plt.subplots()
    ax.plot(np.arange(1980, 2020), years_in_c1, linewidth=2, color='#46B4AF',
            label='Cluster 1')
    ax.plot(np.arange(1980, 2020), years_in_c2, linewidth=2, color='#b4464b',
            label='Cluster 2')
    ax.set_xlim(1979, 2020)
    ax.set_xticks(np.arange(1980, 2020, 10))
    ax.set_xticks(np.arange(1980, 2020), minor=True)
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_yticklabels(['Yes', 'No', 'Yes'])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title('Years represented in each Cluster', fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Is represented?', fontsize=10)
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

        fig, ax = plt.subplots(constrained_layout=True)
        ax.boxplot(entropies_per_cluster,
                   labels=[str(c) for c in sorted(cl_df.cluster.unique())],
                   showfliers=False, showmeans=True)
        ax.grid(linestyle='--', color='black', alpha=0.4)
        ax.set_title("YDCI's Distribution per Cluster" if 'deeper'
                     in sys.argv[3] else "Entropies' Distribution per Cluster")
        ax.set_xlabel('Cluster')
        ax.set_ylabel('YDCI')

        fig.savefig('../images/clustering/entropies_distribution_deeper.pdf' if
                    'deeper' in sys.argv[3] else
                    '../images/clustering/entropies_distribution.pdf',
                    format='pdf')
    elif sys.argv[1] == 'geoplot':
        for id_list in [cl_df[cl_df.cluster == 1]['MAG_id'],
                        cl_df[cl_df.cluster == 2]['MAG_id']]:
            for MAG_id in id_list:
                for year in entropies_dict[MAG_id]:
                    country = entropies_dict[MAG_id][year]['affiliation']

                    if country == 'United States':
                        country = 'United States of America'
                    elif country == 'Korea':
                        country = 'South Korea'
                    elif country == 'Russian Federation':
                        country = 'Russia'
                    elif country == 'Dominican Republic':
                        country = 'Dominican Rep.'
                    elif country == 'Bosnia and Herzegovina':
                        country = 'Bosnia and Herz.'
                    elif country == "Lao People's Democratic Republic":
                        country = 'Laos'
                    elif country == 'Cyprus':
                        country = 'N. Cyprus'
                    elif country == 'Central African Republic':
                        country = 'Central African Rep.'
                    elif country == 'South Sudan':
                        country = 'S. Sudan'
                    elif country == 'Syrian Arab Republic':
                        country = 'Syria'
                    elif country == 'Viet Nam':
                        country = 'Vietnam'

                    entropies_dict[MAG_id][year]['affiliation'] = country

        eu_countries = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria',
                        'Lithuania', 'Croatia', 'Luxembourg', 'N. Cyprus',
                        'Malta', 'Czechia', 'Netherlands', 'Denmark', 'Poland',
                        'Estonia', 'Portugal', 'Finland', 'Romania', 'France',
                        'Slovakia', 'Germany', 'Slovenia', 'Greece', 'Spain',
                        'Hungary', 'Sweden', 'Ireland', 'Armenia', 'Andorra',
                        'Azerbaijan', 'Belarus', 'Georgia', 'Iceland',
                        'Liechtenstein', 'Moldova', 'Monaco', 'Norway',
                        'Russia', 'San Marino', 'Switzerland', 'Ukraine',
                        'United Kingdom', 'Vatican City']

        entropies_per_country = dict()
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = world[world.name != "Antarctica"]
        eu = world[world.continent == 'Europe']

        for MAG_id in cl_df[cl_df.cluster.isin([1, 2])]['MAG_id']:
            for year in entropies_dict[MAG_id]:
                country = entropies_dict[MAG_id][year]['affiliation']
                entropy = entropies_dict[MAG_id][year]['entropy']

                if entropy not in [0.0, -0.0]:
                    if country not in entropies_per_country:
                        entropies_per_country[country] = defaultdict(list)

                    entropies_per_country[country][year].append(entropy)

        fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        ax = ax.reshape((1, -1))[0]

        fig.suptitle('Changes in the YDCI over the Decades',
                     fontsize=10)

        for idx, decade in enumerate([1980, 1990, 2000, 2010]):
            entropies_by_decade = defaultdict(list)

            for country in entropies_per_country:
                for year in np.arange(decade, decade + 10):
                    for e in entropies_per_country[country][str(year)]:
                        entropies_by_decade[country].append(e)

            to_plot = dict()

            for country in entropies_by_decade:
                to_plot[country] = np.mean(entropies_by_decade[country])

            world['entropy'] = world['name'].map(to_plot)

            world.plot(column='entropy', ax=ax[idx], cmap='coolwarm',
                       vmin=-1.0, vmax=1.0, missing_kwds={'color': 'white'},
                       edgecolor='black', linewidth=0.1)
            ax[idx].set_title("From {} to {}".format(decade, decade + 9),
                              fontsize=8)
            ax[idx].axes.xaxis.set_visible(False)
            ax[idx].axes.yaxis.set_visible(False)

            world.drop(['entropy'], axis=1, inplace=True)

        save_n = '../images/clustering/changes_over_entropy_per_decade.pdf'

        fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm',
                                           norm=plt.Normalize(vmin=-1.0,
                                                              vmax=1.0)),
                     ax=ax[-2:], shrink=0.5, label='YDCI',
                     location='bottom')
        fig.savefig(save_n, format='pdf', bbox_inches='tight')

        plt.close(fig)

        fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        ax = ax.reshape((1, -1))[0]

        fig.suptitle('Changes in the YDCI over the Decades'
                     ' - Europe', fontsize=10)

        for idx, decade in enumerate([1980, 1990, 2000, 2010]):
            entropies_by_decade = defaultdict(list)

            for country in eu_countries:
                for year in np.arange(decade, decade + 10):
                    if country in entropies_per_country:
                        for e in entropies_per_country[country][str(year)]:
                            entropies_by_decade[country].append(e)

            to_plot = dict()

            for country in entropies_by_decade:
                to_plot[country] = np.mean(entropies_by_decade[country])

            eu['entropy'] = eu['name'].map(to_plot)

            eu.plot(column='entropy', ax=ax[idx], cmap='coolwarm', vmin=-1.0,
                    vmax=1.0, missing_kwds={'color': 'white'},
                    edgecolor='black', linewidth=0.1)
            ax[idx].set_title("From {} to {}".format(decade, decade + 9),
                              fontsize=8)
            ax[idx].axes.xaxis.set_visible(False)
            ax[idx].axes.yaxis.set_visible(False)
            ax[idx].set_xlim(-35, 125)
            ax[idx].set_ylim(25, 90)

            eu.drop(['entropy'], axis=1, inplace=True)

        save_n = '../images/clustering/changes_over_entropy_per_decade_eu.pdf'

        fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm',
                                           norm=plt.Normalize(vmin=-1.0,
                                                              vmax=1.0)),
                     ax=ax[:], shrink=0.5, label='YDCI',
                     location='bottom')
        fig.savefig(save_n, format='pdf', bbox_inches='tight')

        for cluster in [1, 2]:
            entropies_per_country = dict()
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world = world[world.name != "Antarctica"]

            for MAG_id in cl_df[cl_df.cluster == cluster]['MAG_id']:
                for year in entropies_dict[MAG_id]:
                    country = entropies_dict[MAG_id][year]['affiliation']
                    entropy = entropies_dict[MAG_id][year]['entropy']

                    if country not in entropies_per_country:
                        entropies_per_country[country] = defaultdict(list)

                    entropies_per_country[country][year].append(entropy)

            fig, ax = plt.subplots(nrows=2, ncols=2)

            fig.suptitle('Mean YDCI per Country over the '
                         'Decades - Cluster {}'.format(cluster), fontsize=10)

            for coord, decade in zip([[0, 0], [0, 1], [1, 0], [1, 1]],
                                     [1980, 1990, 2000, 2010]):
                entropies_by_decade = defaultdict(list)

                for country in entropies_per_country:
                    for year in np.arange(decade, decade + 10):
                        for e in entropies_per_country[country][str(year)]:
                            entropies_by_decade[country].append(e)

                for country in entropies_by_decade:
                    entropies_by_decade[country] = \
                        np.mean(entropies_by_decade[country])

                world['entropy'] = world['name']\
                    .map(dict(entropies_by_decade))

                world.plot(column='entropy', ax=ax[coord[0], coord[1]],
                           cmap='coolwarm', vmin=-1.0, vmax=1.0,
                           missing_kwds={'color': 'white'},
                           edgecolor='black', linewidth=0.1)
                ax[coord[0], coord[1]].set_title("From {} to {}"
                                                 .format(decade, decade + 9),
                                                 fontsize=8)
                ax[coord[0], coord[1]].axes.xaxis.set_visible(False)
                ax[coord[0], coord[1]].axes.yaxis.set_visible(False)

                world.drop(['entropy'], axis=1, inplace=True)

            save_n = \
                '../images/clustering/mean_entropy_per_decade_cluster_{}.pdf'\
                .format(cluster)

            fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm',
                                               norm=plt.Normalize(vmin=-1.0,
                                                                  vmax=1.0)),
                         ax=ax[1, :], shrink=0.5, label='YDCI',
                         location='bottom')
            fig.subplots_adjust(top=1.0)
            fig.savefig(save_n, format='pdf', bbox_inches='tight')

            plt.close(fig)

            for country in entropies_per_country:
                entropies = [entropies_per_country[country][year] for year
                             in entropies_per_country[country]]
                entropies_per_country[country] = \
                    np.mean(np.concatenate(entropies))

            world['entropy'] = world['name'].map(dict(entropies_per_country))

            fig, ax = plt.subplots(constrained_layout=True)
            world.plot(column='entropy', ax=ax, cmap='coolwarm', vmin=-1.0,
                       vmax=1.0, missing_kwds={'color': 'white'},
                       edgecolor='black', linewidth=0.1, figsize=(10, 10))
            ax.set_title("Mean YDCI per Country - Cluster {}"
                         .format(cluster), fontsize=10)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            save_n = \
                '../images/clustering/mean_entropy_per_country_cluster_{}.pdf'\
                .format(cluster)

            fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm',
                                               norm=plt.Normalize(vmin=-1.0,
                                                                  vmax=1.0)),
                         ax=ax, shrink=0.5, label='YDCI',
                         location='bottom')
            fig.savefig(save_n, format='pdf', bbox_inches='tight')
