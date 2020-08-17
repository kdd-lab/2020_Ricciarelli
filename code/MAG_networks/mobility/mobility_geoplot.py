import json
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import sys

from tqdm import tqdm

eu_countries = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria',
                'Lithuania', 'Croatia', 'Luxembourg', 'N. Cyprus',
                'Malta', 'Czechia', 'Netherlands', 'Denmark', 'Poland',
                'Estonia', 'Portugal', 'Finland', 'Romania', 'France',
                'Slovakia', 'Germany', 'Slovenia', 'Greece', 'Spain',
                'Hungary', 'Sweden', 'Ireland', 'Armenia',
                'Azerbaijan', 'Belarus', 'Georgia', 'Iceland',
                'Liechtenstein', 'Monaco', 'Norway',
                'Russia', 'Switzerland', 'Ukraine',
                'United Kingdom']

mobility_dict = dict()

with open(sys.argv[1], 'r') as mobility_jsonl:
    for row in tqdm(mobility_jsonl, desc='READING MOBILITY JSONL'):
        country = json.loads(row)

        mobility_dict.update(country)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[world.name != "Antarctica"]

for key_type in ['in', 'out', 'balance']:
    mobility_key = dict()

    for country in mobility_dict:
        mobility_key[country] = dict()

        for year in mobility_dict[country]:
            mobility_key[country][year] = \
                {key_type: mobility_dict[country][year][key_type]}

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax = ax.reshape((1, -1))[0]

    if key_type != 'balance':
        fig.suptitle('{} Mobility Score per Country by Decade'
                     .format(key_type.capitalize()), fontsize=10)
    else:
        fig.suptitle("Mobility Score's Balance per Country by Decade"
                     .format(key_type.capitalize()), fontsize=10)

    for idx, decade in enumerate([1980, 1990, 2000, 2010]):
        mobility_score_dict = dict()

        for country in mobility_key:
            mobility_score = list()

            for year in np.arange(decade, decade + 10):
                if mobility_key[country][str(year)][key_type] == 0.0:
                    mobility_score.append(np.nan)
                else:
                    mobility_score.append(
                        mobility_key[country][str(year)][key_type])

            if len(mobility_score) != 0:
                mobility_score_dict[country] = np.nanmean(mobility_score)
            else:
                mobility_score_dict[country] = np.nan

        world['mobility'] = world['name'].map(mobility_score_dict)

        world.plot(column='mobility', ax=ax[idx], cmap='Oranges',
                   vmin=0.0, vmax=1.0, missing_kwds={'color': 'white'},
                   edgecolor='black', linewidth=0.1)
        ax[idx].set_title("From {} to {}".format(decade, decade + 9),
                          fontsize=8)
        ax[idx].axes.xaxis.set_visible(False)
        ax[idx].axes.yaxis.set_visible(False)

    vmin, vmax = 0.0, 1.0
    if key_type == 'balance':
        vmin = min(world['mobility'].values)
        vmax = max(world['mobility'].values)

    fig.colorbar(plt.cm.ScalarMappable(cmap='Oranges',
                                       norm=plt.Normalize(vmin=vmin,
                                                          vmax=vmax)),
                 ax=ax[-2:], shrink=0.5,
                 label='Mobility Score' if key_type != 'balance'
                 else 'Balance', location='bottom')
    fig.savefig('../images/mobility/mobility_{}_by_decade.pdf'
                .format(key_type), format='pdf', bbox_inches='tight')

    plt.close(fig)

    world.drop(['mobility'], axis=1, inplace=True)

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax = ax.reshape((1, -1))[0]

    if key_type != 'balance':
        fig.suptitle('{} Mobility Score per Country by Decade'
                     .format(key_type.capitalize()), fontsize=10)
    else:
        fig.suptitle("Mobility Score's Balance per Country by Decade"
                     .format(key_type.capitalize()), fontsize=10)

    for idx, decade in enumerate([1980, 1990, 2000, 2010]):
        mobility_score_dict = dict()

        for country in eu_countries:
            mobility_score = list()

            for year in np.arange(decade, decade + 10):
                if mobility_key[country][str(year)][key_type] == 0.0:
                    mobility_score.append(np.nan)
                else:
                    mobility_score.append(
                        mobility_key[country][str(year)][key_type])

            if len(mobility_score) != 0:
                mobility_score_dict[country] = np.nanmean(mobility_score)
            else:
                mobility_score_dict[country] = np.nan

        world['mobility'] = world['name'].map(mobility_score_dict)

        world[world.continent == 'Europe'].plot(
            column='mobility', ax=ax[idx], cmap='Oranges',
            vmin=0.0, vmax=1.0, missing_kwds={'color': 'white'},
            edgecolor='black', linewidth=0.1)
        ax[idx].set_title("From {} to {}".format(decade, decade + 9),
                          fontsize=8)
        ax[idx].axes.xaxis.set_visible(False)
        ax[idx].axes.yaxis.set_visible(False)
        ax[idx].set_xlim(-35, 125)
        ax[idx].set_ylim(25, 90)

    vmin, vmax = 0.0, 1.0
    if key_type == 'balance':
        vmin = min(world['mobility'].values)
        vmax = max(world['mobility'].values)

    fig.colorbar(plt.cm.ScalarMappable(cmap='Oranges',
                                       norm=plt.Normalize(vmin=vmin,
                                                          vmax=vmax)),
                 ax=ax[-2:], shrink=0.5, label='Mobility Score'
                 if key_type != 'balance' else 'Balance', location='bottom')
    fig.savefig('../images/mobility/mobility_{}_by_decade_eu.pdf'
                .format(key_type), format='pdf', bbox_inches='tight')

    plt.close(fig)

    world.drop(['mobility'], axis=1, inplace=True)
