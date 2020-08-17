import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm

mobility_dict = dict()

with open(sys.argv[1], 'r') as mobility_jsonl:
    for row in tqdm(mobility_jsonl, desc='READING MOBILITY JSONL'):
        country = json.loads(row)

        mobility_dict.update(country)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[world.name != "Antarctica"]

for key_type in ['in', 'out']:
    mobility_key = dict()

    for country in mobility_dict:
        mobility_key[country] = dict()

        for year in mobility_dict[country]:
            mobility_key[country][year] = \
                {key_type: mobility_dict[country][year][key_type]}

    world['mobility'] = world['name'].map(mobility_key)

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax = ax.reshape((1, -1))[0]

    fig.suptitle('Mobility Score per Country by Decade', fontsize=10)

    for idx, decade in enumerate([1980, 1990, 2000, 2010]):
        world.plot(column='mobility', ax=ax[idx], cmap='Greens',
                   vmin=0.0, vmax=1.0, missing_kwds={'color': 'white'},
                   edgecolor='black', linewidth=0.1)
        ax[idx].set_title("From {} to {}".format(decade, decade + 9),
                          fontsize=8)
        ax[idx].axes.xaxis.set_visible(False)
        ax[idx].axes.yaxis.set_visible(False)

    fig.colorbar(plt.cm.ScalarMappable(cmap='Greens',
                                       norm=plt.Normalize(vmin=0.0, vmax=1.0)),
                 ax=ax[-2:], shrink=0.5, label='Mobility Score',
                 location='bottom')
    fig.savefig('../images/mobility/mobility_{}_by_decade.pdf'
                .format(key_type), format='pdf', bbox_inches='tight')

    plt.close(fig)

    world.drop(['mobility'], axis=1, inplace=True)
