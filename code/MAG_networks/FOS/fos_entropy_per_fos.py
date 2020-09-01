import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from tqdm import tqdm


def check_country(country):
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

    return country


fos_dict = dict()

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        fos_dict.update(creator)

entropies_dict = dict()

with open(sys.argv[2], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        entropies_dict.update(creator)

cl_df = pd.read_csv(sys.argv[3], dtype={'MAG_id': str, 'cluster': int})

entropy_per_fos_per_year = dict()

for mag_id in cl_df[cl_df.cluster.isin([1, 2])]['MAG_id']:
    if mag_id in fos_dict:
        fos_years = set([year for year in fos_dict[mag_id]])
        entropies_years = set([year for year in entropies_dict[mag_id]])

        for year in fos_years.intersection(entropies_years):
            score = entropies_dict[mag_id][year]['entropy']

            for fos in fos_dict[mag_id][year]:
                if fos not in entropy_per_fos_per_year:
                    entropy_per_fos_per_year[fos] = defaultdict(list)

                entropy_per_fos_per_year[fos][year].append(score)

for fos in entropy_per_fos_per_year:
    for year in np.arange(1980, 2020):
        mean, std = np.nan, np.nan

        if str(year) in entropy_per_fos_per_year[fos]:
            mean = np.mean(entropy_per_fos_per_year[fos][str(year)])
            std = np.std(entropy_per_fos_per_year[fos][str(year)])

        entropy_per_fos_per_year[fos][str(year)] = {'mean': mean, 'std': std}

for i, fos_list in enumerate([sorted(entropy_per_fos_per_year)[:6],
                             sorted(entropy_per_fos_per_year)[6:]]):
    fig, ax = plt.subplots(3 if i == 0 else 4, 2, constrained_layout=True)
    ax = ax.reshape((1, -1))[0]

    fig.suptitle('Xenofilia/Xenofobia per Field of Study over the Years',
                 fontsize=10)

    for idx, fos in enumerate(fos_list):
        title = fos.capitalize() if '_' not in fos else \
            ' '.join([chunk.capitalize() for chunk in fos.split('_')])

        ax[idx].set_title(title, fontsize=8)
        ax[idx].plot(np.arange(1980, 2020),
                     [entropy_per_fos_per_year[fos][y]['mean'] for y in
                     sorted(entropy_per_fos_per_year[fos])], lw=2,
                     color='steelblue')
        ax[idx].fill_between(np.arange(1980, 2020),
                             [entropy_per_fos_per_year[fos][y]['mean'] -
                             entropy_per_fos_per_year[fos][y]['std'] for y in
                             sorted(entropy_per_fos_per_year[fos])],
                             [entropy_per_fos_per_year[fos][y]['mean'] +
                             entropy_per_fos_per_year[fos][y]['std'] for y in
                             sorted(entropy_per_fos_per_year[fos])],
                             color='steelblue', alpha=0.1)
        ax[idx].set_xlim(1979, 2018)
        ax[idx].set_ylim(-0.5, 0.5)
        ax[idx].set_xticks(np.arange(1980, 2018, 10))
        ax[idx].set_xticks(np.arange(1980, 2018), minor=True)
        ax[idx].tick_params(axis='both', which='major', labelsize=6)
        ax[idx].set_xlabel('Year', fontsize=6)
        ax[idx].set_ylabel('Xenofilia/Xenofobia', fontsize=6)

    if i == 1:
        ax[-1].remove()

    fig.savefig('../images/fos/xenofilia_xenofobia_per_fos_per_year_{}.pdf'
                .format(i + 1), format='pdf')
    plt.close(fig)

countries = ['Italy', 'Germany', 'Norway', 'Poland', 'Portugal',
             'United States of America', 'Russia', 'China']

entropy_per_fos_per_country = dict()

for mag_id in cl_df[cl_df.cluster.isin([1, 2])]['MAG_id']:
    if mag_id in fos_dict:
        fos_years = set([year for year in fos_dict[mag_id]])
        entropies_years = set([year for year in entropies_dict[mag_id]])

        for year in fos_years.intersection(entropies_years):
            score = entropies_dict[mag_id][year]['entropy']
            country = check_country(entropies_dict[mag_id][year]['affiliation'])

            if country in countries:
                for fos in fos_dict[mag_id][year]:
                    if fos not in entropy_per_fos_per_country:
                        entropy_per_fos_per_country[fos] = \
                            {country: defaultdict(list)}
                    else:
                        if country not in entropy_per_fos_per_country[fos]:
                            entropy_per_fos_per_country[fos][country] = \
                                defaultdict(list)

                    entropy_per_fos_per_country[fos][country][year]\
                        .append(score)

for fos in entropy_per_fos_per_country:
    for year in np.arange(1980, 2020):
        mean, std = 0.0, 0.0

        for country in entropy_per_fos_per_country[fos]:
            if str(year) in entropy_per_fos_per_country[fos][country]:
                mean = \
                    np.mean(entropy_per_fos_per_country[fos][country]
                            [str(year)])
                std = \
                    np.std(entropy_per_fos_per_country[fos][country]
                           [str(year)])

            entropy_per_fos_per_country[fos][country][str(year)] = \
                {'mean': mean, 'std': std}

for fos in entropy_per_fos_per_country:
    fig, ax = plt.subplots(4, 2, constrained_layout=True)
    ax = ax.reshape((1, -1))[0]

    title = fos.capitalize() if '_' not in fos else ' '\
        .join([chunk.capitalize() for chunk in fos.split('_')])

    fig.suptitle('Xenofilia/Xenofobia per Field of Study over the Years'
                 ' by Country - {}'.format(title), fontsize=10)

    for idx, country in enumerate(sorted(entropy_per_fos_per_country[fos])):
        ax[idx].set_title(country, fontsize=8)
        ax[idx].plot(np.arange(1980, 2020),
                     [entropy_per_fos_per_country[fos][country][y]['mean']
                     for y in
                     sorted(entropy_per_fos_per_country[fos][country])], lw=2,
                     color='steelblue')

        for year, color, style in zip([1986, 1989, 1991, 2001, 2008],
                                      ['#ff6347', '#47ff63', '#6347ff',
                                       '#ff4787', '#47ffbf'],
                                      ['solid', 'dotted', 'dashed', 'dashdot',
                                      (0, (3, 5, 1, 5, 1, 5))]):
            ax[idx].axvline(x=year, ymin=-1.0, ymax=1.0, color=color,
                            alpha=0.7, ls=style)

        ax[idx].set_xlim(1979, 2018)
        ax[idx].set_xticks(np.arange(1980, 2018, 10))
        ax[idx].set_xticks(np.arange(1980, 2018), minor=True)
        ax[idx].tick_params(axis='both', which='major', labelsize=6)
        ax[idx].set_xlabel('Year', fontsize=6)
        ax[idx].set_ylabel('Xenofilia/Xenofobia', fontsize=6)

    fig.legend(ax[0].get_children()[1:6],
               ('Chernobyl disaster', 'Fall of the Berlin Wall',
                'Dissolution of the Soviet Union', '09/11',
                '2008 Economic Crysis'), loc='center left', fontsize=8,
               title='Events', bbox_to_anchor=(1.0, 0.5),
               bbox_transform=ax[1].transAxes)
    fig.savefig('../images/fos/xenofilia_xenofobia_per_fos_per_country_{}.pdf'
                .format(fos), format='pdf')
    plt.close(fig)
