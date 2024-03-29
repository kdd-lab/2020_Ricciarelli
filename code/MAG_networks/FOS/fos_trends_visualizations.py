import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from collections import Counter, defaultdict
from tqdm import tqdm

plt.rcParams['legend.title_fontsize'] = 'x-small'

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

for mag_id in cl_df[cl_df.cluster.isin([1, 2])]['MAG_id']:
    for year in entropies_dict[mag_id]:
        country = entropies_dict[mag_id][year]['affiliation']

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

        entropies_dict[mag_id][year]['affiliation'] = country

fos_counter_per_year = defaultdict(Counter)

for mag_id in tqdm(fos_dict, desc='COUNTING FOS PER YEAR'):
    for year in fos_dict[mag_id]:
        for field_of_study in fos_dict[mag_id][year]:
            fos_counter_per_year[year][field_of_study] += 1

markers = list(mpl.markers.MarkerStyle.markers.keys())

# MOST REPRESENTED FIELD OF STUDY PER YEAR ####################################

fields_of_study = set([fos_counter_per_year[str(x)].most_common()[0][0]
                      for x in np.arange(1980, 2020)])
field_of_study_markers, legend_entries = dict(), dict()

for idx, fos in enumerate(fields_of_study):
    field_of_study_markers[fos] = markers[idx]

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.set_title('Most represented Field of Study per Year', fontsize=10)
ax.plot(np.arange(1980, 2020),
        [fos_counter_per_year[str(year)].most_common()[0][1]
        for year in np.arange(1980, 2020)], lw=2, color='steelblue', alpha=0.5)

for idx, x in enumerate(np.arange(1980, 2020)):
    y = ax.get_children()[idx].properties()['data'][1][idx]
    field_of_study = fos_counter_per_year[str(x)].most_common()[0][0]

    legend_entries[field_of_study] = \
        ax.scatter(x, y, c='steelblue',
                   marker=field_of_study_markers[field_of_study])

vertical_lines = list()

for year, color, style in zip([1986, 1989, 1991, 2001, 2008],
                              ['#ff6347', '#47ff63', '#6347ff', '#ff4787',
                              '#47ffbf'], ['solid', 'dotted', 'dashed',
                              'dashdot', (0, (3, 5, 1, 5, 1, 5))]):
    vertical_lines.append(ax.axvline(x=year, ymin=-1.0, ymax=1.0, color=color,
                                     alpha=0.7, ls=style))

ax.set_xlim(1979, 2020)
ax.set_xticks(np.arange(1980, 2020, 10))
ax.set_xticks(np.arange(1980, 2020), minor=True)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Year', fontsize=8)
ax.set_ylabel('Registered Entries', fontsize=8)
plt.gca().add_artist(plt.legend([legend_entries[fos] for fos in
                                sorted(legend_entries)],
                                sorted(list(legend_entries.keys())),
                                loc='center left', fontsize=6,
                     title='Fields of Study', bbox_to_anchor=(1, 0.5)))
plt.legend(vertical_lines,
           ('Chernobyl disaster', 'Fall of the Berlin Wall',
            'Dissolution of the Soviet Union', '09/11',
            '2008 Economic Crisis'), loc='center left', fontsize=6,
           title='Events', bbox_to_anchor=(1, 0.25))
fig.savefig('../images/fos/most_represented_fos_per_year.pdf',
            format='pdf')
plt.close(fig)

# MOST REPRESENTED FIELD OF STUDY PER CLUSTER PER YEAR ########################

vertical_lines = list()

field_of_study_markers, legend_entries, fos_counter = dict(), dict(), 0

fig, ax = plt.subplots(1, 2, constrained_layout=True)
ax = ax.reshape((1, -1))[0]
fig.suptitle('Most represented Fields of Study per Cluster', fontsize=10)

for idx, cluster in enumerate([1, 2]):
    fos_counter_per_cluster = defaultdict(Counter)

    for mag_id in cl_df[cl_df.cluster == cluster]['MAG_id']:
        if mag_id in fos_dict:
            for year in fos_dict[mag_id]:
                for field_of_study in fos_dict[mag_id][year]:
                    fos_counter_per_cluster[year][field_of_study] += 1

    fields_of_study = set([fos_counter_per_cluster[str(x)].most_common()[0][0]
                          for x in np.arange(1980, 2020)])

    for fos in fields_of_study:
        if fos not in field_of_study_markers:
            field_of_study_markers[fos] = markers[fos_counter]
            fos_counter += 1

    ax[idx].plot(np.arange(1980, 2020),
                 [fos_counter_per_cluster[str(year)].most_common()[0][1]
                 for year in np.arange(1980, 2020)], lw=2, color='steelblue',
                 alpha=0.5)

    for i, x in enumerate(np.arange(1980, 2020)):
        y = ax[idx].get_children()[i].properties()['data'][1][i]
        field_of_study = fos_counter_per_cluster[str(x)].most_common()[0][0]

        legend_entries[field_of_study] = \
            ax[idx].scatter(x, y, c='steelblue',
                            marker=field_of_study_markers[field_of_study], s=8)

    for year, color, style in zip([1986, 1989, 1991, 2001, 2008],
                                  ['#ff6347', '#47ff63', '#6347ff', '#ff4787',
                                   '#47ffbf'], ['solid', 'dotted', 'dashed',
                                  'dashdot', (0, (3, 5, 1, 5, 1, 5))]):
        vertical_lines.append(ax[idx].axvline(x=year, ymin=-1.0, ymax=1.0,
                                              color=color, alpha=0.7,
                                              ls=style))

    ax[idx].set_title('Cluster {}'.format(cluster), fontsize=8)
    ax[idx].set_xlim(1979, 2020)
    ax[idx].set_xticks(np.arange(1980, 2020, 10))
    ax[idx].set_xticks(np.arange(1980, 2020), minor=True)
    ax[idx].tick_params(axis='both', which='major', labelsize=6)
    ax[idx].set_xlabel('Year', fontsize=8)
    ax[idx].set_ylabel('Registered Entries', fontsize=8)

fig.legend([legend_entries[fos] for fos in sorted(legend_entries)],
           sorted(list(legend_entries.keys())), loc='center left', fontsize=6,
           title='Fields of Study', bbox_to_anchor=(1.0, 0.5),
           bbox_transform=ax[-1].transAxes)
fig.legend(vertical_lines[:5],
           ('Chernobyl disaster', 'Fall of the Berlin Wall',
            'Dissolution of the Soviet Union', '09/11',
            '2008 Economic Crisis'), loc='center left', fontsize=6,
           title='Events', bbox_to_anchor=(1, 0.25),
           bbox_transform=ax[-1].transAxes)

fig.savefig('../images/fos/most_represented_fos_per_cluster.pdf',
            bbox_inches='tight', format='pdf')
plt.close(fig)

# MOST REPRESENTED FIELD OF STUDY PER SPECIFIC COUNTRIES ######################

fos_per_country = dict()

for mag_id in cl_df[cl_df.cluster.isin([1, 2])]['MAG_id']:
    if mag_id in fos_dict:
        for year in fos_dict[mag_id]:
            if year in entropies_dict[mag_id]:
                country = entropies_dict[mag_id][year]['affiliation']

                if country not in fos_per_country:
                    fos_per_country[country] = defaultdict(Counter)

                for fos in fos_dict[mag_id][year]:
                    fos_per_country[country][year][fos] += 1

field_of_study_markers, legend_entries = dict(), dict()
fields_of_study = set()

for country in ['Italy', 'Germany', 'France', 'Spain', 'United Kingdom',
                'United States of America', 'Russia', 'China']:
    for year in fos_per_country[country]:
        for fos in fos_per_country[country][year]:
            fields_of_study.add(fos)

for idx, fos in enumerate(fields_of_study):
    field_of_study_markers[fos] = markers[idx]

fig, ax = plt.subplots(4, 2, constrained_layout=True)
ax = ax.reshape((1, -1))[0]

fig.suptitle('Most represented Fields of Study for various Countries',
             fontsize=10)

for idx, country in enumerate(['Italy', 'Germany', 'Norway', 'Poland',
                              'Portugal', 'United States of America', 'Russia',
                               'China']):
    ax[idx].plot(np.arange(1980, 2020),
                 [fos_per_country[country][str(year)].most_common()[0][1]
                 for year in np.arange(1980, 2020)], lw=2, color='steelblue',
                 alpha=0.5)

    for i, x in enumerate(np.arange(1980, 2020)):
        y = ax[idx].get_children()[i].properties()['data'][1][i]
        field_of_study = fos_per_country[country][str(x)].most_common()[0][0]

        legend_entries[field_of_study] = \
            ax[idx].scatter(x, y, c='steelblue',
                            marker=field_of_study_markers[field_of_study], s=8)

    vertical_lines = list()

    for year, color, style in zip([1986, 1989, 1991, 2001, 2008],
                                  ['#ff6347', '#47ff63', '#6347ff', '#ff4787',
                                  '#47ffbf'], ['solid', 'dotted', 'dashed',
                                  'dashdot', (0, (3, 5, 1, 5, 1, 5))]):
        vertical_lines.append(ax[idx].axvline(x=year, ymin=-1.0, ymax=1.0,
                                              color=color, alpha=0.7,
                                              ls=style))

    ax[idx].set_title(country, fontsize=8)
    ax[idx].set_xlim(1979, 2020)
    ax[idx].set_xticks(np.arange(1980, 2020, 10))
    ax[idx].set_xticks(np.arange(1980, 2020), minor=True)
    ax[idx].tick_params(axis='both', which='major', labelsize=6)
    ax[idx].set_xlabel('Year', fontsize=8)
    ax[idx].set_ylabel('Registered Entries', fontsize=8)

fig.legend([legend_entries[fos] for fos in sorted(legend_entries)],
           sorted(list(legend_entries.keys())), loc='center left', fontsize=6,
           title='Fields of Study', bbox_to_anchor=(1.1, 0.5),
           bbox_transform=ax[1].transAxes)
fig.legend(vertical_lines[:5],
           ('Chernobyl disaster', 'Fall of the Berlin Wall',
            'Dissolution of the Soviet Union', '09/11',
            '2008 Economic Crisis'), loc='center left', fontsize=6,
           title='Events', bbox_to_anchor=(1.1, 0.25),
           bbox_transform=ax[3].transAxes)
fig.savefig('../images/fos/fos_of_various_countries.pdf', bbox_inches='tight',
            format='pdf')
plt.close(fig)
