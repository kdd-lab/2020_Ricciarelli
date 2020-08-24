import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['legend.title_fontsize'] = 'x-small'

df = pd.read_csv('../../../1980_1989_statistics.csv', index_col=0)
df.drop(columns=['Average CC', 'Transitivity', 'Diameter', 'Radius'],
        inplace=True)

for path in ['../../../1990_1999_statistics.csv',
             '../../../2000_2009_statistics.csv',
             '../../../2010_2019_statistics.csv']:
    df = df.append(pd.read_csv(path, index_col=0))

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs = axs.reshape((1, -1))[0]

for idx, ax in enumerate(axs):
    ax.set_title('Number of Nodes per Year' if idx == 0
                 else 'Number of Edges per Year', fontsize=10)
    ax.plot(np.arange(1980, 2020), df['Nodes'].values if idx == 0 else
            df['Edges'], lw=2, color='steelblue')
    ax.set_xlim(1979, 2020)
    ax.set_xticks(np.arange(1980, 2020, 10))
    ax.set_xticks(np.arange(1980, 2020), minor=True)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_ylabel('Number of Nodes' if idx == 0 else 'Number of Edges',
                  fontsize=8)

fig.savefig('./images/number_of_nodes_and_edges_per_year.pdf',
            format='pdf')
plt.close(fig)

fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.set_title("Networks' Density per Year", fontsize=10)
ax.plot(np.arange(1980, 2020), df['Density'].values, lw=2, color='steelblue')
ax.set_xlim(1979, 2020)
ax.set_xticks(np.arange(1980, 2020, 10))
ax.set_xticks(np.arange(1980, 2020), minor=True)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Year', fontsize=8)
ax.set_ylabel('Density', fontsize=8)

fig.savefig('./images/density_per_year.pdf',
            format='pdf')
plt.close(fig)
