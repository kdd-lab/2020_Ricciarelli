import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.title_fontsize'] = 'x-small'

df = pd.read_csv('../../../1980_1989_statistics.csv', index_col=0)
df.drop(columns=['Average CC', 'Transitivity', 'Diameter', 'Radius'],
        inplace=True)

for path in ['../../../1990_1999_statistics.csv',
             '../../../2000_2009_statistics.csv',
             '../../../2010_2019_statistics.csv']:
    df = df.append(pd.read_csv(path, index_col=0))

for idx, t in enumerate(['Number of Nodes', 'Number of Edges']):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title('{} per Year'.format(t))
    ax.plot(np.arange(1980, 2020), df['Nodes'].values if idx == 0 else
            df['Edges'], lw=2, color='steelblue')
    ax.set_xlim(1979, 2020)
    ax.set_xticks(np.arange(1980, 2020, 10))
    ax.set_xticks(np.arange(1980, 2020), minor=True)
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel('Year')
    ax.set_ylabel(t)

    fig.savefig('./images/{}_per_year.pdf'.format('_'.join(t.lower().split())),
                format='pdf')
    plt.close(fig)

fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.set_title("Networks' Density per Year")
ax.plot(np.arange(1980, 2020), df['Density'].values, lw=2, color='steelblue')
ax.set_xlim(1979, 2020)
ax.set_xticks(np.arange(1980, 2020, 10))
ax.set_xticks(np.arange(1980, 2020), minor=True)
ax.tick_params(axis='both', which='major')
ax.set_xlabel('Year')
ax.set_ylabel('Density')

fig.savefig('./images/density_per_year.pdf',
            format='pdf')
plt.close(fig)
