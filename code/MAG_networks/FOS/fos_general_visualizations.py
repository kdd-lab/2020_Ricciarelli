import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import defaultdict
from tqdm import tqdm

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8

fos_counter, multidisciplinarity = defaultdict(int), defaultdict(int)

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='READING FOS FILE'):
        creator = json.loads(row)
        creator_fos = set()

        for mag_id in creator:
            for year in creator[mag_id]:
                for fos in creator[mag_id][year]:
                    creator_fos.add(fos)

        for fos in creator_fos:
            fos_counter[fos] += 1

        multidisciplinarity[len(creator_fos)] += 1

total_fos = sum([multidisciplinarity[key] for key in multidisciplinarity])

for m in multidisciplinarity:
    multidisciplinarity[m] = (multidisciplinarity[m] / total_fos) * 100

fos_counter = dict((v, k) for k, v in fos_counter.items())

print(fos_counter)

fig, ax = plt.subplots(constrained_layout=True)
ax.barh(np.arange(len(fos_counter)),
        [fos_counter[fos] for fos in sorted(fos_counter)],
        color='steelblue', edgecolor='steelblue', log=True)
ax.set_title('Fields of Study in the Dataset')
ax.set_yticks(np.arange(len(fos_counter)))
ax.set_yticklabels([fos_counter[fos].replace('_', ' ').capitalize()
                   for fos in sorted(fos_counter)])
ax.set_ylabel('Fileds of Study')
ax.set_xlabel('# in the Dataset')
fig.savefig('../images/fos/fos_in_the_dataset.pdf', bbox_inches='tight',
            format='pdf')
plt.close(fig)

fig, ax = plt.subplots(constrained_layout=True)
ax.barh(np.arange(len(multidisciplinarity)),
        [multidisciplinarity[k] for k in sorted(multidisciplinarity)],
        color='steelblue', edgecolor='steelblue')
ax.set_title('Multidisciplinarity in the Dataset')
ax.set_yticks(np.arange(len(multidisciplinarity)))
ax.set_yticklabels([k for k in sorted(multidisciplinarity)])
ax.set_xlabel('Percentage')
ax.set_ylabel('Number of FOS')
fig.savefig('../images/fos/multidisciplinarity.pdf', bbox_inches='tight',
            format='pdf')
plt.close(fig)
