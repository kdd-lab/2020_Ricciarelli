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

fos_counter = defaultdict(int)

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        for mag_id in creator:
            for year in creator[mag_id]:
                for fos in creator[mag_id][year]:
                    fos_counter[fos] += 1
import ipdb
ipdb.set_trace()
fig, ax = plt.subplots(constrained_layout=True)
ax.barh(np.arange(len(fos_counter)),
        [fos_counter[fos] for fos in sorted(fos_counter)], color='steelblue',
        edgecolor='steelblue', log=True)
ax.set_title('Fields of Study in the Dataset')
ax.set_yticks(np.arange(len(fos_counter)))
ax.set_yticklabels([fos for fos in sorted(fos_counter)])
fig.savefig('../images/fos/fos_in_the_dataset.pdf', bbox_inches='tight',
            format='pdf')
plt.close(fig)
