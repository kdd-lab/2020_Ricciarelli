import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import Counter, defaultdict
from tqdm import tqdm

fos_dict = dict()

with open(sys.argv[1], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        fos_dict.update(creator)

fos_counter_per_year = defaultdict(Counter)

for mag_id in tqdm(fos_dict, desc='COUNTING FOS PER YEAR'):
    for year in fos_dict[mag_id]:
        for field_of_study in fos_dict[mag_id][year]:
            fos_counter_per_year[year][field_of_study] += 1

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.set_title('Most represented Field of Study per Year', fontsize=10)
ax.plot(np.arange(1980, 2020),
        [fos_counter_per_year[year].most_common()[0][1]
        for year in fos_counter_per_year], lw=2, color='steelblue')

for idx, x in enumerate(np.arange(1980, 2020)):
    field_of_study = fos_counter_per_year[str(x)].most_common()[0][0]
    y = ax.get_children()[0].properties()['data'][1][idx] + 50000
    ax.text(x, y, field_of_study, fontdict={'fontsize': 4, 'rotation': 45,
            'ha': 'center', 'va': 'center'})

ax.set_xlim(1979, 2020)
ax.set_xticks(np.arange(1980, 2020, 10))
ax.set_xticks(np.arange(1980, 2020), minor=True)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Year', fontsize=8)
ax.set_ylabel('Registered Entries', fontsize=8)
fig.savefig('../images/fos/most_represented_fos_per_year.pdf',
            format='pdf')
plt.close(fig)
