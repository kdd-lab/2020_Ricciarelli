import json
import sys

from collections import Counter
from tqdm import tqdm

fos_counter = list()

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        for mag_id in creator:
            for year in creator[mag_id]:
                fos_file.append(fos for fos in creator[mag_id][year])

fos_counter = Counter(fos_counter)
print(fos_counter)
