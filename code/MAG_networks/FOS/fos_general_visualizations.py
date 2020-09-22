import json
import sys

from collections import defaultdict
from tqdm import tqdm

fos_counter = defaultdict(int)

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='READING FOS FILE'):
        creator = json.loads(row)

        for mag_id in creator:
            for year in creator[mag_id]:
                for fos in creator[mag_id][year]:
                    fos_counter[fos] += 1

print(fos_counter)
