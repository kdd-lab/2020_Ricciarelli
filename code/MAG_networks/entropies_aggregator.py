import json
import os
import sys

from tqdm import tqdm

global_entropies = dict()

for entropies_file in os.listdir(sys.argv[1]):
    with open(sys.argv[1] + entropies_file, 'r') as entropies_jsonl:
        for creator in tqdm(entropies_jsonl,
                            desc='READING FILE {}'.format(entropies_file)):
            c = json.loads(creator)

            for MAG_id in c:
                if MAG_id not in global_entropies:
                    global_entropies[MAG_id] = c[MAG_id]
                else:
                    global_entropies[MAG_id].update(c[MAG_id])

with open(sys.argv[1] + 'global_entropies.jsonl', 'w') as g_entropies_jsonl:
    for creator in tqdm(global_entropies.items(),
                        desc='WRITING GLOBAL ENTROPIES'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, g_entropies_jsonl)
        g_entropies_jsonl.write('\n')
