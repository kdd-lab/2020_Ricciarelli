import json
import sys

from tqdm import tqdm

valid_mag_ids = set()

with open(sys.argv[2], 'r') as entropies_file:
    for row in tqdm(entropies_file, desc='READING ENTROPIES FILE'):
        creator = json.loads(row)

        for mag_id in creator:
            valid_mag_ids.add(mag_id)

fos_dict = dict()

with open(sys.argv[1], 'r') as fos_file:
    for row in tqdm(fos_file, desc='EXTRACTING FIELDS OF STUDY'):
        r = row.strip().split('\t')

        if int(r[1]) >= 1980 and int(r[1]) <= 2019:
            fields = r[2].split('-')

            if r[0] in valid_mag_ids:
                if r[0] not in fos_dict:
                    fos_dict[r[0]] = {r[1]: fields}
                else:
                    if r[1] not in fos_dict[r[0]]:
                        fos_dict[r[0]][r[1]] = fields
                    else:
                        fos_dict[r[0]][r[1]] += fields

with open('./fields_of_study_1980_2019.jsonl', 'w') as fos_file_jsonl:
    for creator in tqdm(fos_dict.items(), desc='WRITING FIELDS OF STUDY'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, fos_file_jsonl)
        fos_file_jsonl.write('\n')
