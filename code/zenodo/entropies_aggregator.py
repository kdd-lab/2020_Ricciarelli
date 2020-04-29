import json
import os
import sys

from tqdm import tqdm

entropies_by_framework = dict()

dirs = [d for d in os.listdir(sys.argv[1]) if d != '.DS_Store']

for year in sorted(dirs):
    if os.path.exists(sys.argv[1] + year + '/' + 'node_list.jsonl') and \
       os.path.exists(sys.argv[1] + year + '/' + 'edge_list.jsonl'):
        file_name = sys.argv[1] + year + '/' + 'entropies.jsonl'

        with open(file_name, 'r') as entropies_by_year:
            for line in tqdm(entropies_by_year,
                             desc='YEAR {}: READING ENTROPIES'.format(year)):
                creator = json.loads(line)

                for identifier in creator:
                    if identifier not in entropies_by_framework:
                        entropies_by_framework[identifier] = \
                            {year: {'entropy': creator[identifier]['entropy'],
                                    'class': creator[identifier]['class'],
                                    'affiliation':
                                        creator[identifier]['affiliation']}}
                    else:
                        entropies_by_framework[identifier][year] = \
                            {'entropy': creator[identifier]['entropy'],
                             'class': creator[identifier]['class'],
                             'affiliation': creator[identifier]['affiliation']}

to_save = sys.argv[1] + 'entropies_by_researcher.jsonl'

with open(to_save, 'w') as entropies_by_researcher_file:
    for creator in tqdm(entropies_by_framework.items(),
                        desc='WRITING ENTROPIES BY RESEARCHER JSONL'
                        .format(year)):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, entropies_by_researcher_file)
        entropies_by_researcher_file.write('\n')
