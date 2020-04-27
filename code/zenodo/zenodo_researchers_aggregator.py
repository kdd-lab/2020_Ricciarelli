import json
import os
import sys

from tqdm import tqdm

creators_with_MAG, creators_without_MAG = dict(), dict()

for jsonl_file in os.listdir(sys.argv[1]):
    if jsonl_file.split('.')[-1] == 'jsonl':
        framework = jsonl_file.split('_')[1]

        with open(sys.argv[1] + jsonl_file, 'r') as creators_jsonl:
            for creator in tqdm(creators_jsonl,
                                desc='READING FRAMEWORK {}'.format(framework)):
                creator_dict = json.loads(creator.strip())
                identifier = list(creator_dict.keys())[0]
                attributes = creator_dict[identifier]['attributes']
                projects = creator_dict[identifier]['projects']

                if 'MAGIdentifier' in attributes:
                    if identifier not in creators_with_MAG:
                        creators_with_MAG[identifier] = \
                            {'attributes': attributes,
                             'projects': {framework: projects}}
                    elif framework not in creators_with_MAG[identifier][
                            'projects']:
                        creators_with_MAG[identifier]['projects'][
                            framework] = projects
                    else:
                        creators_with_MAG[identifier]['projects'][framework]\
                            .append(projects)
                else:
                    if identifier not in creators_without_MAG:
                        creators_without_MAG[identifier] = \
                            {'attributes': attributes,
                             'projects': {framework: projects}}
                    elif framework not in creators_without_MAG[identifier][
                            'projects']:
                        creators_without_MAG[identifier]['projects'][
                            framework] = projects
                    else:
                        creators_without_MAG[identifier]['projects'][
                            framework].append(projects)

with open(sys.argv[2] + 'researchers_with_MAG.jsonl', 'w') as \
        creators_with_MAG_jsonl:
    for creator in tqdm(creators_with_MAG.items(),
                        desc='WRITING RESEARCHERS WITH MAG'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, creators_with_MAG_jsonl)
        jsonl_file.write('\n')

with open(sys.argv[2] + 'researchers_without_MAG.jsonl', 'w') as \
        creators_without_MAG_jsonl:
    for creator in tqdm(creators_without_MAG.items(),
                        desc='WRITING RESEARCHERS WITHOUT MAG'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, creators_without_MAG_jsonl)
        jsonl_file.write('\n')
