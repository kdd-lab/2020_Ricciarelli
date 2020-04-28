import json
import os
import sys

from tqdm import tqdm

creators_with_MAG, creators_without_MAG = dict(), dict()
projects = dict()

for jsonl_file in os.listdir(sys.argv[1]):
    if jsonl_file.split('.')[-1] == 'jsonl':
        framework = jsonl_file.split('_')[1]

        with open(sys.argv[1] + jsonl_file, 'r') as creators_jsonl:
            for creator in tqdm(creators_jsonl,
                                desc='READING FRAMEWORK {}'.format(framework)):
                creator_dict = json.loads(creator.strip())
                identifier = list(creator_dict.keys())[0]
                attributes = creator_dict[identifier]['attributes']
                papers = creator_dict[identifier]['papers']

                if 'MAGIdentifier' in attributes:
                    if identifier not in creators_with_MAG:
                        creators_with_MAG[identifier] = \
                            {'attributes': attributes,
                             'papers': {framework: papers}}
                    elif framework not in creators_with_MAG[identifier][
                            'papers']:
                        creators_with_MAG[identifier]['papers'][
                            framework] = papers
                    else:
                        creators_with_MAG[identifier]['papers'][framework]\
                            .append(papers)
                else:
                    if identifier not in creators_without_MAG:
                        creators_without_MAG[identifier] = \
                            {'attributes': attributes,
                             'papers': {framework: papers}}
                    elif framework not in creators_without_MAG[identifier][
                            'papers']:
                        creators_without_MAG[identifier]['papers'][
                            framework] = papers
                    else:
                        creators_without_MAG[identifier]['papers'][
                            framework].append(papers)

for jsonl_file in os.listdir(sys.argv[3]):
    if jsonl_file.split('.')[-1] == 'jsonl':
        framework = jsonl_file.split('_')[3]

        with open(sys.argv[3] + jsonl_file, 'r') as projects_jsonl:
            for project in tqdm(projects_jsonl,
                                desc='READING FRAMEWORK {}'.format(framework)):
                project_dict = json.loads(project.strip())
                identifier = list(project_dict.keys())[0]

                if identifier not in projects:
                    projects[identifier] = project_dict[identifier]

with open(sys.argv[2] + 'researchers_with_MAG.jsonl', 'w') as \
        creators_with_MAG_jsonl:
    for creator in tqdm(creators_with_MAG.items(),
                        desc='WRITING RESEARCHERS WITH MAG'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, creators_with_MAG_jsonl)
        creators_with_MAG_jsonl.write('\n')

with open(sys.argv[2] + 'researchers_without_MAG.jsonl', 'w') as \
        creators_without_MAG_jsonl:
    for creator in tqdm(creators_without_MAG.items(),
                        desc='WRITING RESEARCHERS WITHOUT MAG'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, creators_without_MAG_jsonl)
        creators_without_MAG_jsonl.write('\n')

with open(sys.argv[4] + 'projects.jsonl', 'w') as projects_jsonl:
    for project in tqdm(projects.items(), desc='WRITING PROJECTS'):
        to_write = dict()
        to_write[projects[0]] = projects[1]

        json.dump(to_write, projects_jsonl)
        projects_jsonl.write('\n')
