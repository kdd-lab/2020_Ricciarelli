import json
import numpy as np
import pandas as pd
import sys

from lxml import etree
from tqdm import tqdm

framework = sys.argv[1].split('/')[-1].split('_')[1]

creators, creators_with_MAG, records_with_year, unknown_creators = 0, 0, 0, 0
projects_per_creator, creators_per_project = list(), list()
creators_dict = dict()

with open(sys.argv[1], 'r') as xml_file:
    for record in tqdm(xml_file,
                       desc='READING {} XML FILE'.format(framework.upper())):
        r = etree.XML(record)

        crtrs = [creator for creator in r.findall('.//creator')]
        title = r.find('.//title').text.lower() if r.find('.//title').text is \
            not None else None
        year = r.find('.//dateofacceptance').text.split('-')[0] if \
            r.find('.//dateofacceptance').text is not None else None

        creators_per_project.append(len(crtrs))

        for creator in crtrs:
            identifier = None
            attributes = dict()

            for k, v in dict(creator.attrib).items():
                if k != 'rank':
                    attributes[k] = v

            text = creator.text

            if 'MAGIdentifier' in attributes:
                identifier = attributes['MAGIdentifier']

                if identifier not in creators_dict:
                    creators_with_MAG += 1
            elif 'name' in attributes and 'surname' in attributes:
                if attributes['name'][0] == ' ':
                        identifier = attributes['name'][1:].lower() + ' ' + \
                            attributes['surname'].lower()
                else:
                    identifier = attributes['name'].lower() + ' ' + \
                        attributes['surname'].lower()
            elif text is not None:
                if ',' in text:
                        identifier = text.split(',')
                        identifier = identifier[1][1:].lower() + ' ' + \
                            identifier[0].lower()
                else:
                    identifier = text.lower()
            else:
                identifier = ''
                unknown_creators += 1

            if identifier != '':
                if identifier not in creators_dict:
                    creators_dict[identifier] = \
                        {'attributes': attributes,
                         'projects': [{'title': title,
                                       'keywords': None,
                                       'year': year}]}

                    creators += 1
                else:
                    creators_dict[identifier]['projects'].append(
                        {'title': title, 'keywords': None, 'year': year})

                if year is not None:
                    records_with_year += 1

for creator in creators_dict:
    projects_per_creator.append(len(creators_dict[creator]['projects']))

creators_per_project = np.round(np.mean(creators_per_project), 2)
projects_per_creator = np.round(np.mean(projects_per_creator), 2)

d = {'Researchers': creators,
     'Researchers with MAG': creators_with_MAG,
     'Unknown Researchers': unknown_creators,
     'Records with Year': records_with_year,
     'Projects per Researcher': projects_per_creator,
     'Researchers per Project': creators_per_project}

pd.DataFrame(data=d, index=[0]).to_csv(
    path_or_buf='../../datasets/zenodo/{}/zenodo_{}_statistics.csv'
    .format(framework, framework), index=False)

with open('../../datasets/zenodo/{}/zenodo_{}_researchers.jsonl'
          .format(framework, framework), 'w') as jsonl_file:
    for creator in tqdm(creators_dict.items(), desc='WRITING JSONL'):
        to_write = dict()
        to_write[creator[0]] = creator[1]

        json.dump(to_write, jsonl_file)
        jsonl_file.write('\n')
