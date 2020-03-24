import json
import numpy as np
import pandas as pd
import sys

from collections import Counter, defaultdict
from lxml import etree
from tqdm import tqdm

xml_name = sys.argv[1].split('_')[1]

data = defaultdict(list) if xml_name == 'organizations' else dict()
researchers = dict() if xml_name != 'organizations' else None

researchers_num, researchers_with_MAG, projects_per_researcher = 0, 0, 0
researchers_per_project, subjects = list(), list()

with open(sys.argv[1], 'r') as xml_file:
    xml_file.readline()
    xml_file.readline()

    for record in tqdm(xml_file, desc='MINING RECORDS'):
        r = None

        try:
            r = etree.XML(record)
        except Exception as e:
            break

        if xml_name == 'organizations':
            country = r.find('.//country').attrib['classname']
            legalname = r.find('.//legalname').text
            original_id = r.find('.//originalId').text

            if original_id is not None and country != '':
                data[country].append({'legal_name': legalname,
                                     'id': original_id})
        elif xml_name == 'h2020' or xml_name == 'fp7':
            creators = r.findall('.//creator')
            researchers_per_project.append(len(creators))

            [subjects.append(s.text.lower()) for s in r.findall('.//subject')
             if s.text is not None]

            for creator in creators:
                attributes = dict(creator.attrib)

                if creator.text is not None and 'name' in attributes.keys()  \
                        and 'surname' in attributes.keys():
                    name = attributes['name'].lower() + ' ' + \
                        attributes['surname'].lower()

                    if name not in researchers.keys():
                        researchers_num += 1
                        researchers[name] = {'projects': 1}

                        for attr in ['rank', 'name', 'surname']:
                            if attr in attributes.keys():
                                del attributes[attr]

                        if len(list(attributes.keys())) != 0:
                            data[name] = attributes
                            researchers[name]['identifiers'] = attributes

                            if 'MAGIdentifier' in attributes.keys():
                                researchers_with_MAG += 1
                        else:
                            researchers[name]['identifiers'] = None
                    else:
                        researchers[name]['projects'] += 1

if xml_name != 'organizations':
    projects_per_researcher = np.around(np.mean([researchers[n]['projects']
                                        for n in researchers.keys()]),
                                        decimals=2)
    researchers_per_project = np.around(np.mean(researchers_per_project),
                                        decimals=2)
    subjects = Counter(subjects)

    d = {'Researchers': researchers_num,
         'Researchers with MAG': researchers_with_MAG,
         'Projects per Researcher': projects_per_researcher,
         'Researchers per Project': researchers_per_project,
         'Most popular Subject': subjects.most_common(1)[0][0]}

    pd.DataFrame(data=d, index=[0]).\
        to_csv(path_or_buf='../datasets/in/zenodo_{}_statistics.csv'
               .format(xml_name), index=False)

with open('../datasets/zenodo/zenodo_{}.json'.format(xml_name), 'w') as f:
    json.dump(data if xml_name == 'organizations' else researchers, f,
              sort_keys=True, indent=2)
