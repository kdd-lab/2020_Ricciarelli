import json
import numpy as np
import pandas as pd
import sys

from collections import Counter, defaultdict
from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

xml_name = sys.argv[2]

data = defaultdict(list) if xml_name == 'organizations' else dict()
researchers = dict() if xml_name != 'organizations' else None
framework = sys.argv[2] if xml_name != 'organizations' else None

researchers_num, researchers_with_MAG, projects_per_researcher = 0, 0, 0
researchers_per_project, subjects = list(), list()

stopwords = stopwords.words('english')

with open(sys.argv[1], 'r') as xml_file:
    for record in tqdm(xml_file, desc='MINING RECORDS'):
        r = etree.XML(record)

        if xml_name == 'organizations':
            country = r.find('.//country').attrib['classname']
            legalname = r.find('.//legalname').text
            original_id = r.find('.//originalId').text

            if original_id is not None and country != '':
                data[country].append({'legal_name': legalname,
                                     'id': original_id})
        else:
            project_title = r.find('.//title').text.lower() if \
                r.find('.//title').text is not None else None
            d_of_acc = r.find('.//dateofacceptance').text
            relevant_date = r.find('.//relevantdate').text

            creators = r.findall('.//creator')
            researchers_per_project.append(len(creators))

            subs = [s.text.lower() for s in r.findall('.//subject')
                    if s.text is not None]

            for s in subs:
                subjects.append(s)

            for creator in creators:
                attributes = dict(creator.attrib)
                name = None

                if 'name' in attributes.keys() and \
                        'surname' in attributes.keys():
                    if attributes['name'][0] == ' ':
                        name = attributes['name'][1:].lower() + ' ' + \
                            attributes['surname'].lower()
                    else:
                        name = attributes['name'].lower() + ' ' + \
                            attributes['surname'].lower()
                elif creator.text is not None:
                    if ',' in creator.text:
                        name = creator.text.split(',')
                        name = name[1][1:].lower() + ' ' + name[0].lower()
                    else:
                        name = creator.text.lower()
                else:
                    name = ''

                if name not in researchers.keys():
                    researchers_num += 1
                    researchers[name] = {'projects': {project_title:
                                         {'framework': framework,
                                          'keyword': subs,
                                          'date of acceptance': d_of_acc,
                                          'relevant date': relevant_date}}}

                    for a in attributes.keys():
                        if a not in ['rank', 'name', 'surname']:
                            researchers[name][a] = attributes[a]

                            if a == 'MAGIdentifier':
                                researchers_with_MAG += 1
                else:
                    researchers[name]['projects'][project_title] = \
                        {'framework': framework, 'keyword': subs,
                         'date of acceptance': d_of_acc,
                         'relevant date': relevant_date}

if xml_name != 'organizations':
    projects_per_researcher = list()

    for n in researchers.keys():
        projects_per_researcher.append(len(researchers[n]['projects'].keys()))

    projects_per_researcher = np.around(np.mean(projects_per_researcher),
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
        to_csv(path_or_buf='../datasets/zenodo/zenodo_{}_statistics.csv'
               .format(xml_name), index=False)

    pd.DataFrame(data=subjects.values(), index=subjects.keys(),
                 columns=['Counter']).\
        to_csv(
            path_or_buf='../datasets/zenodo/zenodo_{}_subjects_statistics.csv'
            .format(xml_name))

if input('\nSAVE JSONL?[Y/N] ') == 'Y':
    with open('../datasets/zenodo_json_xml/zenodo_{}.jsonl'.format(xml_name),
              'w') as f:
        for n in tqdm(researchers.keys(), desc='SAVING JSONL'):
            to_write = dict()

            for a in researchers[n].keys():
                to_write[a] = researchers[n][a]

            to_write['name'] = n

            json.dump(to_write if xml_name != 'organization' else data, f)
            f.write('\n')
