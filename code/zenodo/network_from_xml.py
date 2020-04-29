import json
import numpy as np
import os
import sys

from lxml import etree
from scipy.stats import mode
from tqdm import tqdm

node_dict, adjacency_d = dict(), dict()
framework = sys.argv[1].split('/')[-1].split('_')[1]

with open(sys.argv[1], 'r') as xml_file:
    for record in tqdm(xml_file, desc='BUILDING NODE LIST AND EDGE LIST'):
        r = etree.XML(record)

        creators = r.findall('.//creator')
        creators = [int(dict(creator.attrib)['MAGIdentifier']) for creator in
                    creators if 'MAGIdentifier' in dict(creator.attrib)]
        title = r.find('.//title').text.lower() if r.find('.//title').text is \
            not None else None
        year = r.find('.//dateofacceptance').text.split('-')[0] if \
            r.find('.//dateofacceptance').text is not None else None
        affiliations = r.findall(".//rel[@inferenceprovenance = "
                                 "'iis::document_affiliations']")
        affiliations_list = list()
        affiliation = None

        if len(affiliations) != 0:
            for aff in affiliations:
                if aff.find('country') is not None:
                    affiliations_list.append(aff.find('country').
                                             attrib['classname'].lower())

            if len(affiliations_list) != 0:
                affiliation = str(mode(affiliations_list)[0][0])

        context = r.find(".//context[@type = 'community']")
        project_id, acronym, label = None, None, None

        if context is not None:
            project_ids = list()

            for concept in context.findall('.//concept'):
                project_ids.append(concept.attrib['id'])

            if len(project_ids) != 0:
                project_id = max(project_ids, key=len)
                label = context.find(".//concept[@id = '{}']"
                                     .format(project_id))
                label = label.attrib['label']
                acronym = r.find('.//acronym').text if r.find('.//acronym') \
                    is not None else None

        if len(creators) != 0 and affiliation is not None and year is not None:
            for creator in creators:
                if year not in node_dict:
                    node_dict[year] = {creator: affiliation}
                elif creator not in node_dict[year]:
                    node_dict[year][creator] = affiliation
                else:
                    if type(node_dict[year][creator]) == str:
                        c = node_dict[year][creator]
                        node_dict[year][creator] = [c, affiliation]
                    else:
                        node_dict[year][creator].append(affiliation)

            infos = [title] if project_id is None else [title, project_id]

            for idx in np.arange(len(creators) - 1):
                cowork = [creators[i] for i in np.arange(idx + 1,
                                                         len(creators))]
                for c in cowork:
                    c1, c2 = None, None

                    if creators[idx] < c:
                        c1, c2 = creators[idx], c
                    else:
                        c1, c2 = c, creators[idx]

                    if year not in adjacency_d:
                        adjacency_d[year] = {c1: {c2: [infos]}}
                    elif c1 not in adjacency_d[year]:
                        adjacency_d[year][c1] = {c2: [infos]}
                    else:
                        if c2 not in adjacency_d[year][c1]:
                            adjacency_d[year][c1][c2] = [infos]
                        else:
                            adjacency_d[year][c1][c2].append(infos)

for year in node_dict:
    for creator in node_dict[year]:
        if type(node_dict[year][creator]) != str:
            node_dict[year][creator] = mode(node_dict[year][creator])[0][0]

for year in sorted(list(node_dict.keys())):
    if not os.path.isdir(sys.argv[2] + framework + '/' + year):
        os.mkdir(sys.argv[2] + framework + '/' + year)

    file_name = sys.argv[2] + framework + '/' + year + '/node_list.jsonl'

    with open(file_name, 'w') as node_list:
        for creator in tqdm(node_dict[year].items(),
                            desc='YEAR {}: WRITING NODE LIST'.format(year)):
            to_write = dict()
            to_write[creator[0]] = creator[1]

            json.dump(to_write, node_list)
            node_list.write('\n')

for year in sorted(list(adjacency_d.keys())):
    file_name = sys.argv[2] + framework + '/' + year + '/edge_list.jsonl'

    with open(file_name, 'w') as edge_list:
        for edge in tqdm(adjacency_d[year].items(),
                         desc='YEAR {}: WRITING EDGE LIST'.format(year)):
            to_write = dict()
            to_write[edge[0]] = edge[1]

            json.dump(to_write, edge_list)
            edge_list.write('\n')
