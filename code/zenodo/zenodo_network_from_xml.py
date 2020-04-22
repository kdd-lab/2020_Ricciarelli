import json
import numpy as np
import sys

from lxml import etree
from tqdm import tqdm

creators_infos = dict()
adjacency_d = dict()
projects_table = dict()

with open(sys.argv[1], 'r') as xml_file:
    for record in tqdm(xml_file, desc='BUILDING ADJANCENCY MATRIX'):
        r = etree.XML(record)

        creators = r.findall('.//creator')
        creators = [int(dict(creator.attrib)['MAGIdentifier']) for creator in
                    creators if 'MAGIdentifier' in dict(creator.attrib)]
        affiliation = r.find(".//rel[@inferenceprovenance = "
                             "'iis::document_affiliations']")
        project_title = r.find('.//title').text
        d_of_acc = r.find('.//dateofacceptance').text

        if affiliation is not None:
            if len(creators) != 0 and project_title is not None and \
                    d_of_acc is not None and \
                    affiliation.find('country') is not None:
                country = affiliation.find('country').\
                    attrib['classname'].lower()
                d_of_acc = d_of_acc.split('-')[0]
                project_title = project_title.lower()

# STORING INFOS FOR THE NETWORK'S NODES #######################################

                for creator in creators:
                    if creator not in creators_infos:
                        creators_infos[creator] = {country: [d_of_acc]}
                    else:
                        if country not in creators_infos[creator]:
                            creators_infos[creator][country] = [d_of_acc]
                        else:
                            if d_of_acc not in \
                                    creators_infos[creator][country]:
                                creators_infos[creator][country]. \
                                    append(d_of_acc)
                                creators_infos[creator][country] = \
                                    sorted(creators_infos[creator][country])

                id_and_label = None
                acronym = None if r.find('.//acronym') is None else \
                    r.find('.//acronym').text

                context = r.find(".//context[@type = 'community']")

                if context is not None:
                    context = context.getchildren()

                    if len(context) != 0:
                        id_and_label = dict(context[0].getchildren()[-1].
                                            attrib)

                infos = [project_title]

                if id_and_label is not None and acronym is not None:
                    if id_and_label['id'] not in projects_table.keys():
                        projects_table[id_and_label['id']] = {
                            'label': id_and_label['label'], 'acronym': acronym}

                    infos.append(id_and_label['id'])

# STORING INFOS FOR THE NETWORK'S EDGES #######################################

                for idx in np.arange(len(creators) - 1):
                    cowork = [creators[i] for i in np.arange(idx + 1,
                                                             len(creators))]
                    for c in cowork:
                        c1, c2 = None, None

                        if creators[idx] < c:
                            c1, c2 = creators[idx], c
                        else:
                            c1, c2 = c, creators[idx]

                        if c1 not in adjacency_d.keys():
                            adjacency_d[c1] = {
                                c2: {d_of_acc: [infos]}}
                        else:
                            if c2 not in adjacency_d[c1].keys():
                                adjacency_d[c1][c2] = {d_of_acc: [infos]}
                            else:
                                if d_of_acc not in adjacency_d[c1][c2].keys():
                                    adjacency_d[c1][c2][d_of_acc] = [infos]
                                else:
                                    adjacency_d[c1][c2][d_of_acc].append(infos)

framework = sys.argv[1].split('/')[-1].split('_')[1]

# WRITING THE NETWORK'S NODE LIST #############################################

with open('./' + framework + '_node_list.jsonl', 'w') as node_list:
    for i in tqdm(creators_infos.items(), desc='WRITING NODE LIST'):
        to_write = dict()
        to_write[i[0]] = i[1]

        json.dump(to_write, node_list)
        node_list.write('\n')

# WRITING THE NETWORK'S EDGE LIST #############################################

with open('./' + framework + '_edge_list.jsonl', 'w') as edge_list:
    for i in tqdm(adjacency_d.items(), desc='WRITING EDGE LIST'):
        to_write = dict()
        to_write[i[0]] = i[1]

        json.dump(to_write, edge_list)
        edge_list.write('\n')

# WRITING THE PROJECTS' TABLE #################################################

if len(projects_table.keys()) != 0:
    save_to = '../datasets/zenodo/zenodo_' + framework + \
        '_project_table.jsonl'

    with open(save_to, 'w') as table:
        for i in tqdm(projects_table.items(), desc='WRITING PROJECTS TABLE'):
            to_write = dict()
            to_write[i[0]] = i[1]

            json.dump(to_write, table)
            table.write('\n')
