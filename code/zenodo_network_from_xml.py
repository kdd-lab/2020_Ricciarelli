import json
import numpy as np
import sys

from lxml import etree
from tqdm import tqdm

adjacency_d = dict()
projects_table = dict()

with open(sys.argv[1], 'r') as xml_file:
    for record in tqdm(xml_file, desc='BUILDING ADJANCENCY MATRIX'):
        r = etree.XML(record)

        creators = r.findall('.//creator')
        creators = [int(dict(creator.attrib)['MAGIdentifier']) for creator in
                    creators if 'MAGIdentifier' in dict(creator.attrib)]

        if len(creators) != 0 and r.find('.//title').text is not None and \
                r.find('.//dateofacceptance').text is not None:
            paper_title = r.find('.//title').text.lower()
            d_of_acc = r.find('.//dateofacceptance').text.split('-')[0]
            id_label = None
            acronym = None if r.find('.//acronym') is None else \
                r.find('.//acronym').text

            context = r.find(".//context[@type = 'community']")

            if context is not None:
                context = context.getchildren()

                if len(context) != 0:
                    id_label = dict(context[0].getchildren()[-1].attrib)

            infos = [paper_title]

            if id_label is not None and acronym is not None:
                if id_label['id'] not in projects_table.keys():
                    projects_table[id_label['id']] = {
                        'label': id_label['label'], 'acronym': acronym}

                infos.append(id_label['id'])

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

with open(sys.argv[2], 'w') as edge_list:
    for i in tqdm(adjacency_d.items(), desc='WRITING EDGE LIST'):
        to_write = dict()
        to_write[i[0]] = i[1]

        json.dump(to_write, edge_list)
        edge_list.write('\n')

if len(projects_table.keys()) != 0:
    with open(sys.argv[3], 'w') as table:
        for i in tqdm(projects_table.items(), desc='WRITING PROJECTS TABLE'):
            to_write = dict()
            to_write[i[0]] = i[1]

            json.dump(to_write, table)
            table.write('\n')
