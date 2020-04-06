import numpy as np
import sys

from lxml import etree
from tqdm import tqdm

adjacency_d = dict()

with open(sys.argv[1], 'r') as xml_file:
    for record in tqdm(xml_file, desc='BUILDING ADJANCENCY MATRIX'):
        r = etree.XML(record)

        creators = r.findall('.//creator')
        creators = [int(dict(creator.attrib)['MAGIdentifier']) for creator in
                    creators if 'MAGIdentifier' in dict(creator.attrib)]

        if len(creators) != 0 and r.find('.//title').text is not None and \
                r.find('.//dateofacceptance').text is not None:
            project_title = r.find('.//title').text.lower()
            d_of_acc = r.find('.//dateofacceptance').text.split('-')[0]

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
                            c2: {d_of_acc: [project_title]}}
                    else:
                        if c2 not in adjacency_d[c1].keys():
                            adjacency_d[c1][c2] = {d_of_acc: [project_title]}
                        else:
                            if d_of_acc not in adjacency_d[c1][c2].keys():
                                adjacency_d[c1][c2][d_of_acc] = [project_title]
                            else:
                                adjacency_d[c1][c2][d_of_acc].\
                                    append(project_title)

with open(sys.argv[2], 'w') as edge_list:
    for magid1 in tqdm(adjacency_d.keys(), desc='WRITING EDGE LIST'):
        for magid2 in adjacency_d[magid1].keys():
            for year in adjacency_d[magid1][magid2].keys():
                weight = len(adjacency_d[magid1][magid2][year])
                projects = ','.join(adjacency_d[magid1][magid2][year])

                edge_list.write('{} {} {} {} {}\n'
                                .format(magid1, magid2, year, weight,
                                        projects))
