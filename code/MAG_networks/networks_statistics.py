import gzip
import igraph as ig
import json
import numpy as np
import sys

from tqdm import tqdm

decade = np.arange(int(sys.argv[1]), int(sys.argv[1]) + 10)

authors_affiliations = dict()

file_name = '/home/ricciarelli/mydata/MAG_networks/authors_affiliation.json'

with open(file_name, 'r') as affiliations:
    for affiliation in tqdm(affiliations, desc='READING AFFILIATIONS FILE'):
        if len(json.loads(affiliation.strip())) != 0:
            a, to_delete = json.loads(affiliation.strip()), list()

            for mag_id in a:
                for aff_id in a[mag_id]:
                    if a[mag_id][aff_id]['from'] not in decade:
                        to_delete.append(aff_id)

            for mag_id in a:
                for aff_id in to_delete:
                    del a[mag_id][aff_id]

            if len(a[mag_id]) != 0:
                authors_affiliations.update(a)

affiliations_countries = dict()

file_name = '/home/ricciarelli/mydata/MAG_networks/affiliations_geo.txt'

with open(file_name, 'r') as affiliations_countries_file:
    for affiliation in tqdm(affiliations_countries_file,
                            desc='READING AFFILIATION COUNTRIES FILE'):
        a = affiliation.strip().split('\t')

        affiliations_countries.update(a)

for year in decade:
    g = ig.Graph()

    file_name = '/home/ricciarelli/mydata/MAG_networks/{}/{}.gz'\
        .format(year, year)

    nodes_list, edges_list, weights_list = set(), list(), list()

    with gzip.open(file_name, 'r') as es_list:
        for edge in tqdm(es_list,
                         desc='YEAR {}: READING EDGES LIST'.format(year)):
            e = edge.decode().strip().split(',')

            nodes_list.add(e[0])
            nodes_list.add(e[1])
            edges_list.append((e[0], e[1]))
            weights_list.append(int(e[2]))
