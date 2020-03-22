import json
import sys

from collections import defaultdict
from lxml import etree
from tqdm import tqdm

root = etree.parse(sys.argv[1]).getroot()

tag, data, xml_file = None, None, sys.argv[1].split('_')[1]

if xml_file == 'organizations':
    tag = '{http://namespace.openaire.eu/oaf}organization'
    data = defaultdict(list)
elif xml_file == 'h2020':
    tag = '{http://namespace.openaire.eu/oaf}result'
    data = dict()

for record in tqdm(root.iter(tag)):
    if xml_file == 'organizations':
        country = record.find('country').attrib['classname']
        legalname = record.find('legalname').text
        original_id = record.find('originalId').text

        if original_id is not None and country != '':
            data[country].append({'legal_name': legalname, 'id': original_id})
    elif xml_file == 'h2020':
        creators = record.findall('creator')

        for creator in creators:
            attributes = dict(creator.attrib)

            for attr in ['rank', 'name', 'surname']:
                if attr in attributes.keys():
                    del attributes[attr]

            if len(list(attributes.keys())) != 0:
                data[creator.text.lower()] = attributes

with open('../datasets/in/zenodo_{}.json'.format(xml_file), 'w') as f:
    json.dump(data, f, sort_keys=True, indent=2)
