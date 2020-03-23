import json
import sys

from collections import defaultdict
from lxml import etree
from tqdm import tqdm

data, xml_name = None, sys.argv[1].split('_')[1]

with open(sys.argv[1], 'r') as xml_file:
    data = defaultdict(list) if xml_name == 'organizations' else dict()

    xml_file.readline()
    xml_file.readline()

    for record in tqdm(xml_file):
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
            for creator in r.findall('.//creator'):
                attributes = dict(creator.attrib)

                for attr in ['rank', 'name', 'surname']:
                    if attr in attributes.keys():
                        del attributes[attr]

                if len(list(attributes.keys())) != 0:
                    data[creator.text.lower()] = attributes

with open('../datasets/in/zenodo_{}.json'.format(xml_name), 'w') as f:
    json.dump(data, f, sort_keys=True, indent=2)
