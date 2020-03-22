import json
import sys

from collections import defaultdict
from lxml import etree
from tqdm import tqdm

root = etree.parse(sys.argv[1]).getroot()
tag = '{http://namespace.openaire.eu/oaf}organization'

data = defaultdict(list)

records = 0

for organization in tqdm(root.iter(tag)):
    country = organization.find('country').attrib['classname']
    legalname = organization.find('legalname').text
    original_id = organization.find('originalId').text

    if original_id is not None and country != '':
        data[country].append({'legal_name': legalname, 'id': original_id})

    records += 1

with open('../datasets/in/zenodo_organizations.json', 'w') as f:
    json.dump(data, f, sort_keys=True, indent=2)
