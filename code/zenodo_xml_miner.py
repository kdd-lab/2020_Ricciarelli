import json

from collections import defaultdict
from lxml import etree

root = etree.parse('../datasets/in/zenodo_organizations.xml').getroot()
tag = '{http://namespace.openaire.eu/oaf}organization'

data = defaultdict(list)

for organization in root.iter(tag):
    country = organization.find('country').attrib['classname']
    legalname = organization.find('legalname').text
    original_id = organization.find('originalId').text

    data[country].append({'legal_name': legalname, 'id': original_id})

with open('../datasets/in/zenodo_organizations.json', 'w') as f:
    json.dump(data, f, sort_keys=True, indent=2)
