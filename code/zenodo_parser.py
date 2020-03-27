import gzip
import json
import sys
import subprocess

from os import path
from tqdm import tqdm

assert path.exists(sys.argv[1])

with gzip.open(sys.argv[1], 'r') as compressed_file:
    with open(sys.argv[2], 'w') as xml_file:
        for line in tqdm(compressed_file, desc='PARSING TO XML'):
            c_rec = json.loads(line)['body']['$binary']

            record = subprocess.\
                run('echo {} | base64 --decode | bsdtar -x -O'.format(c_rec),
                    shell=True, capture_output=True, text=True).stdout
            record = record.replace('\n', '').replace('\t', '')
            record = record.replace('<?xml version="1.0" encoding="UTF-8"?>',
                                    '')

            xml_file.write(record + '\n')
