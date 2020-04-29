import base64
import gzip
import json
import sys
import zipfile

from os import remove
from os import path
from tqdm import tqdm

assert path.exists(sys.argv[1])

not_parsed = 0

with gzip.open(sys.argv[1], 'r') as compressed_file:
    with open(sys.argv[2], 'w') as xml_file:
        for line in tqdm(compressed_file, desc='PARSING TO XML'):
            c_rec = json.loads(line)['body']['$binary'].encode('utf-8')
            record = None

            try:
                with open('decoded_record.zip', 'wb') as decoded_record:
                    decoded_record.write(base64.decodebytes(c_rec))

                with zipfile.ZipFile('./decoded_record.zip', 'r') as \
                        zipped_record:
                    with zipped_record.open('body', 'r') as body:
                        record = body.readlines()

                record = [chunk.decode() for chunk in record[1:]]
                record = [chunk.replace('\n', '') for chunk in record]
                record = ''.join(record)

                xml_file.write(record + '\n')
            except Exception as e:
                not_parsed += 1

print('NOT PARSED: {}'.format(not_parsed))

remove('./decoded_record.zip')
