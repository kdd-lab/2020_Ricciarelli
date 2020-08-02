import json
import sys

from tqdm import tqdm

time_series_raws = dict()

with open(sys.argv[1], 'r') as time_series_json_file:
    for row in tqdm(time_series_json_file, desc='READING TIME SERIES JSON'):
        creator = json.loads(row)

        time_series_raws.update(creator)

for mag_id in time_series_raws:
    print(time_series_raws[mag_id])
