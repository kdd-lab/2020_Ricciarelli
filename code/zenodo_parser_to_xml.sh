#!/bin/bash

# THIS SCRIPT IS USED IN ORDER TO AGGREGATE THE RECORDS CONTAINED IN THE DUMPS PROVIDED BY ZENODO. IN ORDER
# TO RUN THE SCRIPT IT IS MUST PROVIDED BOTH THE PATH TO THE UNCOMPRESSED FILE WHICH CONTAINS THE DUMP AND
# THE PATH WHERE THE FINAL XML FILE HAS TO BE SAVED.
#
# HERE IS AN EXAMPLE: bash zenodo_parser_to_xml.sh path/to/uncompressed/file path/where/to/save/file.xml
#

declare -r to_remove='<?xml version="1.0" encoding="UTF-8"?>'

echo $1
echo $2

if [ -f $1 ]; then
    touch $2

    printf '<?xml version="1.0" encoding="UTF-8"?>\n<data>\n' >> $2

    i=0

    while read line; do
        xml_record=`echo $line | jq '.body."$binary"' -r | base64 --decode | bsdtar -x -O`

        echo $xml_record | sed -e "s/^$to_remove//" >> $2


        if [ $i -eq 1000 ]; then
            break
        fi

        ((i=i+1))
    done < $1

    printf '</data>' >> $2
else
    echo "THE FILE DOESN'T EXISTS!"
fi
