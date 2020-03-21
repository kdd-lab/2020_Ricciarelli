#!/bin/bash

# This script aggregates the records provided by the Zenodo's dumps in a XML file which is provided by the user.
# In order to run the script two positional arguments must be provided, that is, the path to the uncompressed dump
# and the path where the XML have to be saved.
#
# EXAMPLE: bash zenodo_parser_to_xml.sh path/to/dump path/where/the/file/have/to/be/saved.xml

declare -r to_remove='<?xml version="1.0" encoding="UTF-8"?>'

if [ -f $1 ]; then
    touch $2

    printf '<?xml version="1.0" encoding="UTF-8"?>\n<data>\n' >> $2

    i=0

    while read line; do
        xml_record=`echo $line | jq '.body."$binary"' -r | base64 --decode | bsdtar -x -O`

        echo $xml_record | sed -e "s/^$to_remove//" >> $2


        # if [ $i -eq 9999 ]; then
        #     break
        # fi

        ((i=i+1))
    done < $1

    printf '</data>' >> $2
else
    echo "THE FILE DOESN'T EXISTS!"
fi
