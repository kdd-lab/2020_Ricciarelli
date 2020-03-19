#!/bin/bash

declare -r path='/Users/gianmarco/Downloads/organization'
declare -r file_name='/Users/gianmarco/Downloads/organization.xml'
declare -r to_remove='<?xml version="1.0" encoding="UTF-8"?>'

touch $file_name

printf '<?xml version="1.0" encoding="UTF-8"?>\n<data>\n' >> $file_name

i=0

while read line; do
    xml_record=`echo $line | jq '.body."$binary"' -r | base64 --decode | bsdtar -x -O`

    echo $xml_record | sed -e "s/^$to_remove//" >> $file_name


    if [ $i -eq 1000 ]; then
        break
    fi

    ((i=i+1))
done < $path

printf '</data>' >> $file_name
