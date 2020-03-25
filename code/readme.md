# Code
This directory contains my codebase, where every script fulfils a particular 
task .

* The bash script *zenodo_parser_to_xml.sh* is used to convert the compressed 
OpenAIRE Research Graph dumps, which can be downloaded via 
[this link](https://zenodo.org/record/3516918#.Xnt-mtNKgp9), to XML files that
can be freely processed via script or via XML parser.
* The Python script *zenodo_xml_miner.py* takes the XML files produced by the
previous script and extracts the informations contained in every record, 
finally saving them as ease-to-read CSV files that can be found in the 
*datasets* directory. The script also builds a JSON archive for every XML file.
* The *notebook* subdirectory contains the Python notebooks I used for a more
visual approach to the analysis of some of the datasets I studied.
