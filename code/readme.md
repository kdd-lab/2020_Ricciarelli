# Code
This directory contains my codebase, where every script fulfils a specific 
task .

* The Python scripts *zenodo_parser.py* and *zenodo_xml_miner.py* are used to 
convert the compressed OpenAIRE Research Graph dumps, which can be downloaded 
via 
[this link](https://zenodo.org/record/3516918#.Xnt-mtNKgp9), to XML files and 
to extract the informations contained in every record, finally saving them as 
ease-to-read CSV files that can be found in the *datasets* directory. 
* *weighted_coauthors_network_analysis* is used in order to extract some basic
network analysis statistics from the coauthors networks. The script provides
also some visualizations for the degrees' distribution, the weights' 
distribution and other metrics.
* *network_from_xml.py* takes a data dump from Zenodo and turns it into a 
collaboration network in which each node represents a researcher with a MAG 
identifier, while each edge represents a collaboration between two researchers
for a particular project.
* The *notebook* subdirectory contains the Python notebooks I used for a more
visual approach to the analysis of some of the datasets I studied.
