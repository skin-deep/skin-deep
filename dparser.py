import glob
import pickle
#import functools
#import inspect
import os, os.path, sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def ask_for_files():
    files = set()
    while not files: 
        filename=str(input('Input the name of the file to parse: '))
        files.update(glob.glob(filename))
        if not files: print('Invalid filename! Try again!')
    return files

# parsers
def parse_datasets(files=None, verbose=False, *args, **kwargs):
    files = glob.glob(files) if files else tuple()
    if not files: 
        if verbose: files = ask_for_files()
    for filename in files:
        table = None
        with open(filename) as file:
            try:
                table = pd.read_table(file, dtype={'a' : str, 'b' : np.float64},
                                     names = (filename, 'values'))
            except Exception as E:
                sys.exchook(type(E), E.message, sys.exc_traceback)
        if verbose: 
            print ('Could not parse {}!'.format(filename) if not table else '{} parsed successfully.'.format(filename))
        yield table


def parse_miniml(files=None, tags=None, junk_phrases=None, verbose=False, *args, **kwargs):
    if tags is None: tags = {'sample'}
    if junk_phrases is None: junk_phrases = {'{http://www.ncbi.nlm.nih.gov/geo/info/MINiML}',}
    files = glob.glob(files) if files else files
    if not files: 
        if verbose: files = ask_for_files()
        else: return list()
    data = []
    for filename in files:
        xml_data = None
        with open(filename) as xml_file:
            xml_data = xml_file.read()
        if xml_data:
            root = ET.XML(xml_data)
            all_records = []
            if verbose: verbose = True if 'n' == input('Mute ([Y]/n): ').lower().strip() else False
            for i, child in enumerate(root):
                record = {}
                if not any((tag.lower() in child.tag.lower() for tag in tags)): continue
                for subchild in child:
                    subtext = (subchild.text or '').strip()
                    subtag = subchild.tag
                    for junk in junk_phrases:
                        subtag = subtag.replace(junk, '')
                    if not all((subtag, subtext)): continue
                    record[subtag] = subtext
                    if verbose: 
                        print (subtext)
                        situation = input('[B]reak/[S]ilence/[Continue]: ').strip()
                        if situation == 'B': return list()
                        if situation == 'S': verbose = False
                all_records.append(record)
            data.append(pd.DataFrame(all_records))
    return data

def dbgpars(*args, avoid_lists=True, **kwargs):
    # testing, shorthand method!
    parsed_xmls = parse_miniml('*.xml', *args, **kwargs)
    return parsed_xmls[0] if (avoid_lists and len(parsed_xmls) == 1) else parsed_xmls

# Low-level, intra-dataset cleaning logic.
def clean_xmls(parsed_input):
    cleaned = (x for x in parsed_input)
    cleaned = (fr.set_index('Accession') for fr in cleaned)
    for clean_xml in cleaned:
        yield clean_xml

def clean_data(raw_data):
    for datum in raw_data:
        cleaned = datum
        # nothing for now
        yield cleaned

# Higher-level logic for integration
def xml_pipeline(*args, **kwargs):
    raw_parse = parse_miniml('*.xml', *args, **kwargs)
    cleaned = tuple(clean_xmls(raw_parse))
    return cleaned
   
def txt_pipeline(*args, **kwargs):
    raw_data = parse_datasets('*.txt', *args, **kwargs)
    cleaned = tuple(clean_data(raw_parse))
    return cleaned

def main():
    descriptors = xml_pipeline()
    data = txt_pipeline()
    tuple(map(print, descriptors))
    return 1

if __name__ == '__main__':
    sys.exit(main())



