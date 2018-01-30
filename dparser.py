import glob
import pickle
import os, os.path, sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re

DATA_COLNAME = 'Target'

def ask_for_files():
    files = set()
    while not files: 
        filename=str(input('Input the name of the file to parse: '))
        files.update(glob.glob(filename))
        if not files: print('Invalid filename! Try again!')
    return files

# parsers
def parse_datasets(files=None, verbose=False, *args, **kwargs):
    files = glob.glob(files) if files else files
    if not files: 
        if verbose: files = ask_for_files()
    for filepath in files:
        table = []
        filename = os.path.basename(filepath)
        accession = re.match('([A-Z]{3}\d+?)-.*', filename)
        accession = accession.group(1) if accession else filename
        with open(filepath) as file:
            try:
                table = pd.read_table(file,
                                     names = (DATA_COLNAME, accession))
            except Exception as E:
                sys.excepthook(type(E), E.message, sys.exc_traceback)
        if verbose: 
            print ('Could not parse {}!'.format(filename) if not any(table) else '{} parsed successfully.'.format(filename))
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
                        situation = input('\n[B]reak/[S]ilence/[Continue]: ').strip()
                        print('')
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
    cleaned = (get_patient_type(fr) for fr in cleaned)
    #cleaned = (fr.transpose() for fr in cleaned)
    for clean_xml in cleaned:
        yield clean_xml

def get_patient_type(dframe):
    """Retrieves the label of the sample type from the Title field and returns it as a (new) dataframe."""
    #return dframe['Title']
    return dframe.transform({'Title' : lambda x: x.split('_')[2]})

def clean_data(raw_data):
    for datum in raw_data:
        cleaned = datum
        cleaned = cleaned.set_index(DATA_COLNAME)
        #cleaned = cleaned.transpose()
        yield cleaned

# Higher-level logic for integration
def xml_pipeline(path=None, *args, **kwargs):
    raw_parse = parse_miniml(path or '*.xml', *args, **kwargs)
    cleaned = clean_xmls(raw_parse)
    return cleaned
   
def txt_pipeline(path=None, *args, **kwargs):
    raw_data = parse_datasets(path or '*.txt', *args, **kwargs)
    cleaned = clean_data(raw_data)
    return cleaned
    
def combo_pipeline(xml_path=None, txt_path=None, verbose=False, *args, **kwargs):
    xmls = xml_pipeline(path=xml_path, *args, **kwargs)
    #txts = txt_pipeline(path=txt_path, *args, **kwargs)
    count = 0
    
    if False: # TXT-first
        #pool xmls - we'll need to iterate over them repeatedly
        xmls = tuple(xmls)
        
        for txt in txts:
            count += 1
            accession = txt.columns[0]
            sample_type = None
            for xml in xmls:
                sample_type = xml.get(accession)
                #print (sample_type, '\n')
                if sample_type is not None: break
                print(accession)
            if sample_type is None or not len(sample_type):
                if verbose: print('Warning: could not determine any sample type for a sample! Skipping!')
                continue
            yield pd.DataFrame.from_records([txt, sample_type], index=['Accession', 'Type']).transpose()
        
    for xml in xmls:
        sample_groups = xml.groupby('Title').groups
        types = set(sample_groups.keys())
        pos = 0
        while types:
            batch = dict() #dict/set!
            ignored = set()
            for t in types:
                if t in ignored: continue
                try: 
                    batch.update({t : sample_groups[t][pos]}) #in dict
                    count += 1
                except IndexError as IE:
                    ignored.update(t)
            if not len(batch): break
            #print(batch)
            yield batch
            pos += 1
            
   
    print("Found {} datafiles.".format(count))
        

def main(xml=None, txt=None, verbose=None, *args, **kwargs):
    data = combo_pipeline(xml_path=xml, txt_path=txt, verbose=verbose)
    #start = (next(data))
    for d in data:
        #start = start.append(d, ignore_index=True)
        print (d)
        #print(d[0].columns.item(), d[1])
    #print(len(tuple(data)))
    #print(start.groupby('Type').groups)
    return 1

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xml')
    argparser.add_argument('--txt')
    argparser.add_argument('-v', '--verbose', action='store_true')
    args = vars(argparser.parse_args())
    sys.exit(main(**args))



