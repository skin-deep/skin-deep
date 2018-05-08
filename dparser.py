import glob
import pickle
import os, sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import itertools as itt
import re

import project_logging as Logger

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
    files = glob.iglob(files) if files else files
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

# Low-level, intra-dataset cleaning logic.
def clean_xmls(parsed_input, *args, **kwargs):
    cleaned = (x for x in parsed_input)
    cleaned = (fr.set_index('Accession') for fr in cleaned)
    cleaned = (get_patient_type(fr, keys=kwargs.get('category_regexes'), labels=kwargs.get('category_labels')) for fr in cleaned)
    for clean_xml in cleaned:
        yield clean_xml
        
_key_cache = dict()
def get_patient_type(dframe, keys=None, **kwargs):
    """Retrieves the label of the sample type from the Title field and returns it as a (new) dataframe."""
    print('TITLE IS: {}'.format(dframe['Title']))
    print('KEYS: {}'.format(keys))
    #keys = (keys if isinstance(keys, dict) else {}) or _key_cache.get('Cached-1', {})
    #keys = tuple(keys.values()) if keys and isinstance(keys, dict) else keys
    if not isinstance(keys, dict): keys = set(keys or input('Enter sample type regexes (comma-separated): ').strip().split(','))
    labels = (kwargs.get('labels') 
                or {} 
                or {'Severe':'PP', 'Mild':'PP', 'Normal':'NN', '<other>':'NN', 'Control':'NN', 'Involved':'PP', 'Uninvolved':'PN', 'Lesion':'PN', '_Lesional':'PP', '_Non-lesional':'PN', 'NL':'PN', 'LS':'PP'}
             )
    labels = {str(k).upper(): v for k,v in labels.items()} # standardize keys
    print("KS1!: ", keys)
    keys = {str(k).upper(): labels.get(str(k).upper(), str(k).upper()) for k in keys}
    print("KS2!: ", keys)
    _key_cache['Cached-1'] = keys
    Logger.log_params("CACHE: " + str(_key_cache))

    transformed = dframe.transform({'Title' : lambda x: ("/".join(keys.get(pat, 'ERROR') for pat in keys if re.search(pat, str(x), flags=re.I)) or '<other>')})
    #transformed =  #filter out unlabelled?
    print(transformed)
    return transformed

def clean_data(raw_data):
    for datum in raw_data:
        cleaned = datum
        cleaned = cleaned.set_index(DATA_COLNAME)
        yield cleaned

# Higher-level logic for integration
def xml_pipeline(path=None, *args, **kwargs):
    raw_parse = parse_miniml(path or '*.xml', *args, **kwargs)
    cleaned = clean_xmls(raw_parse, *args, **kwargs)
    return cleaned
   
def txt_pipeline(path=None, *args, **kwargs):
    raw_data = parse_datasets(path or '*.txt', *args, **kwargs)
    cleaned = clean_data(raw_data)
    return cleaned
    return raw_data #debugging purposes
    
def combo_pipeline(xml_path=None, txt_path=None, verbose=False, *args, **kwargs):
    xmls = xml_pipeline(path=xml_path, *args, **kwargs)
    #txts = txt_pipeline(path=txt_path, *args, **kwargs)
    count = 0
    import random

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
                    if random.random() < 0.5: batch.update({t : sample_groups[t][pos]}) #in dict
                    count += 1
                except IndexError as IE:
                    ignored.update(t)
            if not len(batch): break
            #print(batch)
            yield batch
            pos += 1

    print("\nFound {} datafiles. \n".format(count))
    if count < 1: raise FileNotFoundError
    #input("DEBUG: Continue?")
        
        

def fetch_batch(stream, batch_size=10, aggregator=list, *args, **kwargs):
    """A generic buffer that reads and yields N values from a passed generator as an iterable.
    If the stream ends prematurely, returns however many elements could were read.
    
    :param stream: a generator or any another object supporting the next() protocol
    :param batch_size: optional; how many values should be fetched, defaults to 10. send() arguments can change its value
    :param aggregator: optional; callable constructor for the iterable type to return data in. 
                                 args and kwargs are passed as the arguments of the call. Default: list
    """
    bsize = batch_size
    batch = aggregator(*args, **kwargs)
    while stream:
        batch = aggregator(*args, **kwargs)
        for _ in range(batch_size):
            try: batch += aggregator((next(stream),))
            except StopIteration: 
                stream = False
                break
        bsize = yield batch
        
def split_data(batches, test_to_train=0.2, shuffle=True):
    batches = itt.cycle(batches)
    for raw_data in batches:
        if not raw_data: continue
        data = raw_data.copy()
        if shuffle: 
            from random import shuffle as randshuffle
            randshuffle(data)
        #data = (map(lambda x: pd.DataFrame.from_dict(x, orient='index'), data))
        
        # split data:
        batch_size = len(data)
        ratio = abs(test_to_train)
        ratio = ratio if ratio <= 1 else ratio / 100 # accept percentages as ints too
        test_size = max(1, (batch_size * ratio) // 1)

        train, test = itt.chain({}), itt.chain({})
        if batch_size > 1:
            # we know there's at least one element from an earlier check
            for i,x in enumerate(data[1::2]):
                # alternating split to be on the safe side
                if i < test_size: test = itt.chain(test, [x])
                else: train = itt.chain(train, [x])
            train = itt.chain(train, itt.islice(data, 0, None, 2))

        yield train
        yield test
        
def build_datastreams_gen2(xml=None, txt=None, dir=None, drop_labels=False, **kwargs):
    #dir = os.getcwd() if not dir else (dir if os.path.isabs(dir) else os.path.join(os.getcwd(), dir))
    distype = kwargs.get('distype') or 'PP'
    dir = r'D:\GitHub\skin-deep\!Partitioned\\' + distype
    files = glob.iglob(os.path.join(dir, '*.csv'))
    gen = (pd.DataFrame.from_csv(csvfile) for csvfile in files)
    gen = (df.rename_axis(distype) for df in gen)
    gen = itt.cycle(gen)
    return [gen, gen]
        
        
def build_datastreams_gen(xml=None, txt=None, dir=None, drop_labels=False, **kwargs):
    DEBUG = kwargs.get('debug', False)
    if DEBUG: DEBUG = str(DEBUG).strip()
    dir = os.getcwd() if not dir else (dir if os.path.isabs(dir) else os.path.join(os.getcwd(), dir))
    xml = xml or os.path.join(dir, '*.xml')
    txt = txt or os.path.join(dir, '*.txt')

    # prepare 
    categories = dict()
    print("KCG: ", _key_cache.get('Cached-1', {}), categories)
    categories.update(_key_cache.get('Cached-1', {}))
    datastream = combo_pipeline(xml_path=xml, txt_path=txt, **kwargs)
    batches = fetch_batch(datastream)
    
    if DEBUG == '1':
        print(batches)
        _nextbt = next(batches)
        print("NEXT BATCH: ", _nextbt)
        return _nextbt
    
    train_test_splitter = split_data(batches)
    train_files, test_files = itt.tee(train_test_splitter)
    
    train_files = itt.chain.from_iterable(itt.islice(train_files, 0, None, 2))
    test_files = itt.chain.from_iterable(itt.islice(test_files, 1, None, 2))
    
    if DEBUG == '2':
        print(next(train_files), '\n\n', next(test_files))
        return next(train_files)
    
    # load values for each accession:
    def get_file_data(f): return next(txt_pipeline(os.path.join(dir, f + '*')))
        
    train = (tuple(zip(x, map(get_file_data, x.values()))) for x in train_files)
    test = (tuple(zip(x, map(get_file_data, x.values()))) for x in test_files)
    if DEBUG == '3':
        print ( ( (next(train)) ) )
        return (next(train))
    
    drop_label = drop_labels
    def nodrop_retrieve(pair):
        #print("NODROP CATS: ", categories)
        sample_lbl = str(pair[0]).upper()
        try: categories.update({sample_lbl: categories.get(sample_lbl, (sample_lbl if sample_lbl in categories.values() else '<unknown>'))})
        except ValueError: None if str(pair[0]).upper() in categories else sys.excepthook(*sys.exc_info())
        #finally: print("PAIR 0 IS: {}".format([pair[0]]))
            
        retrieved = pair[1]
        retrieved = retrieved.rename_axis(pair[0], axis=0)
        retrieved = retrieved.rename_axis(['Diagnosis'], axis=1)
        retrieved = retrieved.rename(columns=(lambda x: "{} ({})".format(pair[1].columns[0], pair[0])))
        return retrieved

    retrieve_df = ((lambda x: x[1]) if drop_label 
                    else (lambda x: nodrop_retrieve(x)))
    train = ((map(retrieve_df, df)) for df in train)
    test = ((map(retrieve_df, df)) for df in test)
    if DEBUG == '4':
        print (tuple(next(train)))
        return tuple(next(train))

    #tuple(next(train)) # ensures a full set was inspected; extremely hacky, find a better way
    train = (x for x in itt.chain.from_iterable(train))
    test = (x for x in itt.chain.from_iterable(test))
    if DEBUG == '5':
        ntrain = next(train)
        print(ntrain, '\n'*3, next(test))
        return (ntrain)
        
    if DEBUG == 'SAVE':
        for x in range(100):
            ntrain = next(train)
            savepath = '{}/!Partitioned/{}'.format(os.getcwd(), ntrain.columns.name)
            if not os.path.exists(savepath): os.makedirs(savepath)
            savepath += '/{}.csv'.format(ntrain.columns.values[0])
            if os.path.exists(savepath): print("Repeated value!")
            else: ntrain.to_csv(savepath)
        return (ntrain)
        
    # turn categories into indices:
    #categories = set((categories or {}).values())
    Logger.log_params("Raw categories: {}".format(categories), to_print=True)
    categories = ['NN', 'PP', 'PN'] #quick hack to be able to work...
    categories = dict(enumerate(sorted(categories)))
    # one-hot the categories:
    from keras.utils import to_categorical as categ_encode
    category_indices = np.array(tuple(categories.keys()), dtype=int).T
    encoding = categ_encode(category_indices)
    Logger.log_params("Encoding: \n{}".format(encoding), to_print=True)
    categories = {categories[i]:np.array([encoding[i]]) for (i,k) in enumerate(categories)}
    Logger.log_params("Categories: {}".format(categories), to_print=True)
    
    mode = 'cross'#kwargs.get('mode', 'train')
    datagen = {'train': lambda: (train, train), 
            'test': lambda: (test, test), 
            'cross': lambda: (train, test),}.get(mode, lambda: None)()
    return datagen, categories
    
def sample_labels(generators, samplesize=None, offset=0, *args, **kwargs):
    #print(generators)
    out_generators = [None for _ in range(len(generators))]
    
    for i, gen in enumerate(generators):
        nxt = next(generators[i])
        #print(nxt)
        size = nxt.size # sample the data for size
        samplesize = samplesize or size # None input - do not subsample
        safe_size = min(size, samplesize)
        nxt = nxt[0+(safe_size*offset):(safe_size*(offset+1))] if size <= (1+offset)*safe_size else nxt ##nxt.sample(safe_size)
        safe_size = nxt.T.shape
        out_generators[i] = itt.chain((nxt,), generators[i]) # return the sampled column for processing
        #import random
        #out_generators[i] = itt.islice(generators[i], 0, None, random.choice(list(range(1,4)))) # Debug; skips steps to ensure the model generalizes outside the training sequence
        labels = nxt.index.values
    
        if kwargs.get('verbose', True): print('LABELS PICKED: ', labels, '\n')
    
    return out_generators, labels, safe_size

def parse_prediction(predarray, labels=None, *args, **kwargs):
    labels = labels or {}
    batch = kwargs.get('batch')
    #Logger.log_params("RAW: \n{}".format(predarray))
    #if batch is not None: print("BATCH: \n", batch)
    if batch is not None: print("SAMPLE: ", batch.columns.values[-1])
    diagarray, topprob, diagnosis = np.array([predarray[1][-1]]), -1., None
    if labels: Logger.log_params("PROBABILITIES: ")
    for (labl, onehot) in labels.items():
        guess = np.round(np.nanmax(onehot*diagarray)*100, 2)
        Logger.log_params("{LAB}: {PROB}%".format(LAB=labl, PROB=(guess or '<0.01')))
        if guess > topprob: diagnosis, topprob = labl, guess
        #print(guess, topprob, diagnosis)
    predarray = pd.Series(predarray[0].flatten(), name="{}=>({} @{}%)".format(batch.columns[0], diagnosis, np.round(topprob, 0))).to_frame()
    genelabels = kwargs.get('genes')
    diff = pd.Series(np.subtract(predarray.values, batch.values).flatten()).rename('diff for {}'.format(batch.columns.values[-1])).to_frame().set_index(genelabels)
    print(diff)
    if genelabels is not None: predarray = predarray.set_index(genelabels)
    #print("PROCESSED: \n\n", predarray)
    predarray = pd.concat({'original':batch, 'predicted':predarray, 'diff':diff}, axis=1)
    Logger.log_params('---'*30)
    return predarray
    
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



