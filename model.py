import sys, os
import pickle
import dparser as geo
import pandas as pd
import itertools as itt
import random
        
def fetch_batch(stream, batch_size=10, aggregator=list, *args, **kwargs):
    """A generic buffer that reads and yields N values from a passed generator as an iterable.
    If the stream ends prematurely, returns however many elements could were read.
    
    :param stream: a generator or any another object supporting the next() protocol
    :param batch_size: optional; how many values should be fetched, defaults to 10. send() arguments can change its value
    :param aggregator: optional; callable constructor for the iterable type to return data in. 
                                 args and kwargs are passed as the arguments of the call. Default: list
    """
    bsize = batch_size
    while stream:
        batch = aggregator(*args, **kwargs)
        for _ in range(batch_size):
            #print (next(stream))
            try: batch += aggregator((next(stream),))
            except StopIteration: 
                stream = False
                break
        bsize = yield batch
        
def split_data(batches, batch_size=5, compressed=32, test_to_train=0.2, shuffle=True):
    batches = itt.cycle(batches)
    #batches = itt.permutations(batches)
    for raw_data in batches:
        if not raw_data: continue
        data = raw_data.copy()
        if shuffle: random.shuffle(data)
        #data = tuple(map(lambda x: pd.DataFrame.from_dict(x, orient='index'), data))
        
        # split data:
        batch_size = len(data)
        split = abs(test_to_train)
        split = split if split <= 1 else split / 100 # accept percentages as ints too
        test_size = (batch_size * test_to_train) // 1
        
        train, test = [data[0]], []
        if batch_size > 1: 
            # we know there's at least one element from an earlier check
            for i,x in enumerate(data[1::2]):
                # alternating split to be on the safe side
                if i < test_size: test.append(x)
                else: train.append(x)
            train.extend((x for x in data [2::2]))
            
        #input_shape = min(series.shape for series, ptype in data)
        yield train, test
        
        
def build_models(datashape, compression_fac=32, activators=('relu', 'sigmoid')):
    import keras
    # A bit more unorthodox than from m import as...
    Model = keras.models.Model
    
    #calculate sizes
    uncompr_size = datashape[0]
    compr_size = min(uncompr_size, uncompr_size // compression_fac)
    
    #layers
    inbound = keras.layers.Input(shape=datashape)
    dummy_in = keras.layers.Input(shape=datashape) # dummy input for feeding into decoder separately
    encoded = keras.layers.Dense(compr_size, activation=activator[0])(inbound)
    decoded = keras.layers.Dense(uncompr_size, activation=activator[1])(encoded)
    
    #models
    encoder = Model(inbound, encoded)
    autoencoder = Model(inbound, decoded)
    decoder = Model(dummy_in, autoencoder.layers[-1](dummy_in))
    
    return (autoencoder, encoder, decoder)   
    
#def prepare_batch(datastream):
    
    
def main(verbose=None, *args, xml=None, txt=None, dir=None, **kwargs):
    dir = os.getcwd() if not dir else (dir if os.path.isabs(dir) else os.path.join(os.getcwd(), dir))
    xml = xml or os.path.join(dir, '*.xml')
    txt = txt or os.path.join(dir, '*.txt')
    
    models = build_models()
    autoencoder = models[0].compile(optimizer='adadelta', loss='binary_crossentropy')
    datastream = geo.combo_pipeline(xml_path=xml, txt_path=txt)
    batches = fetch_batch(datastream)
    train_test_splitter = split_data(batches)
    
    for splitted_data in train_test_splitter:
        #print(a)
        for i, train_or_test in enumerate(splitted_data):
            for di in train_or_test:
                for acc, label in di.items():
                    #print('Train: ' if not i%2 else ' Test: ', acc, label)
                    record = geo.txt_pipeline(path=os.path.join(dir, acc + '*.txt'))
                    #print(next(record).head(), label)
    
        if input(''): break

if __name__ == '__main__':
    print('')
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xml')
    argparser.add_argument('--txt')
    argparser.add_argument('--dir')
    argparser.add_argument('-v', '--verbose', action='store_true')
    args = vars(argparser.parse_args())
    args['dir'] = args['dir'] or 'mag'
    sys.exit(main(**args))
