import sys, os, shutil
import pickle
import dparser as geo
import pandas as pd
import numpy as np
import itertools as itt
import random
#from pprint import pprint as print

def build_models(datashape, compression_fac=32, activators=('relu', 'sigmoid'), **kwargs):
    import keras
    # A bit more unorthodox than from m import as...
    Model = keras.models.Model
    
    # calculate sizes
    try:
        uncompr_size = datashape[0]
        compr_size = max(1, (min(uncompr_size, uncompr_size // compression_fac)))
    except (IndexError, TypeError):
        uncompr_size = None
        compr_size = compression_fac
        
    # deep levels handling:
    deep_lvls = kwargs.get('depth', 1)
    try: deep_lvls = max(1, abs(int(deep_lvls)))
    except Exception: deep_lvls = 1
    
    clamp_size = lambda S: max(1, min(S, uncompr_size-1))
    lay_sizes = [clamp_size(compr_size * (2**lvl)) for lvl in reversed(range(deep_lvls))] # FIFO sizes for encoders!

    # layers
    inbound = keras.layers.Input(shape=datashape[:-1])
    dummy_in = keras.layers.Input(shape=datashape)  # dummy input for feeding into decoder separately
    
    encoded = keras.layers.Dense(lay_sizes[0], activation=activators[0])(inbound)
    for siz in lay_sizes[1:]: #[0]th is already built
        encoded = keras.layers.Dense(siz, activation=activators[0])(encoded)
        
    if deep_lvls > 1:
        decoded = keras.layers.Dense(lay_sizes[-len(lay_sizes)], activation=activators[1])(encoded)
        for siz in lay_sizes[-2:0:-1]:
            decoded = keras.layers.Dense(siz, activation=activators[0])(decoded)
        # let's make sure the last layer is 1:1 to input no matter what
        decoded = keras.layers.Dense(uncompr_size, activation=activators[0])(decoded)
    else:
        decoded = keras.layers.Dense(uncompr_size, activation=activators[0])(encoded)
        
    # models
    encoder = Model(inbound, encoded)
    autoencoder = Model(inbound, decoded)
    decoder = None#Model(dummy_in, autoencoder.layers[-1](dummy_in))

    return (autoencoder, encoder, decoder)

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
        if shuffle: random.shuffle(data)
        #data = tuple(map(lambda x: pd.DataFrame.from_dict(x, orient='index'), data))
        
        # split data:
        batch_size = len(data)
        ratio = abs(test_to_train)
        ratio = ratio if ratio <= 1 else ratio / 100 # accept percentages as ints too
        test_size = (batch_size * ratio) // 1

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
    
def main(verbose=None, *args, xml=None, txt=None, dir=None, **kwargs):
    dir = os.getcwd() if not dir else (dir if os.path.isabs(dir) else os.path.join(os.getcwd(), dir))
    xml = xml or os.path.join(dir, '*.xml')
    txt = txt or os.path.join(dir, '*.txt')

    # prepare 
    datastream = geo.combo_pipeline(xml_path=xml, txt_path=txt)
    batches = fetch_batch(datastream)
    #print(next(batches))
    #return
    
    train_test_splitter = split_data(batches)
    train_files, test_files = itt.tee(train_test_splitter)
    
    train_files = itt.chain.from_iterable(itt.islice(train_files, 0, None, 2))
    test_files = itt.chain.from_iterable(itt.islice(test_files, 1, None, 2))
    # print(next(train_files), '\n\n', next(test_files))
    # return
    
    # load values for each accession:
    def get_file_data(f):
        return next(geo.txt_pipeline(os.path.join(dir, f + '*'))).tail(10000)
        
    train = (tuple(zip(x, map(get_file_data, x.values()))) for x in train_files)
    test = (tuple(zip(x, map(get_file_data, x.values()))) for x in test_files)
    
    drop_label = True
    retrieve_df = ((lambda x: x[1]) if drop_label 
                    else (lambda pair: pair[1].assign(Label=np.array(pair[0]))))
    train = (tuple(map(retrieve_df, df)) for df in train)
    test = (tuple(map(retrieve_df, df)) for df in test)
    #print ( ( (next(train)) ) )
    #return
    
    # sample the data for size:
    train_size = min(x.shape for x in next(train))
    test_size = min(x.shape for x in next(test))
    if train_size != test_size:
        train_size = min(train_size, test_size)
        raise RuntimeWarning('Unequal sizes of test and train data, assuming the minimal value!')
        
    # Split the data up into single datapoints
    
    train = itt.chain.from_iterable(train) #break it up into individual records
    test = itt.chain.from_iterable(test)
    #print(next(train), '\n'*3, next(test))
    #return

    # train = ((x.T.values, np.array([x.index.values])) for x in train)
    # test = ((x.T.values, np.array([x.index.values])) for x in test)
    train = ((x.T.values) for x in train)
    test = ((x.T.values) for x in test)
    #print(next(train), '\n'*3, next(test))
    #return
    traing = (zip(train, test))
    
    models = build_models(datashape=train_size, depth=4)
    #return
    autoencoder = models[0]
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    def pop_and_put(gen):
        nxt = next(gen) # pop...
        return (nxt, itt.chain([nxt], gen)) #...and return for processing
    fits = 1
    mdl_file = NotImplemented
    while fits:
        fits -= 1
        if not fits or fits < 1:
            while True:
                try:
                    fits = int(input("Enter a number of fittings to perform before prompting again.\n (values <= 0 will terminate the program): "))
                    break
                except Exception as Exc:
                    if True: print(Exc)
                    print("Invalid input. Try again.")
                    
        if fits:
            if mdl_file is NotImplemented: mdl_file = input("If you want to save the model, enter the filename of file to save it to: ")
            
            try: autoencoder.fit_generator(traing, steps_per_epoch=75)#, initial_epoch=fits+1)
            except KeyboardInterrupt: fits = 0
            
            if mdl_file and (not fits % 10 or 0 <= fits < 10): 
                try: os.replace(mdl_file, mdl_file+'.backup')
                except Exception: pass
                autoencoder.save(mdl_file)

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
