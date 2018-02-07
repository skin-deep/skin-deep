import sys, os, shutil
import pickle
import dparser as geo
import pandas as pd
import numpy as np
import itertools as itt
import collections as coll
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
                   
def load_model(*args, model_path=None, **kwargs):
    model, loaded_path = None, None
    if not model_path: model_path = input("Path to load:\n>>> ")
    try:
        import keras
        model = keras.models.load_model(model_path)
        loaded_path = model_path
    except Exception as Err:
        if kwargs.get('verbose'): sys.excepthook(*sys.exc_info())
        print("\n\nModel could not be loaded from path {}!".format(loaded_path))
    return model, loaded_path
          
def build_datastreams_gen(xml=None, txt=None, dir=None, **kwargs):
    DEBUG = kwargs.get('debug', False)
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
        return next(geo.txt_pipeline(os.path.join(dir, f + '*')))
        
    train = (tuple(zip(x, map(get_file_data, x.values()))) for x in train_files)
    test = (tuple(zip(x, map(get_file_data, x.values()))) for x in test_files)
    # print ( ( (next(train)) ) )
    # return
    
    drop_label = True
    retrieve_df = ((lambda x: x[1]) if drop_label 
                    else (lambda pair: pair[1].assign(Label=np.array(pair[0]))))
    train = ((map(retrieve_df, df)) for df in train)
    test = ((map(retrieve_df, df)) for df in test)
    
    nxt = next(itt.chain.from_iterable(train))
    nxt = nxt.sample(10000)
    train_size = nxt.shape # sample the data for size
    labels = nxt.index.values
    if kwargs.get('verbose'): print('LABELS PICKED: ', labels, '\n')
    
    # drop everything not picked in this sampling
    train = (itt.chain.from_iterable(map(lambda x: x.loc[labels], df) for df in train))
    test = (itt.chain.from_iterable(map(lambda x: x.loc[labels], df) for df in test))
    train = itt.chain((nxt,), train) # return the sampled column for processing
    #print ( ( (next(train)) ) )
    #return (train,)

    # train = ((x.T.values, np.array([x.index.values])) for x in train)
    # test = ((x.T.values, np.array([x.index.values])) for x in test)
    train = ((x.T.values) for x in train)
    test = ((x.T.values) for x in test)
    #print(next(train), '\n'*3, next(test))
    #return
    #traing = (zip(train, test))
    size = train_size
    mode = kwargs.get('mode', 'train')
    datagen = {'train': lambda: zip(train, train), 
            'test': lambda: zip(test, test), 
            'cross': lambda: zip(train, test),}.get(mode, lambda: None)()
    return datagen, size
          
# rename this!
def run_model(models=None, verbose=None, *args, xml=None, txt=None, dir=None, **kwargs):
    
    mode = kwargs.get('mode', 'train')
    datagen, size = build_datastreams_gen(xml=xml, txt=txt, dir=dir, mode=mode)
    mode_func = {'train' : lambda x: x[0].fit_generator(datagen, steps_per_epoch=75),
                 'test' : lambda x: x[1].predict_generator(datagen, steps=75, verbose=1),
                }.get(mode)
    
    built_models = [None]
    if not all(models): built_models = build_models(datashape=size, depth=kwargs.get('depth', 2))
    models = [x or built_models[i] for (i,x) in enumerate(models)]
    autoencoder = models[0]
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    fits = 1
    mdl_file = NotImplemented
    result = None
    while fits:
        fits -= 1
        if not fits or fits < 1:
            while True:
                try:
                    fits = int(input("Enter a number of fittings to perform before prompting again.\n (value <= 0 to terminate): "))
                    break
                except (KeyboardInterrupt, EOFError):
                    return
                except Exception as Exc:
                    if verbose: print(Exc)
                    print("Invalid input. Try again.")
                    
        if fits:
            if mdl_file is NotImplemented: mdl_file = input("If you want to save the model, enter the filename of file to save it to: ")
            
            try: result = mode_func(models)
            except KeyboardInterrupt: fits = 0
            
            if mdl_file and (not fits % 10 or 0 <= fits < 10): 
                try: os.replace(mdl_file, mdl_file+'.backup')
                except Exception: pass
                autoencoder.save(mdl_file)
                
        else: mdl_file = None if mdl_file is NotImplemented else mdl_file
    
    return (models, result, mdl_file)

class MenuConfiguration(object):
    options = coll.OrderedDict()
        
ACT_TRAIN= '(T)rain'
ACT_PRED = '(P)redict'
ACT_LOAD = '(L)oad model'
ACT_SAVE = '(S)ave model'
ACT_DROP = '(D)rop model'
ACT_CONF = '(C)onfigure'
ACT_QUIT = '(Q)uit'

DBG_DATA = 'DEBUG DATA (!)'
    
def main(verbose=None, *args, xml=None, txt=None, dir=None, **kwargs):
    modes = coll.OrderedDict()
    
    modes.update({ACT_TRAIN: {'1', 'train', 't'},})
    modes.update({ACT_PRED : {'2', 'predict', 'p'},})
    modes.update({ACT_LOAD : {'3', 'load', 'l'},})
    modes.update({ACT_SAVE : {'4', 'save', 's'},})
    modes.update({ACT_DROP : {'5', 'drop', 'd'},})
    modes.update({ACT_CONF : {'6', 'configure', 'c'},})
    modes.update({ACT_QUIT : {'0', 'quit', 'q'},})
    
    modes.update({DBG_DATA : {'!', '-1'},})
    
    mainloop = True
    action = NotImplemented
    model = [None, None, None]
    prediction, history, modelpath = None, None, None
    config = MenuConfiguration
    config.options.update(sorted([('verbosity', verbose), ('xml_path', xml), ('txt_path', txt), ('directory', dir)]))
    config.options.update(kwargs)
    
    baseprompt = """
           __________
          /          \\
          |   MENU   |
          \__________/
 {mdl}
  
 The available options are: 
   - {opts}.
    
>>> """.format(opts=',\n   - '.join(modes), mdl="{mdl}")
    
    while mainloop:
        prompt = baseprompt.format(mdl=(('\n Currently loaded model: '+ str(modelpath))))
        if not action: 
            prompt = '>>> '
            action = NotImplemented
        try:
            action = input(prompt).lower()
        except (KeyboardInterrupt, EOFError):
            action = None
            mainloop = False
        
        if action in modes[ACT_TRAIN]: 
            _tmp = run_model(*args, models=model, verbose=verbose, xml=xml, txt=txt, dir=dir, mode='train', **kwargs)
            
            try: _tmpFN = _tmp[2]
            except Exception as Err: _tmpFN = None
            
            try: history = _tmp[1]
            except Exception as Err: history = None
            
            try: _tmp = _tmp[0]
            except Exception as Err: _tmp = None
            
            model = _tmp if _tmp else model
            modelpath = _tmpFN if (_tmp and _tmpFN) else (str(model) if _tmp else modelpath)
            
        if action in modes[ACT_PRED]:
            _tmp = run_model(*args, models=model, verbose=verbose, xml=xml, txt=txt, dir=dir, mode='test', **kwargs)
            
            try: _tmpFN = _tmp[2]
            except Exception as Err: _tmpFN = None
            
            try: prediction = _tmp[1]
            except Exception as Err: prediction = None
            
            try: _tmp = _tmp[0]
            except Exception as Err: _tmp = None
            
            model = _tmp if _tmp else model
            modelpath = _tmpFN if (_tmp and _tmpFN) else (str(model) if _tmp else modelpath)
            if prediction is not None: print(prediction)
                
            
        if action in modes[ACT_LOAD]:
            try: _tmp, _tmp2 = load_model()
            except Exception as Err: _tmp = None
            if _tmp: 
                model = list(model)
                model[0] = _tmp
                modelpath = str(_tmp2)
                print("Model loaded successfully.")
        
        if action in modes[ACT_SAVE]:
            savepath = input("Enter the filename of file to save it to: ")
            savepath = savepath.strip() if savepath else savepath
            if savepath: 
                model[0].save(savepath)
                modelpath = savepath
                print('Model successfully saved to {}.'.format(savepath))
            
        if action in modes[ACT_DROP]:
            model = None
            modelpath = None
            
        if action in modes[ACT_CONF]:
            while True:
                lines = ["    > o {O:<12}{sep:^3}{V:<15} <".format(O=repr(Opt), sep='-', V=repr(Val)) 
                                                                  for Opt, Val in config.options.items()]
                menusize = max(map(len, lines))
                print(" ")
                print("    >>> {siz} <<<"
                        .format(siz='{text:^'+repr(menusize-12)+'}')
                        .format(text="CONFIGURATION:")
                    )
                print("    \n".join(lines))
                print("    >>> {siz} <<<"
                        .format(siz='{text:^'+repr(menusize-12)+'}')
                        .format(text="'Q' - RETURN"))
                try: option = input('\n    > ')
                except (KeyboardInterrupt, EOFError): option = 'Q'
                print(' ')
                
                if option == 'Q':
                    break
                
                if option in config.options:
                    newval = input("    - {} => ".format(option))
                    if newval:
                        newval = {'None' : None, 'False' : False, 'True' : True}.get(newval, newval)
                        config.options[option] = newval
                        print(" ")
            
        if action in modes[DBG_DATA]:
            _tmp = build_datastreams_gen(*args, xml=config.options.get('xml_path'), 
                                         txt=config.options.get('txt_path'), dir=config.options.get('directory'), 
                                         verbose=config.options.get('verbose'), **kwargs)
            try: _tmp = _tmp[0]
            except Exception as Err: 
                print(Err)
                _tmp = None
            if _tmp: print(next(_tmp))
            
        if action == '!eval':
            while True:
                dbgres = None
                try: query = input('DEBUG: >> ')
                except (KeyboardInterrupt, EOFError): break
                if not query or 'q' == query.lower(): break
                try: dbgres = eval(query)
                except Exception as E: print('>', E)
                if dbgres is not None: print(dbgres)
            
        if action in modes[ACT_QUIT]:
            mainloop = False
    
if __name__ == '__main__':
    print(' ')
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xml')
    argparser.add_argument('--txt')
    argparser.add_argument('--dir')
    #argparser.add_argument('--mode')
    argparser.add_argument('-v', '--verbose', action='store_true')
    args = vars(argparser.parse_args())
    args['dir'] = args['dir'] or 'mag'
    sys.exit(main(**args))
