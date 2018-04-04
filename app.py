import sys, os, shutil
import pickle
import dparser as geo
import pandas as pd
import numpy as np
import itertools as itt
import collections as coll
import matplotlib.pyplot as plt
from importlib import reload as Reimport
Models = NotImplemented # deferred loading to save on keras bootup time; module-scope mostly for reloads

import config
        

class SkinApp(object):    
    ACT_TRAIN= '(T)rain'
    ACT_PRED = '(P)redict'
    ACT_LOAD = '(L)oad model'
    ACT_SAVE = '(S)ave model'
    ACT_DROP = '(D)rop model'
    ACT_CONF = '(C)onfigure'
    ACT_QUIT = '(Q)uit'

    DBG_DATA = 'DEBUG DATA (!)'
    DBG_MODE = '!lvl'
    
    def __init__(self, *args, **kwargs):
        self.prediction, self.history, self.modelpath = None, None, None
        self.model = [None, None, None]
        self.config = config.MenuConfiguration({self.DBG_MODE : False,}, **kwargs)
        self.actionqueque = coll.deque()
        #self.config.options.update(sorted([('verbosity', verbose), ('xml_path', xml), ('txt_path', txt), ('directory', dir)]))
        
        self.modes = coll.OrderedDict()
        self.modes.update({self.ACT_TRAIN: {'1', 'train', 't'},})
        self.modes.update({self.ACT_PRED : {'2', 'predict', 'p'},})
        self.modes.update({self.ACT_LOAD : {'3', 'load', 'l'},})
        self.modes.update({self.ACT_SAVE : {'4', 'save', 's'},})
        self.modes.update({self.ACT_DROP : {'5', 'drop', 'd'},})
        self.modes.update({self.ACT_CONF : {'6', 'configure', 'c'},})
        self.modes.update({self.ACT_QUIT : {'quit', 'q'},})
    
        self.modes.update({self.DBG_DATA : {'!', '-1'},})
    
        self.baseprompt = """
           __________
          /          \\
          |   MENU   |
          \__________/
 {mdl}
  
 The available options are: 
   - {opts}.
    
>>> """.format(opts=',\n   - '.join(self.modes), mdl="{mdl}")
    pass # just to clarify end of def since init doesn't explicitly return

    
    def run_model(self, models=None, verbose=None, *args, xml=None, txt=None, dir=None, **kwargs):
        mode = kwargs.get('mode', 'train')
        
        datagen, catlabels = geo.build_datastreams_gen(xml=xml, txt=txt, dir=dir, 
                                                       mode=mode, debug=False,
                                                       drop_labels=self.config.options.get('drop_labels', False),
                                                       )
                                                    
        #print("CAT LABELS: ", catlabels)
        sampled, genelabels, size = geo.sample_labels(datagen, self.config.options.get(config.LABEL_SAMPLE_SIZE))
        
        def predfunc(models):
            batch = next(sampled[0])
            orig_vals = np.array([batch.values], dtype='float32')
            prediction = models[config.which_model].predict_on_batch(orig_vals)
            #print("RAW prediction:\n", prediction)
            prediction = geo.parse_prediction(prediction, catlabels)
            prediction = prediction.set_index(genelabels)
            prediction = pd.concat({'original':batch.T, 'predicted':prediction}, axis=1)
            #print(batch, prediction)
            return prediction
            
        def trainfunc(models, e):
            batchgen = ({
                         'expression_in': np.array(x.values), 
                         'expression_out': np.array(x.values), 
                         'diagnosis_in': np.array(catlabels.get(x.index.name)), 
                         'diagnosis_out': np.array(catlabels.get(x.index.name))
                        } 
                         for x in sampled[0])
            #validgen = itt.cycle([np.asarray([next(sampled[1]).values], dtype='float32') for x in range(self.config.options.get('train_steps', 75))])
            history = models[0].fit_generator(
                                              zip(batchgen, batchgen),
                                              #zip(((batch[0] + 0.25 * np.random.normal(size=size), batch[1]) for batch in batchgen), batchgen), 
                                              steps_per_epoch=self.config.options.get('train_steps', 20), 
                                              initial_epoch=e-1, epochs=e,
                                              #validation_data=zip(validgen, validgen),
                                              #validation_steps=self.config.options.get('train_steps', 75)
                                              )
            return history
            
        mode_func = {'test': lambda x, e: (predfunc(x)),
                     'train': lambda x, e: (trainfunc(x, e)),
                    }.get(mode)
        
        built_models = [None]
        print("SIZE: ", size)
        if models is None or not all(models): 
            print(built_models)
            built_models = self.build_models(datashape=size, labels=np.array(tuple(catlabels.values())[-1]),
                                            compression_fac=self.config.options.get('compression_fac', 512),
                                            depth=self.config.options.get('model_depth', 3),
                                            activators=self.config.options.get('activators')
                                            )
        
        def Compile(mdl, i=1, *args, **kwargs): 
            print("DEBUG: Compile kwargs for submodel {no} ({mod}): \n".format(no=i, mod=mdl), kwargs)
            if i==0: mdl.compile(optimizer=kwargs.get('optimizer'), 
                                 loss={'expression_out': kwargs.get('loss'), 'diagnosis_out': 'categorical_crossentropy'},
                                 loss_weights={'expression_out': 1, 'diagnosis_out': 9})
            else: mdl.compile(optimizer=kwargs.get('optimizer'), loss=kwargs.get('loss'))
            return mdl
        models = [models[i] or Compile(mdl=built_models[i], i=i, optimizer='adadelta', loss='logcosh') for (i,x) in enumerate(built_models)]
        self.model = models
        autoencoder = models[0]
        
        fits, totalfits = 1, 0
        savepath = NotImplemented
        result = None
        while fits:
            fits -= 1
            if not fits or fits < 1:
                while True:
                    try:
                        fits = int(self.get_input("Enter a number of fittings to perform before prompting again.\n (value <= 0 to terminate): "))
                        break
                    except (KeyboardInterrupt, EOFError):
                        return
                    except Exception as Exc:
                        if verbose: print(Exc)
                        print("Invalid input. Try again.")
                        
            if fits:
                totalfits = max(totalfits, fits)
                if savepath is NotImplemented: 
                    savepath = self.get_input("If you want to save the result, enter the filename of file to save it to: ")
                try:
                    fitting = mode_func(models, fits)
                    #print("\nRuns remaining: {}".format(fits-1))
                    if result is None or mode=='train': result = fitting
                    else: result = result.join(mode_func(models, fits), how='outer', rsuffix=str(totalfits))
                except KeyboardInterrupt: fits = 0
                #sampled, genelabels, size = geo.sample_labels(datagen, self.config.options.get(config.LABEL_SAMPLE_SIZE, 1000)) #resampling
                
                if savepath and (not savepath is NotImplemented) and (not fits % 10 or 0 <= fits < 10): 
                    try: os.replace(savepath, savepath+'.backup')
                    except Exception: pass
                    
                    if mode == 'train': autoencoder.save(savepath)
                    if mode == 'test': result.to_csv(savepath)
                    
            else: savepath = None if savepath is NotImplemented else savepath
        
        return (models, result, savepath)

    def build_models(self, datashape, labels=None, compression_fac=1024, activators=None, **kwargs):
        try: Reimport(models)
        except (NameError, TypeError): import model_defs
        Models = model_defs
        #print('App build_models kwargs: ', kwargs)
        return Models.build_models(datashape, labels=labels, compression_fac=compression_fac, activators=activators, **kwargs)
    
    def load_model(self, *args, model_path=NotImplemented, **kwargs):
        model, loaded_path = None, None
        
        while model_path is NotImplemented: 
            if kwargs.get('list_cwd'): print("Files in current dir: {}".format(
                                                [(x if os.path.isfile(x) else x + '/') 
                                                for x in os.listdir(os.getcwd())])
                                            )
            try: 
                model_path = self.get_input("Path to load:\n>>> ")
            except (KeyboardInterrupt, EOFError):
                break
            if not os.path.exists(model_path):
                if model_path in {'Q',}: 
                    print('Returning to menu...')
                    break
                else:
                    print('Invalid path.')
                    model_path = NotImplemented
        else:
            try:
                import keras
                model = keras.models.load_model(model_path)
                loaded_path = model_path
            except Exception as Err:
                if kwargs.get('verbose'): sys.excepthook(*sys.exc_info())
                print("\n\nModel could not be loaded from path {}!".format(loaded_path))
            return model, loaded_path
            
    def get_input(self, prompt='>>> ', secondary='>>> '):
        curr_prompt = prompt
        action = NotImplemented
        new_cmds = []
        while action is NotImplemented:
            try: 
                action = self.actionqueque.popleft()
                action = str(action) if action and action is not NotImplemented else None
                if action == 'None': action = None
                if action != (new_cmds or [NotImplemented])[0]: print(curr_prompt + str(action))
            except (IndexError, AttributeError):
                action = NotImplemented
                if not self.actionqueque:
                    try: new_cmds = (str(input(curr_prompt)).split() or [None])
                    except (KeyboardInterrupt, EOFError): action = None
                    self.actionqueque = coll.deque(new_cmds)
            curr_prompt = secondary
        return action
            
    def run(self, *args, **kwargs):
        mainloop = True
        self.actionqueque.extend(kwargs.get('cmd', [None]))
        while mainloop:
            prompt = self.baseprompt.format(mdl=(('\n Currently loaded model: '+ str(self.modelpath))))
            action = str(self.get_input(prompt, '>>> ')).lower()
            if not action: action = NotImplemented
            
            if action in self.modes[self.ACT_TRAIN]: 
                try:
                    _tmp = self.run_model(*args, models=self.model, verbose=self.config.options.get('verbose'), 
                                     xml=self.config.options.get('xml'), txt=self.config.options.get('txt'), 
                                     dir=self.config.options.get('dir'), mode='train',  **kwargs)
                except Exception as Err: 
                    _tmp = None
                    sys.excepthook(*sys.exc_info())
                
                try: _tmpFN = _tmp[2]
                except Exception as Err: _tmpFN = None
                
                try: self.history = _tmp[1]
                except Exception as Err: self.history = None
                
                try: _tmp = _tmp[0]
                except Exception as Err: _tmp = None
                
                self.model = _tmp if _tmp else self.model
                self.modelpath = _tmpFN if (_tmp and _tmpFN) else (str(self.model) if _tmp else self.modelpath)
                
            if action in self.modes[self.ACT_PRED]:
                try:
                    _tmp = self.run_model(*args, models=self.model, verbose=self.config.options.get('verbose'), 
                                            xml=self.config.options.get('xml'), txt=self.config.options.get('txt'), 
                                            dir=self.config.options.get('dir'), 
                                            mode='test', **kwargs
                                        )
                except Exception as Err: 
                    _tmp = None
                    sys.excepthook(*sys.exc_info())
                
                try: _tmpFN = _tmp[2]
                except Exception as Err: _tmpFN = None
                
                try: prediction = _tmp[1]
                except Exception as Err: prediction = None
                
                try: _tmp = _tmp[0]
                except Exception as Err: _tmp = None
                
                model = _tmp if _tmp else self.model
                modelpath = _tmpFN if (_tmp and _tmpFN) else (str(self.model) if _tmp else self.modelpath)
                if prediction is not None: 
                    print(prediction)
                    self.prediction = prediction
                    
            if action in self.modes[self.ACT_LOAD]:
                _tmp, _tmp2 = None, None
                import keras
                try: _tmp, _tmp2 = self.load_model(list_cwd=self.config.options.get('list_cwd', False))
                except Exception as Err: pass
                if _tmp: 
                    model = list(self.model or [None, None, None])
                    model[0] = _tmp
                    #try: model[1] = keras.models.Model(keras.layers.Input(model[0].layers[0].input_shape), [x for x in model[0].layers if 'encoder' in x.name][-1](model[0].layers[0]))
                    #except Exception as E: sys.excepthook(*sys.exc_info())
                    #try: model[2] = keras.models.Model(model[0].layers[0], [x for x in model[0].layers if 'decoder' in x.name][-1](model[1].layers[-1]))
                    #except Exception as E: sys.excepthook(*sys.exc_info())
                    self.model = tuple(model)
                    self.modelpath = str(_tmp2)
                    self.config.options[config.LABEL_SAMPLE_SIZE] = self.model[0].input_shape[-1]
                    print("Model loaded successfully.")
            
            if action in self.modes[self.ACT_SAVE]:
                if self.model[0]:
                    savepath = self.get_input("Enter the filename of file to save it to: ")
                    savepath = savepath.strip() if savepath else savepath
                    if savepath: 
                        self.model[0].save(savepath)
                        self.modelpath = savepath
                        print('Model successfully saved to {}.'.format(savepath))
                else: print('Model not currenly loaded.')
                
            if action in self.modes[self.ACT_DROP]:
                self.model = None
                self.modelpath = None
                
            if action in self.modes[self.ACT_CONF]:
                while True:
                    optstrings = [(repr(Opt), repr(Val)) for Opt, Val in self.config.options.items()]
                    leftspace = "{f1}{space}{f2}".format(f1="{O:<", space=max(map(len, (x[0] for x in optstrings))), f2="}")
                    rightspace = "{f1}{space}{f2}".format(f1="{V:>", space=max(map(len, (x[1] for x in optstrings))), f2="}")
                    lines = ["    > o {lsp}{sep:^3}{rsp} <".format(lsp=leftspace, rsp=rightspace, sep=' - ').format(O=repr(Opt), V=repr(Val)) 
                                                                      for Opt, Val in self.config.options.items()]
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
                    try: option = self.get_input('\n    > ', '\n    > ')
                    except (KeyboardInterrupt, EOFError): option = 'Q'
                    print(' ')
                    
                    if option == 'Q':
                        break
                    
                    try: intable = int(option)
                    except Exception as E: intable = False
                    if intable:
                        try: option = tuple(self.config.options.keys())[intable]
                        except Exception as E: print(E)
                    
                    if option in self.config.options:
                        newval = self.get_input("    - {} => ".format(option))
                        if newval:
                            evaluables = {'None' : None, 'False' : False, 'True' : True}
                            if newval in evaluables: newval = evaluables[newval]
                            else: newval = int(newval) if newval.isnumeric() else newval
                            self.config.options[option] = newval
                            print(" ")
                
            if action in self.modes[self.DBG_DATA]:
                _tmp = geo.build_datastreams_gen(*args, xml=self.config.options.get('xml'), 
                                             txt=self.config.options.get('txt'), dir=self.config.options.get('dir'), 
                                             verbose=self.config.options.get('verbose'), 
                                             debug=self.config.options.get(self.DBG_MODE),
                                             **kwargs)
                try: _tmp = _tmp[0]
                except Exception as Err: 
                    print(Err)
                    _tmp = None
                if _tmp is not None: print((_tmp))
                
            if action == '!eval':
                while True:
                    dbgres = None
                    try: query = self.get_input('DEBUG: >> ')
                    except (KeyboardInterrupt, EOFError): break
                    if not query or 'q' == query.lower(): break
                    try: dbgres = eval(query)
                    except Exception as E: print('>', E)
                    if dbgres is not None: print(dbgres)
                
            if action in self.modes[self.ACT_QUIT]:
                mainloop = False
                action = None
    
def main(verbose=None, *args, xml=None, txt=None, dir=None, **kwargs):
    app_instance = SkinApp(*args, verbose=verbose, xml=xml, txt=txt, dir=dir, **kwargs)
    app_instance.run(*args, **kwargs)
    
if __name__ == '__main__':
    print(' ')
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xml')
    argparser.add_argument('--txt')
    argparser.add_argument('--dir')
    argparser.add_argument('--cmd', nargs='*')
    argparser.add_argument('-v', '--verbose', action='store_true')
    args = vars(argparser.parse_args())
    args['dir'] = args['dir'] or 'mag'
    sys.exit(main(**args))
