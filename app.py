import sys, os, shutil
import pickle
import pandas as pd
import numpy as np
import itertools as itt
import collections as coll
import matplotlib.pyplot as plt
from importlib import reload as Reimport

Models = NotImplemented # deferred loading to save on Keras bootup time; module-scope mostly for reloads
Keras = NotImplemented # lazy loading, see above

import dparser as geo
geo._key_cache = dict() #clean run
import config
import project_logging as Logger
     
def kerasLazy():
    """Helper for lazy-loading Keras."""
    global Keras
    if Keras is NotImplemented:
        import keras as k
        Keras = k
    return Keras

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
        
        catprompt = 'Enter sample type regexes (comma-separated) or leave blank to use cached: '
        cat_regexes = (self.config.options.get('category_regexes') 
                        or kwargs.get('category_regexes') 
                        or {rgx for rgx in (self.get_input(catprompt) or '').strip().split(',') if rgx}
                        or tuple(geo._key_cache.get('Cached-1', {}))
                       )
        label_mapping = (self.config.options.get('label_mapping') 
                         or kwargs.get('label_mapping')
                        )
        
        
        Reimport(geo) # drops the cache...
        catlabels, tries = [], 0
        while len(catlabels) < len(cat_regexes):
            tries += 1
            datagen, catlabels = geo.build_datastreams_gen(xml=xml, txt=txt, dir=dir, 
                                                           mode=mode, debug=False,
                                                           drop_labels=self.config.options.get('drop_labels', False),
                                                           category_regexes=cat_regexes, # ...but we're restoring/resetting it here (indirectly)
                                                           category_labels=label_mapping,
                                                           )
            assert tries < 3
        sampled, genelabels, size = geo.sample_labels(datagen, self.config.options.get(config.LABEL_SAMPLE_SIZE))
        
        # Ugly debug hacks:
        if np.max(size) > 60000: raise RuntimeError('Unsafe size!') # debug - prevents crashes due to OOM
        global _encoding
        _encoding = catlabels # TEMPORARY, MEMOs THE LABELS
        import model_defs
        model_type = model_defs.variational_deep_AE
        
        def predfunc(models):
            batch = next(sampled[0])
            #Logger.log_params(batch)
            #orig_vals = np.array(batch.T.values, dtype='float32')
            
            #NORM
            K = kerasLazy().backend
            expression = K.variable(batch.sort_index().T.values)
            expression, expr_mean, expr_var = K.normalize_batch_in_training(expression, gamma=K.variable([1]), beta=K.variable([0]), reduction_axes=[1])
            expression = K.eval(expression)
            #ENDNORM
            
            prediction = models[config.which_model].predict_on_batch([expression])
            Logger.log_params("Actual type: {}".format(str(batch.index.name)))
            prediction = geo.parse_prediction(prediction, catlabels, batch=batch, genes=genelabels)
            return prediction
            
        def trainfunc(models, e=1):
            trained_model = models[0]
            history = trained_model.fit_generator(
                                                  model_type.batchgen(source=sampled[0], catlabels=catlabels),
                                                  steps_per_epoch=self.config.options.get('train_steps', 60), 
                                                  initial_epoch=e-1, epochs=e,
                                                  validation_data=model_type.batchgen(source=sampled[1], catlabels=catlabels),
                                                  validation_steps=self.config.options.get('test_steps', 30),
                                                  )
            return history
            
        mode_func = {'test': lambda x, e: (predfunc(x)),
                     'train': lambda x, e: (trainfunc(x, e)),
                    }.get(mode)
        
        built_models = [None for x in range(self.config.options.get('model_amt', 3))]
        Logger.log_params("SIZE: " + str(size))
        if models is None or not all(models): 
            print(built_models)
            built_models = self.build_models(datashape=size, kind=model_type, labels=np.array(tuple(catlabels.values())),
                                            compression_fac=self.config.options.get('compression_fac', 512),
                                            depth=self.config.options.get('model_depth', 3),
                                            activators=self.config.options.get('activators'),
                                            )
        
        def Compile(mdl, i=1, *args, **kwargs): 
            Logger.log_params("DEBUG: Compile kwargs for submodel {no} ({mod}): \n".format(no=i, mod=mdl) + str(kwargs))
            def merge_losses(*losses): return kerasLazy().backend.mean(kerasLazy().backend.sum(*[l() for l in losses]))
            if i==0: mdl.compile(optimizer=kwargs.get('optimizer'), 
                                 loss={'diagnosis': 'categorical_crossentropy', 'expression_out': getattr(mdl, 'custom_loss', kwargs.get('loss'))},
                                 loss_weights={'diagnosis': 18, 'expression_out': 2, },
                                 metrics={'diagnosis': 'categorical_accuracy'},
                                 )
            else: mdl.compile(optimizer=kwargs.get('optimizer'), loss=kwargs.get('loss'))
            return mdl
            
        models = [(models or [None]*(i+1))[i] or Compile(mdl=built_models[i], i=i, optimizer='adadelta', loss='mse') for (i,x) in enumerate(built_models)]
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
                    if result is None or mode=='train': result = fitting
                    else: result = result.join(mode_func(models, fits), how='outer', rsuffix=str(totalfits))
                except KeyboardInterrupt: fits = 0
                
                checkpoint = self.config.options.get(config.SAVE_EVERY, 10)
                tail_saves = self.config.options.get(config.SAVE_TAILS, False)
                
                if savepath and (not savepath is NotImplemented) and (fits == 0 or (checkpoint > 0 and not fits % checkpoint) or (tail_saves and (0 <= fits < 10))): 
                    try: os.replace(savepath, savepath+'.backup')
                    except Exception: pass
                    
                    if mode == 'train': 
                        for i,m in enumerate(models): m.save('{}.{}'.format(savepath, i))
                    if mode == 'test': result.to_csv(savepath)
                    
            else: savepath = None if savepath is NotImplemented else savepath
        
        return (models, result, savepath)

    @classmethod
    def build_models(cls, datashape, kind=None, labels=None, compression_fac=512, activators=None, **kwargs):
        try: Reimport(models)
        except (NameError, TypeError): import model_defs
        Models = model_defs
        Logger.log_params('App build_models args: {}'.format(", ".join(map(str, [datashape, kind, labels, compression_fac, kwargs]))))
        built = Models.build_models(datashape, labels=labels, compression_fac=compression_fac, activators=activators, num_classes=len(_encoding), **kwargs)
        return built
    
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
                model = kerasLazy().models.load_model(model_path)
                loaded_path = model_path
            except Exception as Err:
                if kwargs.get('verbose'): sys.excepthook(*sys.exc_info())
                Logger.log_params("\n\nModel could not be loaded from path {}!".format(loaded_path))
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
                    try: new_cmds = [x.strip(' ') for x in (str(input(curr_prompt)).split(';'))] or [None]
                    except (KeyboardInterrupt, EOFError): action = None
                    self.actionqueque = coll.deque(new_cmds)
            curr_prompt = secondary
        return action
            
    def run(self, *args, **kwargs):
        mainloop = True
        self.actionqueque.extend(kwargs.get('cmd') or [])
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
                keras = kerasLazy()
                try: _tmp, _tmp2 = self.load_model(list_cwd=self.config.options.get('list_cwd', False))
                except Exception as Err: pass
                if _tmp: 
                    model = list(self.model or [None, None, None])
                    model[0] = _tmp
                    for i, newmod in enumerate(model):
                        # load weights from the main network to subnetworks, if built
                        if i == 0: continue
                        try: 
                            if newmod is not None: newmod.set_weights(model[0].get_weights())
                        except Exception as E: 
                            sys.excepthook(*sys.exc_info())
                    self.model = tuple(model)
                    self.modelpath = str(_tmp2)
                    if self.config.options.get('verbose'): print(self.model[0].summary())
                    #self.config.options[config.LABEL_SAMPLE_SIZE] = self.model[0].input_shape[-1]
                    Logger.log_params("Loading {}".format(self.modelpath), to_print=False)
                    Logger.log_params("Model loaded successfully.")
            
            if action in self.modes[self.ACT_SAVE]:
                if self.model[0]:
                    savepath = self.get_input("Enter the filename of file to save it to: ")
                    savepath = savepath.strip() if savepath else savepath
                    if savepath: 
                        self.model[0].save(savepath)
                        self.modelpath = savepath
                        Logger.log_params('Model successfully saved to {}.'.format(savepath))
                else: Logger.log_params('Model not currenly loaded.')
                
            if action in self.modes[self.ACT_DROP]:
                self.model = None
                self.modelpath = None
                geo._key_cache = dict()
                
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
    sys.exit(main(**args))
