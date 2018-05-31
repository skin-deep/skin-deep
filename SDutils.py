import sys, os, shutil
import pickle
import datetime
import functools
from sklearn import preprocessing

def log_params(logstr, dest='LOGS', to_print=True, *args, **kwargs):
    logstr = str(logstr)
    dest = os.path.abspath(dest or os.getcwd())
    if not os.path.isdir(dest):
        try: os.makedirs(dest)
        except Exception:
            sys.excepthook(*sys.exc_info())
            print('\nCould not create the log directory!')
    logfname = "{main}{addons}.txt".format(main='log', addons='')
    logfp = os.path.join(dest, logfname)
    with open(logfp, 'a') as log:
        log.write("\n[{timestamp}]: ".format(timestamp=datetime.datetime.utcnow().isoformat()))
        log.write("{logged}".format(logged=logstr))
        if to_print: print(logstr)

        
def inp_batch_norm(expression, verbose=False):
    """Provides a common set of parameters for preprocessing batch normalization between modules."""
    #import keras
    #import keras.backend as K
    
    #import numpy as np
    #if verbose: print("RAW DATA: ", expression)
    #logscale = (np.log(expression))
    #if verbose: print("RAW LOGSCALE: ", logscale)
    
    #logscale = preprocessing.scale(logscale, axis=1)
    #if verbose: print(fix_logscale)
  
    return expression
        
def main(*args, **kwargs):
    pass
    
if __name__ == '__main__':
    print('This module is not meant to be executed directly.')
    input('Press Enter to quit...')
    sys.exit(main(**args))
