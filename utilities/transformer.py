import os, sys, shutil
import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing

def copy_xmls(src, dst):
    for xml in glob.iglob(os.path.join(src, "*.xml")): 
        try: shutil.copy(xml, dst)
        except Exception as E: print('Failed to copy {} - {}.'.format(xml, E))

def transformation(tbl, from_log2=True):
    newtbl = tbl
    if from_log2: newtbl = newtbl.apply(np.exp2)
    newtbl = newtbl.apply(np.log)
    newtbl = newtbl.apply(preprocessing.scale)
    return newtbl

def transform_files(filedir=None, suffix='-tr', retrieve_xmls=True):
    if not os.path.isdir(filedir): raise RuntimeError('No such directory!')
    filedir = filedir or os.getcwd()
    savedir = filedir+str(suffix)
    
    for fname in glob.iglob(os.path.join(filedir, 'GSM*.txt')):
        print("Processing {}...".format(fname))
        table = pd.read_csv(fname, sep=None, index_col=0, header=None)
        table = transformation(table)
        if not os.path.exists(savedir): os.mkdir(savedir)
        table.to_csv(os.path.join(savedir, os.path.basename(fname)), sep='\t', header=False)
        
    if retrieve_xmls:
        try: copy_xmls(filedir, savedir)
        except Exception as E: raise E
        else: print('XMLs retrieved successfully.')
        
def main(targdir=None):
    targdir = (targdir 
               or (sys.argv[1:][0] if len(sys.argv)>1 else None)
               or input('Specify the directory to run: ')
               )
    transform_files(targdir)
    
    print('Done.')
    
if __name__ == '__main__':
    main()
    input('Press Enter to quit...')