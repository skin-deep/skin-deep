import os
import glob
import pandas as pd
import numpy as np

def transformation(tbl):
    newtbl = tbl
    newtbl = tbl.apply(np.log)
    return newtbl

def transform_files(filedir=None, suffix='-tr'):
    if not os.path.isdir(filedir): raise RuntimeError('No such directory!')
    filedir = filedir or os.getcwd()
    for fname in glob.iglob(os.path.join(filedir, 'GSM*.txt')):
        table = pd.read_csv(fname, sep=None, index_col=0, header=None)
        table = transformation(table)
        savedir = filedir+str(suffix)
        if not os.path.exists(savedir): os.mkdir(savedir)
        table.to_csv(os.path.join(savedir, os.path.basename(fname)), sep='\t', header=False)
        
def main():
    targdir = input('Specify the directory to run: ')
    transform_files(targdir)
    
if __name__ == '__main__':
    main()
    input('Press Enter to quit...')