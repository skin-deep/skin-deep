import os, shutil
import glob

def transform_files(filedir=None, suffix='-part'):
    if not os.path.isdir(filedir): raise RuntimeError('No such directory!')
    filedir = filedir or os.getcwd()
    savedir = filedir+str(suffix)
    if not os.path.exists(savedir): os.mkdir(savedir)
    
    for fname in glob.iglob(os.path.join(filedir, 'GSM*.txt')):
        shutil.copy2(fname, os.path.j)
        
def main():
    targdir = input('Specify the directory to run: ')
    transform_files(targdir)
    
if __name__ == '__main__':
    main()
    input('Press Enter to quit...')