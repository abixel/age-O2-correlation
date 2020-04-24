from matplotlib import pyplot as plt
import multiprocessing as mp
from models import *
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr,mannwhitneyu,ttest_ind
from skimage.measure import find_contours
import tests
from tqdm import tqdm as bar
from types import FunctionType

def draw_sample(f_life,f_false,N,sample,model,sample_args=(),model_args=()):
    if f_life>1:
        print("Error: f_life must be <= 1")
        return None
    if f_false>1:
        print("Error: f_false must be <= 1")
        return None
    
    # Draw the planet ages from the specified sample function and start with O2 = false for all
    ages = sample(N,*sample_args)
    O2 = np.zeros(N,dtype=bool)
    
    # Determine which planets have life
    life = np.random.uniform(0,1,N)<f_life
    
    # Assign O2 to planets with life based on the specified model function
    P = model(ages[life],*model_args)
    O2[life] = np.random.uniform(0,1,life.sum())<P
    
    # Also determine which planets have abiotic O2
    if f_false > 0:
        O2 = O2 | (np.random.uniform(0,1,N)<f_false)
        
    return ages,O2
 
def run_grid(f_lifes,f_falses,Ns,sample,model,sample_args=(),model_args=(),test=tests.MannWhitney,N_runs=500,log=True,filename=None,do_bar=True,N_proc=1):
    # Generates the p value grid for the specific model, sample, and statistical test
    p,sig_p,nans = np.zeros((len(f_lifes),len(f_falses),len(Ns)),dtype=float),np.zeros((len(f_lifes),len(f_falses),len(Ns)),dtype=float),np.zeros((len(f_lifes),len(f_falses),len(Ns)),dtype=int)
    if do_bar:
        bar0 = bar
    else:
        def bar0(arg):
            return arg
           
    # Calculates and saves the typical p-value and std deviation for a single combination of f_life, f_false, N
    def run_bin(i,j,k,p,sig_p,nans):
        p0 = np.zeros(N_runs,dtype=float)
        for ii in range(N_runs):
            ages,O2 = draw_sample(f_lifes[i],f_falses[j],int(Ns[k]),sample,model,sample_args=sample_args,model_args=model_args)
            if O2.sum() > 0 and (~O2).sum() > 0:
                p0[ii] = test(ages,O2,p_only=True)
            else:
                p0[ii] = np.nan
            
        # Get the mean and std deviation for this bin and count the number of nans
        if (~np.isnan(p0)).sum() > 0:
            if log:
                p[i,j,k],sig_p[i,j,k] = np.nanmean(np.log10(p0)),np.nanstd(np.log10(p0))
            else:
                p[i,j,k],sig_p[i,j,k] = np.nanmean(p0),np.nanstd(p0)
        else:
            p[i,j,k],sig_p[i,j,k] = np.nan,np.nan
        
        nans[i,j,k] = np.isnan(p0).sum()
    
    # Get every combination of f_life,f_false,N
    combos = []
    for i in range(len(f_lifes)):
        for j in range(len(f_falses)):
            for k in range(len(Ns)):
                combos.append((i,j,k))

    # (Single process) Run each combo sequentially
    if N_proc == 1:
        for combo in bar(combos): run_bin(*combo,p,sig_p,nans)
    
    # (Multi process) Split up the combos and run in parallel
    else:
        print("N_proc > 1 currently isn't working, so use N_proc = 1")
        return 
        processes = [mp.Process(target=run_bin,args=((*combo,p,sig_p))) for combo in combos]
        for i1 in range(0,len(processes),N_proc):
            # Run the next set of processes
            for i2 in range(i1,i1+N_proc):
                processes[i2].start()
                 
            # Wait for the processes to complete
            for i2 in range(i1,i1+N_proc):
                processes[i2].join()

    # Save the results to a specified file
    if filename is not None:
        try:
            pkl = {}
            pkl['f_life'],pkl['f_false'],pkl['N'],pkl['nans'] = np.array(f_lifes),np.array(f_falses),np.array(Ns),np.array(nans)
            if log:
                pkl['logp'],pkl['sig_logp'] = p,sig_p
            else:
                pkl['p'],pkl['sig_p'] = p,sig_p
            pickle.dump(pkl,open(filename,'wb'))
            print("Saved {:s}".format(filename))
        except:
            print("Error saving file: {:s}".format(filename))
    
    return p,sig_p,nans

def run_grids(f_lifes,f_falses,Ns,samples,models,tests=[tests.MannWhitney],test=None,N_runs=1000,log=True,save=True,N_proc=1):
    # Does `run_grid` for every combination of `samples` and `models` and `tests`
    # and saves to results/<sample>_<model>_<test>.pkl
    if type(samples) is FunctionType: samples = [samples]
    if type(models) is FunctionType: models = [models]
    if test is not None: tests=[test]
    if type(tests) is FunctionType: tests = [tests]
    
    # Warn the user if they aren't saving the results
    if not save:
        for i in range(3): print("Not saving the results!!!")
    
    # Enables multi-processing
    def run(sample,model,test,do_bar):
        filename = 'results/{:s}_{:s}_{:s}.pkl'.format(sample.__name__,model.__name__,test.__name__)
        p,sig_p,nans = run_grid(f_lifes,f_falses,Ns,sample,model,log=log,N_runs=N_runs,test=test,do_bar=do_bar,
                           filename=filename if save else None)
        
    # Get every combination of samples, models, and tests
    combos = []
    for i in range(len(samples)):
        for j in range(len(models)):
            for k in range(len(tests)):
                combos.append((samples[i],models[j],tests[k],not len(combos)%N_proc))
    
    # If single-processing, just loop through the combinations one-by-one
    if N_proc == 1:
        for combo in combos: run(*combo)
    
    # If multi-processing, go through the combos N_proc at a time until done
    else:
        processes = [mp.Process(target=run,args=combo) for combo in combos]
        for i1 in range(0,len(processes),N_proc):
            # Run the next set of processes
            for i2 in range(i1,i1+N_proc):
                processes[i2].start()
                 
            # Wait for the processes to complete
            for i2 in range(i1,i1+N_proc):
                processes[i2].join()

def load_result(filename,old=False):
    pkl = pickle.load(open(filename,'rb'))
    if 'p' in pkl.keys():
        if old:
            return pkl['f'],pkl['N'],pkl['p'],pkl['sig_p']
        else:
            return pkl['f_life'],pkl['f_false'],pkl['N'],pkl['p'],pkl['sig_p']
    else:
        if old:
            return pkl['f'],pkl['N'],pkl['logp'],pkl['sig_logp']
        else:
            return pkl['f_life'],pkl['f_false'],pkl['N'],pkl['logp'],pkl['sig_logp']

def get_contour(x,y,z,val):
    # Returns the x,y coordinates for a contour line along z = val
    yidx,xidx = find_contours(np.swapaxes(z,0,1),val)[0].T
    xs = np.interp(xidx,range(len(x)),x)
    ys = np.interp(yidx,range(len(y)),y)
    
    # Sort by x
    idx = np.argsort(xs)
    return xs[idx],ys[idx]

def filter_data(z,dz,val=None,mode='reflect'):
    if val is None: val = min([z.shape[0],z.shape[1]])/100
    return gaussian_filter(z,val,mode=mode),gaussian_filter(dz,val,mode=mode)

    
    
    
    
    
    
