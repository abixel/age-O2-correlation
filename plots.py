import matplotlib
from matplotlib import pyplot as plt
from models import *
import numpy as np
import os
import pickle
from generate import load_result,get_contour,filter_data,ROOT_DIR
from tests import *


outdir = ROOT_DIR+'/output/'

# Pyplot parameters
plt.rcParams['font.size'] = 24
lfs = 32
afs = 16
lw = 5

# Color cycle
cc = ['#bc4244', '#609b5e','#517b9e','#8b5f92','#cc7f33','#d6d65c','#8d5d41','#df99be']

# Figure sizes
size1 = (16,8) # two columns, landscape
size1b = (16,6) # two columns, landscape (shorter)
size1c = (12,8) # two columns, landspace (narrower)
size2 = (8,8) # one column, square
size3 = (16,20) # full page
size4 = (6,4)

# Significance threshold p-value
thresh = 0.05
do_lines = False

def init_outdir(outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

def models_plot(show=False,model_list=[Model2,Model1,Model3],ax=None,fontsize=lfs):
    save = ax is None
    x = np.linspace(0,10,100)
    
    if ax is None:
        fig,ax = plt.subplots(figsize=(1.1*size1c[0],size1c[1]))
    
    for i in range(len(model_list)):
        model = model_list[i]
        y = model(x)
        ax.plot(x,y,label=model_names[model.__name__],lw=lw,c=[cc[1],cc[0],cc[2]][i],zorder=100)
    
    ax.set_xlabel('Age (Gyr)',fontsize=fontsize)
    ax.set_ylabel('Fraction of inhabited\nplanets with O$_2$',fontsize=fontsize)
    
    ax.set_xticks([0,2,4,6,8,10])
    ax.set_xlim([0,10])
    
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_yticklabels(['{:.0f}%'.format(100*yt) for yt in ax.get_yticks()])
    
    ax.legend(loc='lower right')
    
    # Also plot Earth's O2 content vs time from Reinhard et al. 2017
    
    # Gya
    x0 = [4.5,2.4,2.3,2.03,1.93,0.73,0.63,0.]
    y0 = [-5,-5,-1,-1,-3,-3,0,0]
    err = 0.8
    dy0 = [0.05,0.05,err,err,err,err,0.1,0.1]
    x0,y0,dy0 = np.array(x0),np.array(y0),np.array(dy0)
    x0 = 4.5-x0 # Gya to Gy
    
    # Interpolate
    x = np.linspace(x0.min(),x0.max(),1000)
    y = np.interp(x,x0,y0)
    dy = np.interp(x,x0,dy0)
    
    alpha = 0.5
    ax2 = ax.twinx()
    ax2.plot(x,y,lw=10,c='grey',alpha=alpha,zorder=-1000,solid_capstyle='round')
    xmax = 3.8
    
    ax2.set_yticks([-5,-4,-3,-2,-1,0])
    ax2.set_ylim([-5.25,0.25])
    ax2.set_yticklabels(['','$10^{-4}$','','$10^{-2}$','','$10^{0}$'],color='grey',alpha=1)
    ax2.set_ylabel('Historical Earth $p$O$_2$ (PAL)',fontsize=fontsize,color='grey',alpha=1)
    
    ax.set_zorder(10)
    ax.patch.set_visible(False)
    
    plt.subplots_adjust(left=0.22,bottom=0.14,right=0.78)
    if show:
        plt.show()
    elif save:
        init_outdir(outdir)
        plt.savefig(outdir+'/models.pdf')
        plt.close()
    else:
        return ax
        
def contour_plot(models=['Model1','Model2','Model3'],sample='Sample1',test='MannWhitney',show=False,filtered=True,f_false=0.,old=False,xlog=False,ax=None,fontsize=lfs):
    save = ax is None
    if ax is None:
        fig,ax = plt.subplots(figsize=(1.1*size1c[0],size1c[1]))
        ax = [ax]
    else:
        ax = [ax]

    for j in range(len(models)):
        filename = 'results/{:s}_{:s}_{:s}.pkl'.format(sample.replace(' ',''),models[j].replace(' ',''),test)

        if old:
            x,y,z,dz = load_result(filename,old=True)
        else:
            x,f_falses,y,z,dz = load_result(filename)
            
            # Grab the slice for the correct value of f_false
            idx = np.argmin(np.abs(f_false-f_falses))
            z,dz = z[:,idx,:],dz[:,idx,:]
            if np.abs(f_false-f_falses[idx]) > 0.01:
                print("Warning: closest match to f_false = {:.4f} in the results pkl is f_false = {:.4f}".format(f_false,f_falses[idx]))
        
        # Gaussian filter for smoother data
        if filtered: z,dz = filter_data(z,dz)

        xs,ys = get_contour(x,y,z,np.log10(thresh))
        col = cc[j]
        ax[0].plot(xs,ys,c=col,lw=lw)
        
        x0 = [0.45,0.67,0.53]
        y0 = [0.2,0.152,0.34]
        rot = [-27,-23,-23]
        
        ax[0].annotate(model_names[models[j]],xy=(x0[j],y0[j]),xycoords='axes fraction',color=col,ha='center',va='center',
                        rotation=rot[j],zorder=100,weight='bold' if j == 1 else 'normal',fontsize=0.9*plt.rcParams['font.size'])

        if j == 1:
            # Shaded region marking detectable correlation
            ys = ys
            ax[0].fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
            ax[0].annotate('Detectable correlation',xy=(0.5,0.69),xycoords='axes fraction',ha='center',va='center',color='grey',rotation=0)
    
    if xlog:
        ax[0].set_xscale('log')
    else:
        ax[0].set_xticks([0,0.25,0.5,0.75,1.0])
        ax[0].set_xticklabels(['   {:d}%'.format(int(xt*100)) for xt in ax[0].get_xticks()],y=-0.01)
    ax[0].set_xlim([0,1])
    
    ax[0].grid(True,axis='y',which='major',lw=2)
    ax[0].grid(True,axis='y',which='minor',lw=0.3)
    
    ax[0].set_yscale('log')
    ax[0].set_ylim([10,1000])
    ax[0].set_yticks([10,100,1000])
    ax[0].set_yticklabels(['10','100','1000'])
    
    ax[0].axvline(0.1,linestyle='dashed',c='black',lw=2)
    ax[0].axvline(0.8,linestyle='dashed',c='black',lw=2)
    ax[0].annotate('Optimistic',xy=(0.81,950),rotation=90,va='top',ha='left',fontsize=1.25*afs)
    ax[0].annotate('Pessimistic',xy=(0.11,950),rotation=90,va='top',ha='left',fontsize=1.25*afs)
    
        
    # Estimates for LUVOIR, Nautilus, LIFE, Origins
    obs_names = ['LUVOIR','Nautilus Space\nObservatory','LIFE','Origins Space\nTelescope']
    obs_vals = [55.5,986,44.5,26]
    ax2 = ax[0].twinx()
    ax2.set_ylim([1,3])
    ax2.set_yticks(np.log10(obs_vals))
    ax2.set_yticklabels(obs_names,fontsize=1.25*afs)
    
    ax2.tick_params(axis='y',width=4,length=15,pad=15)

    ax[0].set_ylabel('Number of\nexo-Earth candidates',fontsize=fontsize)
    ax[0].set_xlabel('Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',ha='center',va='center',fontsize=fontsize,labelpad=30)
        
    plt.subplots_adjust(left=0.22,bottom=0.14,right=0.78)
    
    if show:
        plt.show()
    elif save:
        init_outdir(outdir)
        plt.savefig(outdir+'/contours.pdf')
        plt.close()
    else:
        return ax[0]    
        
def ratio_plots(models=['Model1','Model2','Model3'],samples=['Sample1','Sample2'],test='MannWhitney',ncols=2,show=False,old=False):
    # Plots the ratio of the number of planets required for p = 0.05 as a function of f_life between two samples
    fig,ax = plt.subplots(1,1,figsize=size1)
    
    # Loop through each model
    for i in range(len(models)):
        filename1 = 'results/{:s}_{:s}_{:s}.pkl'.format(samples[0].replace(' ',''),models[i].replace(' ',''),test)
        filename2 = 'results/{:s}_{:s}_{:s}.pkl'.format(samples[1].replace(' ',''),models[i].replace(' ',''),test)
        x1,f_false,y1,z1,dz1 = load_result(filename1,old=old)
        x2,f_false,y2,z2,dz2 = load_result(filename2,old=old)
        z1,dz1 = z1[:,0,:],dz1[:,0,:]
        z2,dz2 = z2[:,0,:],dz2[:,0,:]
        if not (np.array_equal(x1,x2) and np.array_equal(y1,y2)):
            print("Warning! x/y don't match between samples for {:s}".format(models[i]))

        # Get the contour for each
        try:
            xs1,ys1 = get_contour(x1,y1,z1,np.log10(thresh))
            xs2,ys2 = get_contour(x2,y2,z2,np.log10(thresh))
        except IndexError:
            print("Can't get contour for {:s}".format(models[i]))
            continue
    
        # Interpolate onto a common x grid
        xs = np.linspace(0,1,100)
        ys1 = np.interp(xs,xs1,ys1)
        ys2 = np.interp(xs,xs2,ys2)
        
        # Minimum x value
        mask = xs>0.15
        xs,ys1,ys2 = xs[mask],ys1[mask],ys2[mask]
        
        # Plot the ratio of sample 2 to sample 1
        ax.plot(xs,ys2/ys1,c=cc[i],lw=lw,label=models[i]+' ({:.2f})'.format(np.median(ys2/ys1)))
        ax.axhline(np.median(ys2/ys1),c=cc[i])
    
    ax.set_title('{:s} / {:s}'.format(sample_names[samples[1]],sample_names[samples[0]]),fontsize=fontsize)
    ax.set_xlim([0,1])
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['{:d}%'.format(int(xt*100)) for xt in ax.get_xticks()])
    
    ax.legend(loc='best')
    
    plt.show()

def false_positive_plot(model='Model2',sample='Sample1',test='MannWhitney',show=False,filtered=True,old=False,filename=None,ax=None):
    save = ax is None
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=size1c)

    if filename is None:
        filename = 'results/{:s}_{:s}_{:s}.pkl'.format(sample.replace(' ',''),model.replace(' ',''),test)
    x,f_false,y,z,dz = load_result(filename)
    
    # Gaussian filter for smoother data
    if filtered: z,dz = filter_data(z,dz)
        
    ls = ['solid','dashed','dotted']
    for i in range(len(f_false)):
        xs,ys = get_contour(x,y,z[:,i,:],np.log10(thresh))
        ax.plot(xs,ys,c=cc[1],lw=lw,ls=ls[i])
        idx = np.argmin(np.abs(xs-0.3))
        x0 = [0.45,0.5,0.6]
        y0 = [0.25,0.41,0.55]
        rots = [-25,-26,-30]
        label = 'No abiotic O$_2$' if f_false[i] == 0. else '{:.0f}% have abiotic O$_2$'.format(f_false[i]*100)
        ax.annotate(label,xy=(x0[i],y0[i]),xycoords='axes fraction',color=cc[1],ha='center',va='center',rotation=rots[i])
    
        if i == 0:
            # Shaded region marking detectable correlation
            ys = ys
            ax.fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
    
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['{:d}%'.format(int(xt*100)) for xt in ax.get_xticks()],y=-0.01)
    ax.set_xlim([0,1])
    
    ax.grid(True,axis='y',which='major',lw=2)
    ax.grid(True,axis='y',which='minor',lw=0.3)
    
    ax.set_yscale('log')
    ax.set_ylim([10,1000])
    ax.set_yticks([10,100,1000])
    ax.set_yticklabels(['10','100','1000'])
    
    if do_lines:
        ax.axvline(0.1,linestyle='dashed',c='black',lw=2)
        ax.axvline(0.8,linestyle='dashed',c='black',lw=2)
    
    ax.set_ylabel('Number of exo-Earth candidates',fontsize=lfs)
    ax.set_xlabel('Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',ha='center',va='center',fontsize=lfs,labelpad=30)
     
    plt.subplots_adjust(left=0.13,bottom=0.14,right=0.88)
    
    if show:
        plt.show()
    elif save:
        init_outdir(outdir)
        plt.savefig(outdir+'/false_positives.pdf')
        plt.close()
    else:
        return ax

def p_values_plot(model='Model2',sample='Sample1',test='MannWhitney',show=False,filtered=True,old=False,filename=None,ax=None):
    save = ax is None
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=size1c)

    if filename is None:
        filename = 'results/{:s}_{:s}_{:s}.pkl'.format(sample.replace(' ',''),model.replace(' ',''),test)
    x,f_false,y,z,dz = load_result(filename)
    z,dz = z[:,0,:],dz[:,0,:]
    
    # Gaussian filter for smoother data
    if filtered: z,dz = filter_data(z,dz)

    ls = ['solid','dashed','dotted']
    levels = [0.05,0.1,0.01]
    #lws = [lw,1.5*lw,lw/1.5]
    lws = [lw,lw,lw]
    for i in range(len(levels)):
        xs,ys = get_contour(x,y,z,np.log10(levels[i]))
        ax.plot(xs,ys,c=cc[1],lw=lws[i],ls=ls[i])
        idx = np.argmin(np.abs(xs-0.5))
        x0s = [0.475,0.44,0.5]
        shifts = [14,-3,25]
        rots = [-25,-23,-25]
        ax.annotate('p = {:.2f}'.format(levels[i]),xy=(x0s[i],ys[idx]+shifts[i]),color=cc[1],ha='center',va='center',rotation=rots[i])
    
        
        if i == 0:
            # Shaded region marking detectable correlation
            ys = ys
            ax.fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
    
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['{:d}%'.format(int(xt*100)) for xt in ax.get_xticks()],y=-0.01)
    ax.set_xlim([0,1])
    
    ax.grid(True,axis='y',which='major',lw=2)
    ax.grid(True,axis='y',which='minor',lw=0.3)
    
    ax.set_yscale('log')
    ax.set_ylim([10,1000])
    ax.set_yticks([10,100,1000])
    ax.set_yticklabels(['10','100','1000'])
    if do_lines:
        ax.axvline(0.1,linestyle='dashed',c='black',lw=2)
        ax.axvline(0.8,linestyle='dashed',c='black',lw=2)
    
    ax.set_ylabel('Number of exo-Earth candidates',fontsize=lfs)
    ax.set_xlabel('Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',ha='center',va='center',fontsize=lfs,labelpad=30)
     
    plt.subplots_adjust(left=0.13,bottom=0.14,right=0.88)
    
    if show:
        plt.show()
    elif save:
        init_outdir(outdir)
        plt.savefig(outdir+'/p_values.pdf')
        plt.close()
    else:
        return ax

def test_comparison_plot(model='Model2',sample='Sample1',tests=['MannWhitney','Spearman','Student'],show=False,filtered=True,ax=None):
    save = ax is None
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=size1c)

    for i in range(len(tests)):
        filename = 'results/{:s}_{:s}_{:s}.pkl'.format(sample.replace(' ',''),model.replace(' ',''),tests[i])
        x,f_false,y,z,dz = load_result(filename)
        z,dz = z[:,0,:],dz[:,0,:]
        
        # Gaussian filter for smoother data
        if filtered: z,dz = filter_data(z,dz)

        ls = ['solid','dashed','dotted','dashdot']
        label = test_names[tests[i]]
    
        xs,ys = get_contour(x,y,z,np.log10(thresh))
        ax.plot(xs+(0.005 if i == 2 else 0),ys,c=cc[1],lw=lw,ls=ls[i])
        idx = np.argmin(np.abs(xs-0.5))
        
        
        if i == 0:
            # Shaded region marking detectable correlation
            ys = ys
            ax.fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
        
    for i in range(2):
        x0 = [0.45,0.522]
        y0 = [0.25,0.423]
        rots = [-22.5,-25.5]
        labels = ['Mann-Whitney U test',"Spearman's / Student's tests"]
        ax.annotate(labels[i],xy=(x0[i],y0[i]),color=cc[1],xycoords='axes fraction',ha='center',va='center',rotation=rots[i])
    
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['{:d}%'.format(int(xt*100)) for xt in ax.get_xticks()],y=-0.01)
    ax.set_xlim([0,1])
    
    ax.grid(True,axis='y',which='major',lw=2)
    ax.grid(True,axis='y',which='minor',lw=0.3)
    
    ax.set_yscale('log')
    ax.set_ylim([10,1000])
    ax.set_yticks([10,100,1000])
    ax.set_yticklabels(['10','100','1000'])
    
    if do_lines:
        ax.axvline(0.1,linestyle='dashed',c='black',lw=2)
        ax.axvline(0.8,linestyle='dashed',c='black',lw=2)
    
    ax.set_ylabel('Number of exo-Earth candidates',fontsize=lfs)
    ax.set_xlabel('Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',ha='center',va='center',fontsize=lfs,labelpad=30)   

    plt.subplots_adjust(left=0.13,bottom=0.14,right=0.88)
    
    if show:
        plt.show()
    elif save:
        init_outdir(outdir)
        plt.savefig(outdir+'/test_comparison.pdf')
        plt.close()
    else:
        return ax

def sample_selection_plot(model='Model2',samples=['Sample1','Sample2','Sample4'],test='MannWhitney',show=False,filtered=True,old=False,ax=None):
    save = ax is None
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=size1c)

    for i in range(len(samples)):
        filename = 'results/{:s}_{:s}_{:s}.pkl'.format(samples[i].replace(' ',''),model.replace(' ',''),test)
        x,f_false,y,z,dz = load_result(filename)
        z,dz = z[:,0,:],dz[:,0,:]
        
        # Gaussian filter for smoother data
        if filtered: z,dz = filter_data(z,dz)

        ls = ['solid','dashed','dotted']
        labels = ['All planets','Young and old planets only','No young planets']
    
        xs,ys = get_contour(x,y,z,np.log10(thresh))
        ax.plot(xs,ys,c=cc[1],lw=lw,ls=ls[i])
        idx = np.argmin(np.abs(xs-0.45))
        shifts = [11,-1.75,5]
        xshifts = [0,-.085,.08]
        rots = [-25,-24,-25]
        ax.annotate(labels[i],xy=(0.45+xshifts[i],ys[idx]+shifts[i]),color=cc[1],ha='center',va='center',rotation=rots[i])
    
    
        if i == 0:
            # Shaded region marking detectable correlation
            ys = ys
            ax.fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
            ax.annotate('Detectable correlation',xy=(0.95,0.92),xycoords='axes fraction',ha='right',va='center',color='grey',rotation=0,weight='bold',
                        bbox=dict(boxstyle='round',fc='grey',lw=0,alpha=0.05))
    
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['{:d}%'.format(int(xt*100)) for xt in ax.get_xticks()],y=-0.01)
    ax.set_xlim([0,1])
    
    ax.grid(True,axis='y',which='major',lw=2)
    ax.grid(True,axis='y',which='minor',lw=0.3)
    
    ax.set_yscale('log')
    ax.set_ylim([10,1000])
    ax.set_yticks([10,100,1000])
    ax.set_yticklabels(['10','100','1000'])
    
    if do_lines:
        ax.axvline(0.1,linestyle='dashed',c='black',lw=2)
        ax.axvline(0.8,linestyle='dashed',c='black',lw=2)
    
    ax.set_ylabel('Number of exo-Earth candidates',fontsize=lfs)
    ax.set_xlabel('Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',ha='center',va='center',fontsize=lfs,labelpad=30)
     
    plt.subplots_adjust(left=0.13,bottom=0.14,right=0.88)
    
    if show:
        plt.show()
    elif save:
        init_outdir(outdir)
        plt.savefig(outdir+'/sample_selection.pdf')
        plt.close()
    else:
        return ax
    

def top_legend(ax,ncol=2,frameon=True,loc=3,bbox_to_anchor=(0., 1.02, 1., .102),fontsize=lfs):
    return ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc,
                   ncol=ncol, mode="expand", borderaxespad=0.,numpoints=1,frameon=frameon,fontsize=fontsize)

def alt_cases(show=False,filtered=True):
    y=1.05
    fs = lfs
    
    # Combines four of the plots above into one
    fig,ax = plt.subplots(2,2,figsize=(2*size1c[0],2*size1c[1]))
    ax = ax.flatten()
    
    # Top left: age selection plot
    ax[0] = sample_selection_plot(filtered=filtered,ax=ax[0])
    
    # Top right: abiotic O2 plot
    ax[1] = false_positive_plot(filtered=filtered,ax=ax[1],filename='results/false_positives.pkl')
    
    # Bottom right: statistical test comparison
    ax[2] = test_comparison_plot(filtered=filtered,ax=ax[2])
    
    # Bottom left: p-value contours
    ax[3] = p_values_plot(filtered=filtered,ax=ax[3])
    
    # Titles
    for i in range(len(ax)):
        titles = ['Modifying the target age distribution:','Including planets with abiotic oxygen:','Using different statistical tests:','Enforcing higher/lower confidence thresholds:']
        chars = ['(a)','(b)','(c)','(d)']
        xx,yy = -0.165,1.11
        ax[i].annotate(chars[i],xy=(xx,yy),xycoords='axes fraction',fontsize=lfs,va='center',ha='left',annotation_clip=False,weight='bold')
        ax[i].annotate(titles[i],xy=(xx+0.115,yy),xycoords='axes fraction',fontsize=lfs,va='center',ha='left',annotation_clip=False)

    # Top legend
    ax[0].plot([-2,-1],[-100,-200],c=cc[1],lw=lw,label='Baseline case')
    fig.legend(loc='upper center',frameon=False,fontsize=lfs)
    
    # Strip all axis labels
    for axx in ax:
        axx.set_xlabel('')
        axx.set_ylabel('')
    
    fig.text(0.5,0.05,'Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',va='center',ha='center',fontsize=1.5*lfs)
    fig.text(0.05,0.5,'Number of exo-Earth candidates',va='center',ha='center',rotation=90,fontsize=1.5*lfs)
    
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    
    if show:
        plt.show()
    else:
        plt.savefig(outdir+'/alt_cases.pdf')
        plt.close()
    
    
if __name__ == "__main__":
    models_plot()
    contour_plot()
    alt_cases()
