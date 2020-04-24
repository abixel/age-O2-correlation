import matplotlib
from matplotlib import pyplot as plt
from models import *
import numpy as np
import pickle
from utils import load_result,get_contour,filter_data
from tests import *

outdir = '/home/abixel/Dropbox/scripts/o2/output/'

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


def models_plot(show=False,model_list=[Model2,Model1,Model3],ax=None,prop_plot=False,fontsize=lfs):
    save = ax is None
    x = np.linspace(0,10,100)
    
    #fig,ax = plt.subplots(figsize=size1)
    if ax is None:
        fig,ax = plt.subplots(figsize=(1.1*size1c[0],size1c[1]))
    print(ax)
    
    for i in range(len(model_list)):
        model = model_list[i]
        y = model(x)
        ax.plot(x,y,label=model_names[model.__name__],lw=lw,c=[cc[1],cc[0],cc[2]][i],zorder=100)
    
    #ax.set_xlabel('Time for which planet\nhas been habitable $t_H$ (Gyr)',fontsize=fontsize)
    ax.set_xlabel('Age (Gyr)',fontsize=fontsize)
    #ax.set_ylabel(r'$P_{\mathregular{GOE}}$',fontsize=fontsize)
    ax.set_ylabel('Fraction of inhabited\nplanets with O$_2$',fontsize=fontsize)
    
    ax.set_xticks([0,2,4,6,8,10])
    ax.set_xlim([0,10])
    
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_yticklabels(['{:.0f}%'.format(100*yt) for yt in ax.get_yticks()])
    
    if not prop_plot:
        ax.legend(loc='lower right')
    
    # Also plot Earth's O2 content vs time from Reinhard et al. 2017 (original source unknown)
    
    #x0,y0 = np.loadtxt('Earth_O2.dat',unpack=True)
    
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
    
    from scipy.ndimage import gaussian_filter1d
    line1,line2 = y-dy,y+dy
    line2[x<2.15] = -8
    
    #line1 = gaussian_filter1d(line1,len(line1)/300)
    #line2 = gaussian_filter1d(line2,len(line2)/300)
    
    if not prop_plot:
        alpha = 0.5
        ax2 = ax.twinx()
        ax2.plot(x,y,lw=10,c='grey',alpha=alpha,zorder=-1000,solid_capstyle='round')
        #ax2.errorbar([2.335,3.17],[-1,-3],yerr=[-1,-1],c='grey',marker='None',alpha=alpha,linestyle='None',elinewidth=5,capsize=10,capthick=5)
        xmax = 3.8
        #ax2.fill_between(x[x<xmax],line1[x<xmax],line2[x<xmax],color='grey',alpha=0.2,zorder=-1000)
        #ax2.plot(x,line1,c='grey',alpha=0.7,linestyle='dashed',lw=7,zorder=-100)
        #ax2.plot(x-0.033,line2,c='grey',alpha=0.7,linestyle='dashed',lw=7,zorder=-100)
        #ax2.errorbar([2.3,3.1],[-1,-3],yerr=[err,err],c='grey',lw=5,alpha=alpha,linestyle='None',capsize=5,marker='None')
        #ax2.plot([x0[2],x0[3]],[-1-err,-1-err],linestyle='dashed',c='grey',alpha=alpha,lw=lw)
        #ax2.plot([x0[2],x0[3]],[-1+err,-1+err],linestyle='dashed',c='grey',alpha=alpha,lw=lw)
        #ax2.plot([x0[4],x0[5]],[-3-err,-3-err],linestyle='dashed',c='grey',alpha=alpha,lw=lw)
        #ax2.plot([x0[4],x0[5]],[-3+err,-3+err],linestyle='dashed',c='grey',alpha=alpha,lw=lw)
        
        ax2.set_yticks([-5,-4,-3,-2,-1,0])
        ax2.set_ylim([-5.25,0.25])
        ax2.set_yticklabels(['','$10^{-4}$','','$10^{-2}$','','$10^{0}$'],color='grey',alpha=1)
        ax2.set_ylabel('Historical Earth $p$O$_2$ (PAL)',fontsize=fontsize,color='grey',alpha=1)
    else:
        ax.axvline(2.3,lw=5,c='grey',zorder=-10,alpha=0.8,linestyle='dashed')
        ax.annotate('Great\nOxidation\nEvent',xy=(2.5,0.15),ha='left',va='center',fontsize=afs,color='grey')
        ax.annotate('Possible correlation',xy=(7.5,0.78),ha='center',va='center',rotation=9,color=cc[1],fontsize=afs)
    
    #plt.subplots_adjust(bottom=0.12)
    ax.set_zorder(10)
    ax.patch.set_visible(False)
    #plt.subplots_adjust(left=0.18,bottom=0.14,right=0.82)
    
    if not prop_plot:
        plt.subplots_adjust(left=0.22,bottom=0.14,right=0.78)
    if show:
        plt.show()
    elif save:
        plt.savefig(outdir+'/models.pdf')
        plt.close()
    else:
        return ax
        
def contour_plots(models=['Model1','Model2','Model3'],sample='Sample1',test='MannWhitney',show=False,filtered=True,f_false=0.,old=False,xlog=False,ax=None,prop_plot=False,fontsize=lfs):
    #nrows = int(len(filenames)/ncols) + int(len(filenames)%ncols > 0)
    save = ax is None
    if ax is None:
        fig,ax = plt.subplots(figsize=(1.1*size1c[0],size1c[1]))
        #ax = ax.flatten()
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
        col = cc[1] if prop_plot else cc[j]
        ax[0].plot(xs,ys,c=col,lw=lw)
        
        x0 = [0.45,0.67,0.53]
        y0 = [0.2,0.152,0.34]
        rot = [-27,-23,-23]
        
        if not prop_plot:
            ax[0].annotate(model_names[models[j]],xy=(x0[j],y0[j]),xycoords='axes fraction',color=col,ha='center',va='center',
                            rotation=rot[j],zorder=100,weight='bold' if j == 1 else 'normal',fontsize=0.9*plt.rcParams['font.size'])

        if j == 1 or prop_plot:
            # Shaded region marking detectable correlation
            ys = ys
            ax[0].fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
            if prop_plot:
                ax[0].annotate('Detectable correlation',xy=(0.65,0.57),xycoords='axes fraction',ha='center',va='center',color='grey',rotation=0,fontsize=afs)
            else:
                ax[0].annotate('Detectable correlation',xy=(0.5,0.69),xycoords='axes fraction',ha='center',va='center',color='grey',rotation=0)

    # Ranges for LUVOIR, Nautilus
    """
    xline,xann = 1.02,1.03
    ax[0].plot([xline,xline],[20,115],c='black',lw=4,clip_on=False)
    ax[0].annotate('LUVOIR',xy=(xann,53),va='center',ha='left',rotation=90,annotation_clip=False)
    ax[0].plot([xline,xline],[300,1000],c='black',lw=4,clip_on=False)
    ax[0].annotate('Nautilus Space',xy=(xann,580),va='center',ha='left',rotation=90,annotation_clip=False)
   # ax[0].annotate('Space',xy=(xann+.04,580),va='center',ha='left',rotation=90,annotation_clip=False)
    ax[0].annotate('Observatory',xy=(xann+.04,580),va='center',ha='left',rotation=90,annotation_clip=False)
    """
    
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
    
    #if prop_plot:
        #ax[0].yaxis.tick_right()
        #ax[0].yaxis.set_label_position("right")
    
    if not prop_plot:
        ax[0].axvline(0.1,linestyle='dashed',c='black',lw=2)
        ax[0].axvline(0.8,linestyle='dashed',c='black',lw=2)
        ax[0].annotate('Optimistic',xy=(0.81,950),rotation=90,va='top',ha='left',fontsize=1.25*afs)
        ax[0].annotate('Pessimistic',xy=(0.11,950),rotation=90,va='top',ha='left',fontsize=1.25*afs)
    
    #sample_titles = ['All ages (0-8 Gyr)','Young/old systems only']
    #ax[0].set_title(sample_names[samples[i]],fontsize=fontsize)
    
        
    # Estimates for LUVOIR, Nautilus, LIFE, Origins
    if not prop_plot:
        obs_names = ['LUVOIR','Nautilus Space\nObservatory','LIFE','Origins Space\nTelescope']
        obs_vals = [55.5,986,44.5,26]
        ax2 = ax[0].twinx()
        #ax2.set_yscale('log')
        ax2.set_ylim([1,3])
        ax2.set_yticks(np.log10(obs_vals))
        ax2.set_yticklabels(obs_names,fontsize=1.25*afs)
        
        ax2.tick_params(axis='y',width=4,length=15,pad=15)
    
    """
    for i in range(len(obs_names)):
        opt = dict(color='black', arrowstyle = 'simple,head_width=.35,head_length=.35',connectionstyle = 'arc3,rad=0')
        v = obs_vals[i]
        ax[0].annotate(obs_names[i],xy=(1.01,v*1.1),annotation_clip=False,fontsize=1.25*afs)
        #plt.arrow(1.05,v,-0.04,0.,clip_on=False,length_includes_head=True,head_length=0.05,head_width=v)
    """
    
    if prop_plot:
        ax[0].set_ylabel('Number of exo-Earth\ncandidates observed',fontsize=fontsize)
        ax[0].set_xlabel('Fraction of exo-Earth candidates with life',ha='center',va='center',fontsize=fontsize,labelpad=15)
    else:
        ax[0].set_ylabel('Number of\nexo-Earth candidates',fontsize=fontsize)
        ax[0].set_xlabel('Fraction of exo-Earth candidates with life ($f_\mathrm{life}$)',ha='center',va='center',fontsize=fontsize,labelpad=30)
        
    
    if not prop_plot:
        plt.subplots_adjust(left=0.22,bottom=0.14,right=0.78)
    #plt.subplots_adjust(bottom=0.15,wspace=0.25,left=0.08,right=0.92)
    
    if show:
        plt.show()
    elif save:
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
    
def false_positive_ratio_plot(filename='results/false_positives.pkl',show=False,old=True):
    fig,ax = plt.subplots(1,1,figsize=size1)
    
    x,f_false,y,z,dz = load_result(filename)
    
    # Get the contour for each false positive value
    ls = ['solid','dashed','dotted']
    for j in range(1,len(f_false)):
        xs1,ys1 = get_contour(x,y,z[:,0,:],np.log10(thresh))
        xs2,ys2 = get_contour(x,y,z[:,j,:],np.log10(thresh))

        # Interpolate onto a common x grid
        xs = np.linspace(0,1,100)
        ys1 = np.interp(xs,xs1,ys1)
        ys2 = np.interp(xs,xs2,ys2)
    
        # Minimum x value
        mask = xs>0.05
        xs,ys1,ys2 = xs[mask],ys1[mask],ys2[mask]
        
        # Plot the ratio of sample 2 to sample 1
        ax.plot(xs,ys2/ys1,c=cc[1],lw=lw,ls=ls[j])

    
        if j == 0:
            # Shaded region marking detectable correlation
            ys = ys
            ax.fill_between(xs,ys,ys+1e4,color='grey',alpha=0.15,lw=0)
    
    ax.set_xlim([0,1])
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['{:d}%'.format(int(xt*100)) for xt in ax.get_xticks()])
    
    ax.set_xlabel("Fraction of potentially habitable planets with life",fontsize=lfs)
    ax.set_ylabel("Multiple on\nrequired sample size",fontsize=lfs)
    
    plt.subplots_adjust(left=0.11,bottom=0.12)
    
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
    #plt.subplots_adjust(bottom=0.15,wspace=0.25,left=0.08,right=0.92)
    
    if show:
        plt.show()
    elif save:
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
    #plt.subplots_adjust(bottom=0.15,wspace=0.25,left=0.08,right=0.92)
    
    if show:
        plt.show()
    elif save:
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
    #ax.legend(loc='upper right',ncol=2)
    #plt.subplots_adjust(bottom=0.15,wspace=0.25,left=0.08,right=0.92)
    
    if show:
        plt.show()
    elif save:
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
    #plt.subplots_adjust(bottom=0.15,wspace=0.25,left=0.08,right=0.92)
    
    if show:
        plt.show()
    elif save:
        plt.savefig(outdir+'/sample_selection.pdf')
        plt.close()
    else:
        return ax
    

def top_legend(ax,ncol=2,frameon=True,loc=3,bbox_to_anchor=(0., 1.02, 1., .102),fontsize=lfs):
    return ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc,
                   ncol=ncol, mode="expand", borderaxespad=0.,numpoints=1,frameon=frameon,fontsize=fontsize)

def alt_cases(show=False,filtered=True):
    
    # some finagling
    #plt.rcParams['font.size'] = 1.5 * plt.rcParams['font.size']
    
    y=1.05
    fs = lfs
    
    # Combines four of the plots above into one
    fig,ax = plt.subplots(2,2,figsize=(2*size1c[0],2*size1c[1]))
    ax = ax.flatten()
    
    # Top left: age selection plot
    ax[0] = sample_selection_plot(filtered=filtered,ax=ax[0])
    #ax[0].set_title('Modifying the target age distribution:',fontsize=fs,y=y)
    
    # Top right: abiotic O2 plot
    ax[1] = false_positive_plot(filtered=filtered,ax=ax[1],filename='results/false_positives.pkl')
    #ax[1].set_title('Including planets with abiotic oxygen:',fontsize=fs,y=y)
    
    # Bottom right: statistical test comparison
    ax[2] = test_comparison_plot(filtered=filtered,ax=ax[2])
    #ax[2].set_title('Using different statistical tests:',fontsize=fs,y=y)
    
    # Bottom left: p-value contours
    ax[3] = p_values_plot(filtered=filtered,ax=ax[3])
    #ax[3].set_title('Enforcing higher/lower confidence thresholds:',fontsize=fs,y=y)
    
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
    
    #fig.suptitle('Required sample sizes for alternative cases',fontsize=lfs)
    
    # Add some annotations
    #for i in range(len(ax)):
    #    label = ['(a)','(b)','(c)','(d)'][i]
    #    ax[i].annotate(label,xy=(-0.15,1.1),xycoords='axes fraction',fontsize=lfs,va='center',ha='center',annotation_clip=False)
    
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    
    #plt.rcParams['font.size'] = plt.rcParams['font.size'] / 1.5
    
    if show:
        plt.show()
    else:
        plt.savefig(outdir+'/alt_cases.pdf')
        plt.close()
    

def AlienEarths(show=False,filtered=True):
    fig,ax = plt.subplots(1,2,figsize=(8.5*2,2*2))
    fs = 0.75*lfs
    ax[0] = models_plot(ax=ax[0],model_list=[Model2],prop_plot=True,fontsize=fs)
    ax[1] = contour_plots(ax=ax[1],models=['Model2'],prop_plot=True,fontsize=fs)
    
    plt.subplots_adjust(bottom=0.25,wspace=0.5)
    if show:
        plt.show()
    else:
        plt.savefig('output/O2_correlation.pdf')

def AAS(show=False,filtered=True):
    fig,ax = plt.subplots(1,1,figsize=(8.5,2*2))
    fs = 0.75*lfs
    ax = models_plot(ax=ax,model_list=[Model2],prop_plot=True,fontsize=fs)
    plt.subplots_adjust(bottom=0.25,left=0.25)
    if show:
        plt.show()
    else:
        plt.savefig('output/AAS.png',dpi=300)
    
    
    
    
    
    
    
    
    
    
