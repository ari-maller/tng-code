import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

import plotting_functions as pf

vol = [51.7**3,110.7**3,302.6**3]

def divide_by_zero(a,b):
    ans=np.zeros(a.shape)
    not_zero = b !=0
    ans[not_zero]=a[not_zero] / b[not_zero]
    return ans

def massfuncplot(axis, values, volume, color='gray', range=(5,15), label=None, **kwargs):
    counts, bins = np.histogram(values, bins=50, range=range, **kwargs)
    axis.stairs(np.log10(counts/volume), bins, color=color, baseline=-4,label=label) 

def massfunctions(df50,df100,df300):
    '''plot mass functions for quantities in all 3 boxes'''
    mvir = 'GalpropMvir'
    mass_fields = ['GalpropMstar', 'GalpropMbulge', 'GalpropMBH',
        'GalpropMcold', 'GalpropMH2', 'GalpropMHI']
    mrange=[3.5,10]
    mvir = 'SubhaloMass'
    mass_fields = ['SubhaloMstar', 'SubhaloBHMass','SubhaloMgas', 
                   'SubhaloMH2', 'SubhaloMHI']
    mrange=[7,13]
    f = plt.figure(figsize=[11,5])
    idx=[2,3,4,6,7,8]
    for i,field in enumerate(mass_fields):
        ax = f.add_subplot(2,4,idx[i])
        massfuncplot(ax,np.log10(df50[field]),vol[0], color = 'blue', range=mrange)
        massfuncplot(ax,np.log10(df100[field]),vol[1], color = 'green', range=mrange)
        massfuncplot(ax,np.log10(df300[field]),vol[2], color = 'red', range=mrange)
        ax.annotate(field,(0.4,0.85),xycoords='axes fraction')
        ax.set_ylim([-5.8,-0.1])
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    ax = f.add_axes([0.08,0.4,0.2,0.3])
    massfuncplot(ax,np.log10(df50[mvir]), vol[0], color='blue', 
        range=[7.5,15], label='TNG50')
    massfuncplot(ax,np.log10(df100[mvir]), vol[1], color='green', 
        range=[7.5,15], label='TNG100')
    massfuncplot(ax,np.log10(df300[mvir]), vol[2], color='red', 
        range=[7.5,15], label='TNG300')
    ax.annotate('Mvir',(0.5,0.85),xycoords='axes fraction')
    ax.legend(loc = 'lower left', frameon=False)
    plt.show()

def stellar_to_halo_plot(axis,df,color='gray'):
    s_to_h = 50*df['GalpropMstar']/df['GalpropMvir']
    pf.medianline(axis,np.log10(df['GalpropMvir']),s_to_h, N=50, xrange=(10,15), color=color)

def bulge_to_total_plot(axis,df,color='gray'):
    b_to_t = divide_by_zero(df['GalpropMbulge'],df['GalpropMstar'])
    pf.medianline(axis,np.log10(df['GalpropMstar']),b_to_t, N=50, xrange=(4,13),color=color)

def gas_to_total_plot(axis,df,color='gray'):
    g_to_t = divide_by_zero(df['GalpropMcold'],df['GalpropMcold']+df['GalpropMstar'])
    pf.medianline(axis,np.log10(df['GalpropMstar']),g_to_t, N=50, xrange=(4,13),color=color)

def ratioplot(df50,df100,df300):
    fig,axes = pf.setup_multiplot(1, 2, sharey=True, fs=(8,5), ytitle='Ratio')
#    stellar_to_halo_plot(axes[0],df50,color='blue')
#    stellar_to_halo_plot(axes[0],df100,color='green')
#    stellar_to_halo_plot(axes[0],df300,color='red')
    bulge_to_total_plot(axes[0],df50,color='blue')
    bulge_to_total_plot(axes[0],df100,color='green')
    bulge_to_total_plot(axes[0],df300,color='red')
    gas_to_total_plot(axes[1],df50,color='blue')
    gas_to_total_plot(axes[1],df100,color='green')
    gas_to_total_plot(axes[1],df300,color='red')
    axes[0].annotate('Bulge to Total Ratio',(4,0.9))
    axes[0].set_xlabel('Stellar Mass')
    axes[1].annotate('Gas to Total Baryon Ratio',(4.0,0.1))
    axes[1].set_xlabel('Baryonic Mass')
    plt.show()

def sizeplot(df50,df100,df300):
    fig,axis = plt.subplots()
    pf.medianline(axis,np.log10(df50['GalpropMstar']),np.log10(df50['GalpropRdisk']),
        N=30,xrange=(6,12),color='blue')
    pf.medianline(axis,np.log10(df50['GalpropMstar']),np.log10(df50['GalpropRbulge']), linestyle='dashed',
        N=30,xrange=(6,12),color='blue')
    pf.medianline(axis,np.log10(df100['GalpropMstar']),np.log10(df100['GalpropRdisk']),
        N=30,xrange=(6,12),color='green')
    pf.medianline(axis,np.log10(df100['GalpropMstar']),np.log10(df100['GalpropRbulge']),linestyle='dashed',
        N=30,xrange=(6,12),color='green')
    pf.medianline(axis,np.log10(df300['GalpropMstar']),np.log10(df300['GalpropRdisk']),
        N=30,xrange=(6,13),color='red')
    pf.medianline(axis,np.log10(df300['GalpropMstar']),np.log10(df300['GalpropRbulge']),linestyle='dashed',
        N=30,xrange=(6,13),color='red')
    axis.set_ylim([-1,2])
    axis.set_xlabel('stellar mass')
    axis.set_ylabel('scale radius')
    plt.show()

def joint3():
    sam1 = pd.read_hdf('tng50-sam.h5')
    N = sam1.shape[0]
    sam1['boxsize'] = np.full(N, 50, dtype=int)
    sam2 = pd.read_hdf('tng100-sam.h5')
    N = sam2.shape[0]
    sam2['boxsize'] = np.full(N, 100, dtype=int)
    sam3 = pd.read_hdf('tng300-sam.h5')
    N = sam3.shape[0]
    sam3['boxsize'] = np.full(N, 300, dtype=int)

    df = pd.concat([sam1,sam2,sam3],axis=0)
    sns.jointplot(data= df, x="GalpropMstar", y="GalpropRdisk")
    plt.show()

def frac(values):
    if len(values)==0:
        return 0
    else:
        ans = float(values < 0.02).sum()/len(values)
        print('fraction',type(ans),ans)
    return ans

def fdiskplot(axes, df, color='gray',label=None):
    cents = np.logical_and(df['GalpropSatType']==0,df['GalpropMstar'] > 1.e8)
    logmass = np.log10(df['GalpropMstar'][cents])
    fdisk = (df['GalpropMstar'][cents]+df['GalpropMcold'][cents])/df['GalpropMvir'][cents]
    mask = fdisk < 0.02
    total, edges, _ = stats.binned_statistic(logmass, fdisk, bins=25, range=[8,11.5], statistic='count')
    fdisk_low, edges, _= stats.binned_statistic(logmass, mask, bins=25, range=[8,11.5], statistic='sum')
    axes.stairs(fdisk_low/total, edges=edges, color=color, label=label)

def fdisk(df50,df100,df300):
    f,axes = plt.subplots()
    fdiskplot(axes,df50, color='blue', label='TNG50')
    fdiskplot(axes,df100, color='green', label='TNG100')
    fdiskplot(axes,df300, color='red', label='TNG300')
    axes.set_xlabel('log $M_*$')
    axes.set_ylabel('fraction $f_{disk} < 0.02$')
    axes.legend(loc='upper center')
    plt.show()

def main(xname,yname):
    fnames=['all_tng300-sam.h5','all_tng100-sam.h5','all_tng50-sam.h5']
    c = ['red','green','blue']
    fig,axis=plt.subplots()
    for i,fname in enumerate(fnames):
        df = pd.read_hdf(fname)
        if xname[0:3]=='log':
            x = np.log10(df[xname[3:]])
        else:
            x = df[xname]
        if yname[0:3]=='log':
            y = np.log10(df[yname[3:]])
        else:
            y = df[yname]
        pf.hist2dplot(axis,x,y,range=[[5,12],[3,12]],fill=False, 
            contour_color=c[i],bins=100)
    axis.set_xlim([6,12])
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='plot 3 boxes')
    parser.add_argument('x', nargs='?',
        help = 'the x field to be plotted (adding log will take the log)')
    parser.add_argument('y', nargs='?', 
        help = 'the y field to plot (adding log will take the log')
    parser.add_argument('-mf','--mass_function', action='store_true', default=False, 
        help = 'plot massfunctions')
    parser.add_argument('type',choices=['sim','scsam','lgal'])
    args = parser.parse_args()
    if args.type == 'scsam':
        fname50 = 'all_tng50-sam.h5'
        fname100 = 'all_tng100-sam.h5'
        fname300 = 'all_tng300-sam.h5'
    else:
        fname50 = 'all_tng50-sim.h5'
        fname100 = 'all_tng100-sim.h5'
        fname300 = 'all_tng300-sim.h5'       
    if args.mass_function:
        df50  = pd.read_hdf(fname50)
        df100 = pd.read_hdf(fname100)
        df300 = pd.read_hdf(fname300)
        massfunctions(df50,df100,df300)
#        ratioplot(df50,df100,df300)
#        sizeplot(df50,df100,df300)
#        fdisk(df50,df100,df300)
    else:
        main(args.x,args.y)
