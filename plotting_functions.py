import numpy as np
import matplotlib.pyplot as plt

def setup_multiplot(Nr,Nc,xtitle=None,ytitle=None,ytitle2=None,fs=(15,5),**kwargs):
    '''return the axes for a multiplot with x and y titles'''
    f,axs=plt.subplots(Nr,Nc,figsize=fs,**kwargs)
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    ax=f.add_subplot(111,frameon=False)
    ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(xtitle,fontsize='x-large')
    ax.set_ylabel(ytitle,fontsize='x-large',labelpad=20)
    if ytitle2:
        ax2=ax.twinx()
        ax2.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False) 
        ax2.set_ylabel(ytitle2,fontsize='x-large',labelpad=20)   
    return f,axs

def hist2dplot(axis,x,y,fill=True,contour_color='green',**kwargs):
    h,xed,yed=np.histogram2d(x,y,**kwargs)
    h=np.transpose(h)
    total=h.sum()
    h=h/total
    hflat=np.sort(np.reshape(h,-1)) #makes 1D and sorted 
    csum=np.cumsum(hflat)
    values=1.0-np.array([0.9973,0.9545,0.6827,0.0])
    levels=[]
    col_dict={'blue':['#deebf7','#9ecae1','#3182bd'],
            'green':['#e5f5e0','#a1d99b','#31a354'],
            'gray':['#f0f0f0','#bdbdbd','#636363'],
            'orange':['#fee6ce','#fdae6b', '#e6550d'],
            'purple':['#efedf5','#bcbddc','#756bb1'],
            'red':['#fee0d2','#fc9272','#de2d26']}
        
    colors = col_dict[contour_color]
    for val in values:
        idx = (np.abs(csum - val)).argmin()
        levels.append(hflat[idx])

    print(levels)

    if fill:
        axis.contourf(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]])
    else:
        axis.contour(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]]) 

def medianline(axis, x, y, xrange=None, N=10, **kwargs):
    '''bins an array x and then calculates median of y in those bins'''
    if np.any(np.isnan(y)):
        print(f"contains {(np.isnan(y)).sum()} NaN in y")
    if xrange==None:
        xrange=[np.min(x),np.max(x)]
    xedges=np.linspace(xrange[0],xrange[1],N)
    xmids=0.5*(xedges[0:-1]+xedges[1:])
    bin_number=np.digitize(x,xedges)-1 #start with 0
    med=np.zeros(N-1)

    for i in range(N-1):
        bin=bin_number==i
        med[i]=np.median(y[bin])
 
    axis.plot(xmids,med,**kwargs)
    return xmids,med