import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#fitting functions for MMW
def f_c(c):
	return 2.0 / 3.0 + (c / 21.5)**0.7

def f_R(lprime, md, c):
	p = -0.06 + 2.71 * md + 0.0047 / lprime;
	return (lprime/0.1)**p * (1.0-3.0*md+5.2*md*md)*(1.0-0.019*c+0.00025*c*c + 0.52/c)

def disk_infall(m_halo,r_vir,conc,spin,mstar_disk,mass_cold_gas):
    ''' disk infall'''
    if conc < 2.0: 
        conc = 2.0
    if conc > 30.0: 
        conc = 30.0
    if spin < 0.02:
        spin=0.02
    r_iso = spin*r_vir/np.sqrt(2.0)   
    mdisk = (mstar_disk + mass_cold_gas)
    fdisk = mdisk/m_halo
    fj=1.0
    if fdisk > 0.02:
        r_disk = fj*r_iso*f_R(fj*spin, fdisk, conc)/np.sqrt(f_c(conc))
    else:
        r_disk = 0.02*r_vir
    return r_disk

if __name__=='__main__':
    df = pd.read_hdf('tng100-sam.h5')
    cent = df['GalpropSatType']==0
    print((df['HalopropSpin'][cent]).describe())
    print((df['GalpropMdisk'][cent]/df['HalopropMvir'][cent]).describe())
    vvir = np.sqrt(4.3e-6 * df['HalopropMvir']/(1000*df['GalpropRhalo']))
    plt.scatter(df['HalopropC_nfw'][cent],1.14*((vvir[cent]/df['GalpropVdisk'][cent])**2),
        alpha=0.3,marker='.',s=1)
    cnfw = np.arange(2,30,0.5)
    plt.plot(cnfw,1.678*f_R(0.034,0.0022,cnfw)/np.sqrt(2*f_c(cnfw)), color='red', linestyle='dotted')   
    plt.plot(cnfw,1.678*f_R(0.022,0.0035,cnfw)/np.sqrt(2*f_c(cnfw)), color='red',linestyle='dashed')
    plt.plot(cnfw,1.678*f_R(0.034,0.0022,cnfw)/np.sqrt(2*f_c(cnfw)), color='red')
    plt.plot(cnfw,1.678*f_R(0.05,0.0035,cnfw)/np.sqrt(2*f_c(cnfw)), color='red',linestyle='dashed')
    plt.plot(cnfw,1.678*f_R(0.034,0.007,cnfw)/np.sqrt(2*f_c(cnfw)),color='red', linestyle='dotted')

    plt.xlabel('c$_{nfw}$')
    plt.ylabel('F')
    plt.xlim([0,30])
    plt.ylim([0.2,1.5])
    plt.show()

    cdisk = np.logical_and(df['GalpropSatType']==0,df['GalpropMbulge']/df['GalpropMstar'] < 0.5)
    fdisk = df['GalpropMdisk']/df['GalpropMstar']
    fbulge = 1.0 - fdisk
    f,axis = plt.subplots()
#    plt.scatter(df['GalpropR50'][cent],
#        fbulge[cent]*df['GalpropRbulge'][cent]+fdisk[cent]*1.68*df['GalpropRdisk'][cent],
#        marker=',',s=1,alpha=0.3)
    axis.scatter(df['GalpropR50'][cdisk],fdisk[cdisk]*1.68*df['GalpropRdisk'][cdisk],
        marker=',',s=1, alpha=0.3, label = r'$M_{disk} / M_{stellar} < 0.5$')
    axis.set_xlabel('True Half Mass Radius')
    axis.set_ylabel(r'($M_{disk} / {M_{star}}) 1.68 R_{disk}$')
    axis.set_xlim([0,50])
    axis.set_ylim([0,50])
    axis.set_aspect('equal','box')
    axis.legend(loc='lower right')
    plt.show()