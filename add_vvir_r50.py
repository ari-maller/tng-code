import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import galsim #install with conda install -c conda_forge galsim

G = 4.3e-6 #units of kpc/Msun (km/s)^2

def velocity_circular(mass,radius):
    return np.sqrt(G*mass/radius)

def calculate_spin(jx,jy,jz,vvir,rvir):
    '''calculate spin from subfind jx,jy,jz'''
    return np.sqrt(jx**2 + jy**2 + jz**2) / (np.sqrt(2)*vvir*rvir)

def calculate_r50(df,field_names):
    Md = (df.get(field_names[0])).to_numpy()
    Rd = (df.get(field_names[1])).to_numpy()
    Mb = (df.get(field_names[2])).to_numpy()
    Rb = (df.get(field_names[3])).to_numpy()
    N = df.shape[0]
    r50 = np.zeros(N)
    for i in range(N):
        r50[i] = half_mass_radius(Md[i],Rd[i],Mb[i],Rb[i])
    return r50

def half_mass_radius(Md,Rd,Mb,Rb,tol=1.e-6,figure=False):
    '''calculate half mass radius for disk(n=1)+bulge(n=4) galaxy
    using bisection'''
    if Md==0 and Mb==0:
        return 0.0
    disk_fraction=Md/(Md+Mb)
    bulge_fraction=Mb/(Md+Mb)
    if bulge_fraction==0:
        disk=galsim.Sersic(scale_radius=Rd,n=1)  
        return disk.half_light_radius     
    if disk_fraction==0:
        return Rb
    disk=galsim.Sersic(scale_radius=Rd,n=1)
    bulge=galsim.Sersic(half_light_radius=Rb,n=4)
    #starting points for bisection
    a=bulge.half_light_radius
    b=disk.half_light_radius
    if b < a:
        a=disk.half_light_radius
        b=bulge.half_light_radius

    #bisection
    Ma=disk_fraction*disk.calculateIntegratedFlux(a)+bulge_fraction*bulge.calculateIntegratedFlux(a)
    Mb=disk_fraction*disk.calculateIntegratedFlux(b)+bulge_fraction*bulge.calculateIntegratedFlux(b)
    if np.sign(Ma-0.5)==np.sign(Mb-0.5):
        raise Exception("a and b do not bound a root")
    while(b-a > tol):
        m=0.5*(a+b)
        f=(disk_fraction*disk.calculateIntegratedFlux(m)+
            bulge_fraction*bulge.calculateIntegratedFlux(m))-0.5
        if np.sign(f)==1:
            b=m
        else:
            a=m
    half_radius=0.5*(a+b)
    if figure:
        half=np.array([0.45,0.55]) #used for half marks
        R=np.arange(0,3*b,0.02)
        N=len(R)
        yd=np.zeros(N)
        yb=np.zeros(N)
        for i in range(N):
            yd[i]=disk_fraction*disk.calculateIntegratedFlux(R[i])
            yb[i]=bulge_fraction*bulge.calculateIntegratedFlux(R[i])
        plt.plot(R,yd,color='b',label='disk')
        plt.plot(R,yb,color='r',label='bulge')
        plt.plot(R,yd+yb,color='k',label='total')
        plt.plot([bulge.half_light_radius,bulge.half_light_radius],bulge_fraction*half,color='g')
        plt.plot([disk.half_light_radius,disk.half_light_radius],disk_fraction*half,color='g')
        plt.plot([half_radius,half_radius],half,color='g')
        plt.xlim([0,3*b])
        plt.ylim([0,1.0])
        plt.legend()
        plt.show()
    return half_radius

def main(df, plots=False):
    print('min stellar mass = {:2e}'.format(np.min(df['StellarMass'])))
    column_names = df.columns.values
    centrals= df['Type']==0
    #virial velocity
    field_names=['M_Crit200','R_Crit200']
    vvir = velocity_circular(df[field_names[0]],df[field_names[1]])
    print('vvir max diff {:4f} km/s'.format(np.max(df['Vvir']-vvir)))
    if plots:
        plt.scatter(df['Vvir'][centrals],vvir[centrals],s=1)
        plt.xlabel('Virial Velocity from source')
        plt.ylabel('Virial velocity calculated from M_crit and R_crit')
        plt.show()
    #halfmass radius
    field_names = ['StellarDiskMass','StellarDiskRadius','StellarBulgeMass','BulgeSize']
    r50=calculate_r50(df,field_names)
    if plots:
        plt.scatter(np.log10(df['StellarMass'][centrals]),np.log10(r50[centrals]),s=1,marker='.')
        plt.xlabel('log Stellar Mass')
        plt.ylabel(r'log $R_{50}$')
        plt.show()
    else:
        df['StellarR50'] = r50
    #spin - only fron subfind so fieldnames the same
#    field_names = ['SubhaloSpinX', 'SubhaloSpinY', 'SubhaloSpinZ', 'Group_M_Crit200', 'Group_R_Crit200']
    field_names = ['SubhaloSpinX', 'SubhaloSpinY', 'SubhaloSpinZ', 'Vvir', 'R_Crit200']

    spin = calculate_spin(df[field_names[0]], df[field_names[1]],df[field_names[2]],
        df[field_names[3]],df[field_names[4]])
    print('mean spin = {}'.format(np.mean(spin[centrals])))
    if plots:   
        plt.hist(spin[centrals],range=[0,0.1],bins=100)
        plt.xlabel('Spin Parameter Bullock')
        plt.show()
    else:
        df['HaloSpin'] = spin
    
    if plots:
        counts, bins = np.histogram(np.log10(df['M_Crit200']), range=(9,15), bins=100)
        plt.stairs(np.log10(counts), bins)
        plt.xlabel('log Halo Mass')
        plt.show()
    else:
        df_new.to_hdf('new_'+args.filename, key='s', mode='w')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'add vvir and r50 to galaxy dataframe')
    parser.add_argument('filename',help = 'pandas dataframe to add new field too.')
    parser.add_argument('--plots',action = 'store_true', default = False, 
        help = 'First show plots to test, must run again without plots to save new file' )
    args = parser.parse_args()
    df = pd.read_hdf(args.filename)
    df_new = (df[df['StellarMass'] > 1.e8]).copy()
    main(df_new, plots=args.plots)
    