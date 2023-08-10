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

def weighted_radius(Md,Rd,Mb,Rb):
    Mtot = Md+Mb
    return Md/Mtot*1.68*Rd+Mb/Mtot*Rb

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

    if Rb==0: #some bulges have Rb=0. Why?
        Rb = 0.01 * Rd

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
        w_rad = weighted_radius(Md,Rd,Mb,Rb)
        plt.plot([w_rad,w_rad],half,color='green',linestyle='dotted')
        plt.xlim([0,3*b])
        plt.ylim([0,1.0])
        plt.legend()
        plt.show()
    return half_radius

def main(df, type, plots=False):
    '''add some fields to tng based data sets'''
    column_names = df.columns.values
    if type == 'lgal': # LGALAXIES
        centrals= df['Type']==0
        mass_fields = ['M_Crit200','StellarMass','StellarDiskMass','StellarBulgeMass']
        spin_name = 'HaloSpin'
        r50name='r50'
        vvir = velocity_circular(df['M_Crit200'],df['R_Crit200'])
        r50 = calculate_r50(df,['StellarDiskMass','StellarDiskRadius','StellarBulgeMass','BulgeSize'])
        df['StellarR50'] = r50
        spin = calculate_spin(df['SubhaloSpinX'], df['SubhaloSpinY'],df['SubhaloSpinZ'],
            df['Vvir'],df['R_Crit200'])
        df['HaloSpin'] = spin
    elif type == 'sam':  #SC-SAM
        centrals = df['GalpropSatType']==0
        mass_fields = ['HalopropMvir','GalpropMstar','GalpropMdisk','GalpropMbulge']
        spin_name = 'HalopropSpin'
        r50name='r50'
        df['GalpropMdisk'] = df['GalpropMstar'] - df['GalpropMbulge']
        r50 = calculate_r50(df,['GalpropMdisk','GalpropRdisk','GalpropMbulge','GalpropRbulge'])
        df['GalpropHalfmassRadius'] = r50
    elif type=='sim' or type=='matchLHalo': #TNG simulations 
        centrals = df['SubhaloCentral']==True
        mass_fields = ['SubhaloMass','SubhaloMstar']
        spin_name = 'GroupSpin'
        r50name = 'SubhaloRstar'
        mass_defs = ['TopHat200','Crit200'] #more can be added
        for mdef in mass_defs:
            vvir = velocity_circular(df['Group_M_'+mdef],df['Group_R_'+mdef])
            df['Group_V_'+mdef] = vvir
        #there is an incosistency here thet J is for subhalo while R,V for Group
        spin = calculate_spin(df['SubhaloJx'][centrals], df['SubhaloJy'][centrals],
                            df['SubhaloJz'][centrals], df['Group_V_TopHat200'][centrals],
                            df['Group_R_TopHat200'][centrals])
        df['GroupSpin'] = 0
        df.loc[centrals,'GroupSpin'] = spin
        print('Spins are set to zero for satellites.')
    if type=='matchLHalo':
        mass_defs = ['TopHat200_dmo','Crit200_dmo'] #more can be added
        for mdef in mass_defs:
            vvir = velocity_circular(df['Group_M_'+mdef],df['Group_R_'+mdef])
            df['Group_V_'+mdef] = vvir
        spin = calculate_spin(df['SubhaloJx_dmo'][centrals], df['SubhaloJy_dmo'][centrals],
            df['SubhaloJz_dmo'][centrals], df['Group_V_TopHat200_dmo'][centrals],
            df['Group_R_TopHat200_dmo'][centrals])
        df['GroupSpin_dmo'] = 0
        df.loc[centrals,'GroupSpin_dmo'] = spin
    else:
        print('failed to detect file type')
        exit(1)

    print('min stellar mass = {:2e}'.format(np.min(df[mass_fields[1]])))
    print('mean spin = {}'.format(np.mean(df[spin_name][centrals])))
    #virial velocity
    if 'Vvir' in column_names:
        print('vvir max diff {:4f} km/s'.format(np.max(df['Vvir']-vvir)))
    if plots:
#        plt.scatter(np.log10(df[mass_fields[1]][centrals]),np.log10(r50name[centrals]),s=1,marker='.')
#        plt.ylim([-0.5,1.5])
#        plt.xlabel('log Stellar Mass')
#        plt.ylabel(r'log $R_{50}$')
#        plt.show()
  
        plt.hist(df[spin_name][centrals],range=[0,0.1],bins=100)
        plt.xlabel('Spin Parameter Bullock')
        plt.show()
 
        for field in mass_fields:
            counts, bins = np.histogram(np.log10(df[field][centrals]), range=[7,14],bins=40)
            plt.stairs(np.log10(counts), bins, label=field)
        plt.xlabel(r'log$_{10}$ Mass')
        plt.legend()
        plt.show()
    
    df_new.to_hdf('new_'+args.filename, key='s', mode='w')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'add vvir and r50 to galaxy dataframe')
    parser.add_argument('filename',help = 'pandas dataframe to add new field too.')
    parser.add_argument('--plots',action = 'store_true', default = False, 
        help = 'First show plots to test, must run again without plots to save new file' )
    args = parser.parse_args()
    df = pd.read_hdf(args.filename)
    fname=args.filename
    bs = fname[fname.find('tng')+3:fname.rfind('-')]
    type = fname[fname.find('-')+1:fname.rfind('.')]
    print(bs,type)
    if type=='lgal': #add halfmass radius, spin and Vvir
        mass = 'StellarMass'
        mcut={'50':1.e6,'100':1.e7,'300':1.e8}
    elif type=='sam':#add halfmass radius
        mass = 'GalpropMstar'  
        mcut={'50':1.e6,'100':1.e7,'300':1.e8}
    elif type=='sim' or type=='matchLHalo': #add spin and Vvir
        mass = 'SubhaloMstar'
        mcut = {'50':1.e8,'100':1.e9,'300':1.e10}
    else:
        print("error: {type} is not a known type")
    df_new = (df[df[mass] > mcut[bs]]).copy()
    main(df_new, type, plots=args.plots)
    