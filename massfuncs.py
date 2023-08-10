import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mass_func_vals(values,**kwargs):
    counts, bins = np.histogram(np.log10(values), **kwargs)  
    return np.log10(counts),bins  

df_dmo = pd.read_hdf('tng-dmo.h5')
df_sam1 = pd.read_hdf('tng50-sam.h5')
df_sam2 = pd.read_hdf('tng100-sam.h5')
df_sam3 = pd.read_hdf('tng300-sam.h5')
df_lgal = pd.read_hdf('tng100-lgal.h5')

fig,axes = plt.subplots(nrows=1,ncols=2, figsize=(8,4))

cent_sam1 = df_sam1['GalpropSatType']==0
cent_sam2 = df_sam2['GalpropSatType']==0
cent_sam3 = df_sam3['GalpropSatType']==0
cent_lgal = df_lgal['Type']==0

#halomass
#y, bins = mass_func_vals(df_dmo['Group_M_TopHat200'],range=[8,14],bins=100)
#axes[0].stairs(y, bins, label='Subfind Halo Mass')
#y, bins = mass_func_vals(df_sam['HalopropMvir'][cent_sam],range=[8,14],bins=100)
#axes[0].stairs(y, bins, label='SC-SAM Halo Mass')
#y, bins = mass_func_vals(df_lgal['M_Crit200'][cent_lgal],range=[8,14],bins=100)
#axes[0].stairs(y, bins, label= 'LGal Halo Mass')
y, bins = mass_func_vals(df_sam3['HalopropMvir'][cent_sam3],range=[7,15],bins=100)
axes[0].stairs(y, bins, label='TNG300-SCSAM')
y, bins = mass_func_vals(df_sam2['HalopropMvir'][cent_sam2],range=[7,15],bins=100)
axes[0].stairs(y, bins, label='TNG100-SCSAM')
y, bins = mass_func_vals(df_sam1['HalopropMvir'][cent_sam1],range=[7,15],bins=100)
axes[0].stairs(y, bins, label='TNG50-SCSAM')
axes[0].legend(loc='lower left')
axes[0].set_ylabel('$\phi$')
axes[0].set_xlabel(r'log$_{10}$ Halo Mass')
#stellar mass
y, bins = mass_func_vals(df_sam3['GalpropMstar'],range=[4,12],bins=100)
axes[1].stairs(y,bins,label= 'TNG300-SCSAM')
y, bins = mass_func_vals(df_sam3['GalpropMstar'][cent_sam3],range=[4,12],bins=100)
axes[1].stairs(y,bins,label= 'TNG300-SCSAM Centrals')
y, bins = mass_func_vals(df_sam2['GalpropMstar'],range=[4,12],bins=100)
axes[1].stairs(y,bins,label= 'TNG100-SCSAM')
y, bins = mass_func_vals(df_sam2['GalpropMstar'][cent_sam2],range=[4,12],bins=100)
axes[1].stairs(y,bins,label= 'TNG100-SCSAM Centrals')
y, bins = mass_func_vals(df_sam1['GalpropMstar'],range=[4,12],bins=100)
axes[1].stairs(y,bins,label= 'TNG50-SCSAM')
y, bins = mass_func_vals(df_sam1['GalpropMstar'][cent_sam1],range=[4,12],bins=100)
axes[1].stairs(y,bins,label= 'TNG50-SCSAM Centrals')

#y, bins = mass_func_vals(df_lgal['StellarMass'],range=[5,12],bins=100)
#axes[1].stairs(y,bins,label= 'LGal All Stellar Mass')
#y, bins = mass_func_vals(df_sam['GalpropMstar'][cent_sam],range=[5,12],bins=100)
#axes[1].stairs(y,bins,label= 'SC-SAM Central Stellar Mass')
#y, bins = mass_func_vals(df_lgal['StellarMass'][cent_lgal],range=[5,12],bins=100)
#axes[1].stairs(y,bins,label= 'LGal Central Stellar Mass')
axes[1].legend(loc='lower left')
axes[1].set_xlabel(r'log$_{10}$ Stellar Mass')


plt.show()

print(f"SC-SAM Mstar > 1.e8: {(df_sam1['GalpropMstar'] > 1.e8).sum()}")
print(f"SC-SAM Mstar > 1.e8: {(df_sam2['GalpropMstar'] > 1.e8).sum()}")
print(f"SC-SAM Mstar > 1.e8: {(df_sam3['GalpropMstar'] > 1.e8).sum()}")
#print(f"LGals Mstar > 1.e8: {(df_lgal['StellarMass'] > 1.e8).sum()}")
