import site 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

import illustris_python as ilsim
site.addsitedir('/Users/ari/Code/')
import illustris_sam as ilsam

hubble_value = 0.704
tng_mass_unit = 1.e10
scsam_mass_unit = 1.e9
scale_factor = 1.0 #need to set based on snapshot z 
G_Msunkpc = 4.3e-6 

def generate_sub_volume_list(n):
    """generates a list of all the subvolumes, used for loading in illustris-sam"""
    subvolume_list = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                subvolume_list.append([i, j, k])

    return subvolume_list

def tngdict2df(catalog):
    '''convert tng group/sub catalog to a pandas data frame'''
    df = pd.DataFrame()
    for field in catalog.keys():
            tmp = catalog[field][()]
            if tmp.ndim > 1:
                if tmp.shape[1]==3: #x,y,z
                    df[field+'X'] = tmp[:,0]
                    df[field+'Y'] = tmp[:,1]
                    df[field+'Z'] = tmp[:,2]
                elif tmp.shape[1]==5: #Gas,DM,_,Stars,BHs
                    df[field+'gas'] = tmp[:,0]
                    df[field+'dm'] = tmp[:,1]
                    df[field+'star'] = tmp[:,4]    
                    df[field+'bh'] = tmp[:,2]               
            else:
                df[field] = tmp

def read_tng_hih2(filename):
    df=pd.DataFrame()
    masses = ['m_h2_GD14_map','m_h2_GD14_vol','m_h2_GK11_map',
                'm_h2_GK11_vol','m_h2_K13_map','m_h2_K13_vol',
                'm_h2_L08_map','m_h2_S14_map','m_h2_S14_vol',
                'm_hi_GD14_map','m_hi_GD14_vol','m_hi_GK11_map',
                'm_hi_GK11_vol','m_hi_K13_map','m_hi_K13_vol',
                'm_hi_L08_map','m_hi_S14_map','m_hi_S14_vol']
    with h5py.File(filename) as f:
        df['id_subhalo'] = (f['id_subhalo'][()]).astype(int)
        df['is_primary'] = (f['is_primary'][()]).astype(bool)
        #using GK11 as default model
        df['SubhaloMHI'] = f['m_hi_GK11_vol'][()]
        df['SubhaloMH2'] = f['m_h2_GK11_vol'][()]

    return df

def read_tng_match(filename, snapshot=99):
    '''this loads the match ids from tng hydro to dmo for a given snapshot.
        Note all ids are stored in one file'''
    #contains all snapshots
    df = pd.DataFrame()
    fields = ['SubhaloIndexDark_LHaloTree', 'SubhaloIndexDark_Lagrange', 
              'SubhaloIndexDark_SubLink', 'SubhaloSnapDark_Lagrange']
    fields = ['SubhaloIndexDark_LHaloTree', 'SubhaloIndexDark_SubLink']
    snap='Snapshot_{:2}/'.format(snapshot) #need to allow this to be set from args
    with h5py.File(filename) as f:
        for field in fields:
            df[field] = f[snap+field][()]

    return df

#functions for loading the santa cruz semianalytic catalogs
def load_scsam_halos(base_path, snapshot=99, N=5):
    """load sam halos returns a pandas dataframe"""
    sv_list = generate_sub_volume_list(N)
    sam_halos = ilsam.groupcat.load_snapshot_halos(base_path, snapshot, sv_list)
    mass_fields = ['HalopropMass_ejected','HalopropMcooldot','HalopropMhot',
     'HalopropMstar_diffuse','HalopropMvir','HalopropZhot']
    df = pd.DataFrame(sam_halos)
    for f in mass_fields:
        df[f] = scsam_mass_unit * df[f]

    return df

def load_scsam_galaxies(base_path, snapshot=99, N=5):
    """load sam galaxies returns a pandas dataframe"""
    sv_list = generate_sub_volume_list(N)
    sam_gals = ilsam.groupcat.load_snapshot_subhalos(base_path, snapshot, sv_list)
    pos = sam_gals.pop('GalpropPos')
    vel = sam_gals.pop('GalpropVel')
    df = pd.DataFrame(sam_gals)
    pos_field = ['X', 'Y', 'Z'] #these are in Mpc
    vel_field = ['Vx', 'Vy', 'Vz']
    for i in range(3):
        df['Galprop' + pos_field[i]] = pos[:, i]
        df['Galprop' + vel_field[i]] = vel[:, i]

    mass_fields = ['GalpropMBH','GalpropMH2','GalpropMHI','GalpropMHII',
        'GalpropMbulge','GalpropMcold','GalpropMvir','GalpropMstar',
        'GalpropMstar_merge','GalpropMstrip']
    for f in mass_fields:
        df[f] = scsam_mass_unit * df[f]
    df['GalpropRhalo'] = 1000*df['GalpropRhalo'] #this is in Mpc, convert to kpc
    return df


def load_scsam_gals_and_halos(base_path, snapshot=99, N=5):
    """load and combine sc-sam galaxy and halo properties into 1 df"""
    df_gals = load_scsam_galaxies(base_path, snapshot=snapshot, N=N)
    df_halos = load_scsam_halos(base_path, snapshot=snapshot, N=N)
    df = df_gals.merge(df_halos, left_on='GalpropHaloIndex_Snapshot',
        right_on='HalopropIndex_Snapshot')
    return df

#functions for loading the L-Galaxies semianalytic catalogs
def load_lgal_galaxies(base_path, snapshot=99):
    '''load the LGal SAM model for TNG, does not include magnitudes'''
    file_path=base_path+'postprocessing/LGalaxies_{:03d}.hdf5'.format(snapshot)
    df = pd.DataFrame()
    with h5py.File(file_path,'r') as f:
        for field in f['Galaxy'].keys():
            tmp = f['Galaxy/'+field][()]
            if tmp.ndim > 1:
                if tmp.shape[1]==3: #if 20 a magnitude, skip for now
                    df[field+'X'] = tmp[:,0]
                    df[field+'Y'] = tmp[:,1]
                    df[field+'Z'] = tmp[:,2]
            else:
                df[field] = tmp
    
    mass_fields = ['BlackHoleMass', 'Central_M_Crit200', 'ColdGasMass', 'EjectedMass',
        'HaloStellarMass', 'HotGasMass', 'InfallHotGasMass','M_Crit200',
        'MetalsBulgeMass', 'MetalsColdGasMass', 'MetalsDiskMass', 'MetalsEjectedMass', 
        'MetalsHaloStellarMass', 'MetalsHotGasMass', 'MetalsStellarMass',
        'StellarMass', 'StellarBulgeMass', 'StellarDiskMass' ]
    for f in mass_fields:
        df[f] = tng_mass_unit*df[f]/ hubble_value
    radii_fields = ['BulgeSize','Central_R_Crit200','CoolingRadius',
        'DistanceToCentralGalX', 'DistanceToCentralGalY', 'DistanceToCentralGalZ',
        'GasDiskRadius','HotGasRadius','R_Crit200', 'StellarDiskRadius']
    for f in radii_fields:
        df[f] = df[f] * scale_factor/hubble_value
    df['StellarDiskRadius'] = df['StellarDiskRadius']/3.0 #convert to disk scale length
    return df

def join_lgal_dmo(df_gal,dmo_path,snapshot=99):
    '''join the dmo halos to the Lgal galaxies, only subhalo not FOF fields'''
    fields = ['SubhaloHalfmassRad', 'SubhaloMass', 'SubhaloMassInMaxRad', 
        'SubhaloParent', 'SubhaloPos', 'SubhaloSpin', 'SubhaloVel', 'SubhaloVelDisp', 
        'SubhaloVmax', 'SubhaloVmaxRad']
    subs = ilsim.groupcat.loadSubhalos(dmo_path, snapshot, fields=fields)
    for f in fields:
        if subs[f].ndim > 1:
            df_gal[f+'X'] = subs[f][df_gal['SubhaloIndex_TNG-Dark'],0]
            df_gal[f+'Y'] = subs[f][df_gal['SubhaloIndex_TNG-Dark'],1]
            df_gal[f+'Z'] = subs[f][df_gal['SubhaloIndex_TNG-Dark'],2]
        else:
            df_gal[f]=subs[f][df_gal['SubhaloIndex_TNG-Dark']]

    mass_fields = ['SubhaloHalfmassRad', 'SubhaloMass', 'SubhaloMassInMaxRad']
    for f in mass_fields:
        df_gal[f] = tng_mass_unit*df_gal[f]/ hubble_value
    spin_fields = ['SubhaloSpinX','SubhaloSpinY','SubhaloSpinZ']
    for f in spin_fields:
        df_gal[f] = df_gal[f] / hubble_value
    return df_gal

#load tng simulation catalog files 
def load_sim_halos(base_path, snapshot=99):
    """create a dataframe of TNG subfind halos"""
    fields = ['GroupBHMass','GroupFirstSub','GroupMass','GroupNsubs',
        'GroupPos','GroupVel','Group_M_Crit200', 'Group_M_TopHat200', 'Group_R_Crit200',
        'Group_R_TopHat200']
    halos = ilsim.groupcat.loadHalos(base_path, snapshot, fields=fields)
    pos = halos.pop('GroupPos')
    vel = halos.pop('GroupVel')
    df = pd.DataFrame(halos)
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']

    for i in range(3):
        df['Group' + pos_field[i]] = pos[:, i] #cMpc
        df['Group' + vel_field[i]] = vel[:, i]

    mass_fields = ['GroupBHMass','GroupMass','Group_M_Crit200', 'Group_M_TopHat200']
    for f in mass_fields:
        df[f] = df[f] * tng_mass_unit / hubble_value
    radii_fields = ['Group_R_Crit200','Group_R_TopHat200']
    for f in radii_fields:
            df[f] = df[f] / hubble_value
#    df['GroupBHMdot'] = (tng_mass_unit/ 0.978) * df['GroupBHMdot'] #Msun/Gyr
    return df


def load_sim_galaxies(base_path, snapshot=99):
    """create a dataframe of TNG subfind galaxies (subhalos)"""
    fields = ['SubhaloFlag','SubhaloBHMass','SubhaloBHMdot','SubhaloGasMetallicity',
        'SubhaloGrNr','SubhaloHalfmassRadType','SubhaloMass','SubhaloMassInMaxRadType',
        'SubhaloParent','SubhaloPos','SubhaloSFRinRad','SubhaloSpin','SubhaloStarMetallicity',
        'SubhaloVel','SubhaloVelDisp','SubhaloVmax','SubhaloVmaxRad']
    subs = ilsim.groupcat.loadSubhalos(base_path, snapshot, fields=fields)
    #pop out fields that are more than 1D
    count = subs.pop('count') # this is the number of subs
    radii = subs.pop('SubhaloHalfmassRadType')
    masses = subs.pop('SubhaloMassInMaxRadType')
    pos = subs.pop('SubhaloPos')
    vel = subs.pop('SubhaloVel')
    spins = subs.pop('SubhaloSpin')
    df = pd.DataFrame(subs)
    df['SubhaloRgas'] = radii[:, 0] * scale_factor / hubble_value 
#    df['SubhaloRhalo'] = radii[:, 1] * scale_factor / hubble_value 
    df['SubhaloRstar'] = radii[:, 4] * scale_factor/ hubble_value
    df['SubhaloMgas'] = masses[:, 0] * tng_mass_unit / hubble_value
#    df['SubhaloMdark'] = masses[:, 1] * tng_mass_unit / hubble_value
    df['SubhaloMstar'] = masses[:, 4] * tng_mass_unit / hubble_value
    #SubhaloMass should equal Subhalo(Mgas+Mdark+Mstar+BHMass)
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    spin_field = ['Jx', 'Jy', 'Jz']
    for i in range(3):
        df['Subhalo' + pos_field[i]] = pos[:, i]  # cMpc/h
        df['Subhalo' + vel_field[i]] = vel[:,i]
        df['Subhalo' + spin_field[i]] = spins[:, i] / hubble_value

    mass_fields = ['SubhaloBHMass', 'SubhaloMass']
    for f in mass_fields:
        df[f] = df[f] * tng_mass_unit / hubble_value

    df['SubhaloVmaxRad'] = df['SubhaloVmaxRad'] * scale_factor/hubble_value
    return df

def load_sim_gals_and_halos(base_path,snapshot=99):
    df = load_sim_galaxies(base_path, snapshot=99)
    fields = ['GroupBHMass','GroupFirstSub','GroupMass','GroupNsubs',
        'GroupPos','GroupVel','Group_M_Crit200', 'Group_M_TopHat200', 
        'Group_R_Crit200', 'Group_R_TopHat200']
    halos = ilsim.groupcat.loadHalos(base_path, snapshot, fields=fields)
    #identify central galaxies
    Nhalos = halos.pop('count')
    central = np.zeros((df.shape)[0], dtype='int')
    central[halos['GroupFirstSub']] = 1
    df['SubhaloCentral'] = central
    pos = halos.pop('GroupPos')
    vel = halos.pop('GroupVel')
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    df['GroupNsubs'] = halos['GroupNsubs'][df['SubhaloGrNr']]
    for i in range(3):
        df['Group' + pos_field[i]] = pos[df['SubhaloGrNr'], i] #cMpc
        df['Group' + vel_field[i]] = vel[df['SubhaloGrNr'], i]
    
    mass_fields = ['GroupBHMass','GroupMass','Group_M_Crit200', 'Group_M_TopHat200']
    for f in mass_fields:
        df[f] = halos[f][df['SubhaloGrNr']] * tng_mass_unit / hubble_value
    radii_fields = ['Group_R_Crit200','Group_R_TopHat200']
    for f in radii_fields:
        df[f] = halos[f][df['SubhaloGrNr']] / hubble_value

    #add post processing fields
    post_dir=base_path[0:base_path.rfind('output')]+'postprocessing/'
    #match id to DMO runs
    match_file = post_dir + 'subhalo_matching_to_dark.hdf5' 
    df_match = read_tng_match(match_file)
    df['SubhaloIndexDark_LHaloTree'] = df_match['SubhaloIndexDark_LHaloTree']
    df['SubhaloIndexDark_SubLink'] = df_match['SubhaloIndexDark_SubLink']
    #HI and H2 masses (only for subset of galaxies)
    hih2_file = post_dir + 'hih2_galaxy_099.hdf5'
    df_hih2 = read_tng_hih2(hih2_file)
    mask = df_hih2['id_subhalo']
    for field in df_hih2.columns.values:
        if field !='id_subhalo':
            df[field] = np.NaN
            df.loc[mask,field] = df_hih2.loc[:,field]

    return df

def load_sim_dmo(base_path,snapshot=99):
    '''load DMO halos from Illustris-TNG simulations (group and subhalo)'''
    #first load subhalo catalog
    fields = ['SubhaloGrNr', 'SubhaloMass', 'SubhaloParent', 'SubhaloPos', 
        'SubhaloSpin', 'SubhaloVel', 'SubhaloVelDisp', 'SubhaloVmax', 'SubhaloVmaxRad']
    subs = ilsim.groupcat.loadSubhalos(base_path, snapshot,fields=fields)
    #pop out fields that are more than 1D
    count = subs.pop('count') # this is the number of subs
    pos = subs.pop('SubhaloPos')
    vel = subs.pop('SubhaloVel')
    spins = subs.pop('SubhaloSpin')
    df = pd.DataFrame(subs)
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    spin_field = ['Jx', 'Jy', 'Jz']
    for i in range(3):
        df['Subhalo' + pos_field[i]] = pos[:, i]  # cMpc/h
        df['Subhalo' + vel_field[i]] = vel[:,i]
        df['Subhalo' + spin_field[i]] = spins[:, i] / hubble_value

    df['SubhaloMass'] = df['SubhaloMass'] * tng_mass_unit / hubble_value
    df['SubhaloVmaxRad'] = df['SubhaloVmaxRad'] * scale_factor/hubble_value
    
    #now load halo catalog
    fields = ['GroupFirstSub','GroupMass','GroupNsubs',
        'GroupPos','GroupVel','Group_M_Crit200', 'Group_M_TopHat200', 
        'Group_R_Crit200', 'Group_R_TopHat200']

    halos = ilsim.groupcat.loadHalos(base_path, snapshot, fields=fields)
    #identify central galaxies
    Nhalos = halos.pop('count')
    central = np.zeros((df.shape)[0], dtype='int')
    central[halos['GroupFirstSub']] = 1
    df['SubhaloCentral'] = central
    pos = halos.pop('GroupPos')
    vel = halos.pop('GroupVel')
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    df['GroupNsubs'] = halos['GroupNsubs'][df['SubhaloGrNr']]
    for i in range(3):
        df['Group' + pos_field[i]] = pos[df['SubhaloGrNr'], i] #cMpc
        df['Group' + vel_field[i]] = vel[df['SubhaloGrNr'], i]
    
    mass_fields = ['GroupMass', 'Group_M_Crit200', 'Group_M_TopHat200']
    for f in mass_fields:
        df[f] = halos[f][df['SubhaloGrNr']] * tng_mass_unit / hubble_value
    radii_fields = ['Group_R_Crit200','Group_R_TopHat200']
    for f in radii_fields:
        df[f] = halos[f][df['SubhaloGrNr']] / hubble_value
 
    return df

def load_sim_and_dmo(sim_path,dmo_path, match = 'LHalo'):
    '''load the simulation with matched dmo halos'''
    df_sim = load_sim_gals_and_halos(sim_path,snapshot=99)
    df_dmo = load_sim_dmo(dmo_path,snapshot=99)
    if match=='LHalo':
        match_idx = 'SubhaloIndexDark_LHaloTree'
    elif match=='SubLink':
        match_idx = 'SubhaloIndexDark_SubLink'
    else:
        print('match set to unkonwn value')
        exit(1)

    mask = df_sim[match_idx] != -1
    print('Fraction of all subhalos with a match {:.3f}'.format(mask.sum()/len(mask)))
#    i = 10000
#    print(df_sim.loc[i,'SubhaloMass']/1.e11)
#    idx = df_sim.loc[i,'SubhaloIndexDark_LHaloTree']
#    print(df_dmo.loc[idx,'SubhaloMass']/1.e11)
#    exit(1)
    df = df_sim.loc[mask,:].copy()
    for field in list(df_dmo):
        df[field+'_dmo'] = df_dmo[field][df[match_idx]]
    
    return df

def join_sam_and_sim(df_sam, df_sim):
    """join the sam and sim dataframes"""
    df = df_sam.join(df_sim, on='GalpropSubfindIndex_FP', rsuffix='_sim')
    return df

def center_distance(df, type='sim'):
    if type == 'gals':
        p1name = ['GalpropX', 'GalpropY', 'GalpropZ']
        p2name = ['GroupX', 'GroupY', 'GroupZ']
    elif type == 'sim':
        p1name = ['GroupX', 'GroupY', 'GroupZ']
        p2name = ['SubhaloX', 'SubhaloY', 'SubhaloZ']
    elif type == 'halos':
        p1name = ['GalpropX', 'GalpropY', 'GalpropZ']
        p2name = ['SubhaloX', 'SubhaloY', 'SubhaloZ']
    elif type=='lgal':
        p1name = ['PosX','PosY','PosZ']
        p2name = ['SubhaloPosX', 'SubhaloPosY', 'SubhaloPosZ']
    elif type=='dmo':
        p1name = ['SubhaloX', 'SubhaloY', 'SubhaloZ']
        p2name = ['SubhaloX_dmo', 'SubhaloY_dmo', 'SubhaloZ_dmo']
    else:
        print('Not a valid type')
        return
    s = 0
    for i in range(3):
        s = s + (df[p1name[i]] - df[p2name[i]]) ** 2

    return np.sqrt(s)


def remove_columns(df, fields):
    """Remove the listed columns from a dataframe"""
    for f in fields:
        del df[f]

    return df


def main(args):
    """create a pandas dataframe for a TNG dataset. Options include:
    --sim just simulation data 
    --sam just the SC-SAM data
    --dmo just the subfind halos from TNG-Dark
    --sim and --sam joins the SC-SAM and the simulated galaxies (centrals only)
    --sim and --dmo joins the simulated galaxies with the TNG-Dark matches (centrals only)
    --sam and --dmo joins the SC-SAM with the subfind catalog of TNG-DARK (centrals only)
    --lgal and --dmo the LGalixes SAM linked with the TNG-Dark catalog 
    --sim and --lgal joins the LGalaxies SAM with the matched simulated galaxies (centrals only)
    --tests flag checks that matching is correct"""
    if args.sam: #sc-sam
        N = {50:6, 100:5, 300:7}
        df_sam = load_scsam_gals_and_halos(args.sam,N = N[args.boxsize])
        df_sam['GalpropVvir'] = np.sqrt(G_Msunkpc*df_sam['GalpropMvir']/df_sam['GalpropRhalo'])
        print('sam catalog: ', df_sam.shape)
        if args.tests:
            print(df_sam.columns.values)
            cents = df_sam['GalpropSatType']==0
            print("max Mvir diff = {}".format(np.max(df_sam['GalpropMvir'][cents] 
                 - df_sam['HalopropMvir'][cents])))
        df_sam = remove_columns(df_sam, ['GalpropBirthHaloID', 'GalpropHaloIndex', 
            'GalpropHaloIndex_Snapshot','GalpropRootHaloID','GalpropRedshift',
            'HalopropHaloID','HalopropIndex','HalopropIndex_Snapshot','HalopropRedshift',
            'HalopropRockstarHaloID','HalopropRootHaloID', 'HalopropSnapNum'])
        if not args.sim:
#            df_sam = remove_columns(df_sam, ['GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP',
#                'HalopropFoFIndex_DM', 'HalopropFoFIndex_FP'])
            df_sam.to_hdf(f'all_tng{args.boxsize}-sam.h5', key='s', mode='w')
      
    if args.sim and not args.dmo:
        df_sim = load_sim_gals_and_halos(args.sim, snapshot = args.snapshot)
        df_sim = df_sim[df_sim['SubhaloFlag']==True]
        df_sim = remove_columns(df_sim, ['SubhaloFlag','SubhaloGrNr', 'SubhaloParent'])
        print('sim catalog: ', df_sim.shape)
        if args.tests:
            print(df_sim.columns.values)
            cents = df_sim['SubhaloCentral']  == True
            print(np.max(center_distance(df_sim[cents], type='sim')))
        
#        df_sim['SubhaloVvir'] = np.sqrt(G_Msunkpc*df_sim['Group_M_Crit200']/df_sim['Group_R_Crit200'])
        df_sim.to_hdf(f'all_tng{args.boxsize}-sim.h5', key='s', mode='w')
          
    if args.dmo and not args.sim:
        df_dmo = load_sim_dmo(args.dmo)
        df_dmo = remove_columns(df_dmo, ['SubhaloGrNr', 'SubhaloParent'])
        if args.tests:
            print(df_dmo.columns.values)
            cents = df_dmo['SubhaloCentral']  == True
            print(np.max(center_distance(df_dmo[cents], type='sim')))
#        df_dmo = remove_columns(df_dmo, ['GroupFirstSub'])
        df_dmo.to_hdf(f'all_tng{args.boxsize}-dmo.h5', key='s', mode='w')

    if args.lgal and args.dmo:
        df_lgal = load_lgal_galaxies(args.lgal)
        df = join_lgal_dmo(df_lgal,args.dmo)

        if args.tests:
            centrals = df['Type']==0
            print('max center distance {}'.format(np.max(center_distance(df[centrals],type='lgal'))))
            print('max vmax difference {}'.format(np.max(df['Vmax'][centrals]-df['SubhaloVmax'][centrals])))
            print('Number central galaxies with SubhaloParent not zero {}'.format(df['SubhaloParent'][centrals].sum()))
        df = remove_columns(df, ['SubhaloIndex_TNG-Dark','VelX', 'VelY','VelZ',
            'SubhaloVelX', 'SubhaloVelY','SubhaloVelZ', 'SubhaloVmax'])
        df.to_hdf(f'all_tng{args.boxsize}-lgal.h5',  key='s', mode='w')

    if args.sim and args.dmo:
        df = load_sim_and_dmo(args.sim, args.dmo, match=args.match)
        df = df[df['SubhaloFlag']==True]
        df = remove_columns(df, ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloParent',
            'SubhaloIndexDark_LHaloTree', 'SubhaloIndexDark_SubLink', 'is_primary',
            'SubhaloGrNr_dmo', 'SubhaloParent_dmo'])
        if args.tests:
            print(df.columns.values)
            cents = df['SubhaloCentral']  == True
            print('max center distance {}'.format(np.max(center_distance(df[cents],type='dmo'))))
        df = remove_columns(df, ['SubhaloX', 'SubhaloVx', 'SubhaloY',
            'SubhaloVy','SubhaloZ', 'SubhaloVz', 'GroupX', 'GroupVx','GroupY', 
            'GroupVy', 'GroupZ', 'GroupVz',
            'SubhaloX_dmo', 'SubhaloVx_dmo','SubhaloY_dmo','SubhaloVy_dmo',
            'SubhaloZ_dmo', 'SubhaloVz_dmo','GroupX_dmo', 'GroupVx_dmo', 
            'GroupY_dmo', 'GroupVy_dmo', 'GroupZ_dmo', 'GroupVz_dmo'])
        df.to_hdf(f'all_tng{args.boxsize}-match{args.match}.h5', key='s',mode='w')

    if args.sam and args.sim:
        df = join_sam_and_sim(df_sam, df_sim)
        df = remove_columns(df, ['GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP',
            'HalopropFoFIndex_DM', 'HalopropFoFIndex_FP'])
        df.to_hdf(f'all_tng{args.boxsize}-match-sam.h5', key='s', mode='w')
        if args.tests:
            print(df.columns.values)
            print('max center distance {}'.format(np.max(center_distance(df, type='gals'))))
            print('max center distance {}'.format(np.max(center_distance(df, type='halos'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates pandas datafame from TNG galaxy catalogs.')
    parser.add_argument('--sam', help='Path to the SAM files, will not read if not set')
    parser.add_argument('--sim', help='Path to simulation files, will not read if not set.')
    parser.add_argument('--dmo', help='Path to subfind DMO files, will match with simulation if set')
    parser.add_argument('--lgal',help='Path to Lgalaxies SAM file same as to simulation, DMO must also be set')
    parser.add_argument('--match', type=str, default = 'LHalo',
                        help='match the simulation to the dmo using LHalo or SubLink')
    Type = parser.add_mutually_exclusive_group()
    Type.add_argument('--halos', action='store_true', default=True, help='Create dataframe using halos (centrals only)')
    Type.add_argument('--gals', action='store_true', default=False, help='Create dataframe using galaxies (satellites included)')
    parser.add_argument('-s', '--snapshot', default=99, type=int, help='Snapshot to be converted to dataframe (default=99)')
    parser.add_argument('-t', '--tests', action='store_true', default=False, help='Make some plots to check that the matches are correct')
    parser.add_argument('-bs','--boxsize', default=100, type=int, 
                        help='the boxsize (50/100/300) used for labeling output')
    args = parser.parse_args()
    print(args)
    main(args)
