import site 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

import illustris_python as ilsim
site.addsitedir('/Users/ari/Code/')
#import illustris_sam as ilsam

hubble_value = 0.74
tng_mass_unit = 1.e10
scsam_mass_unit = 1.e9
scale_factor = 1.0 #need to set based on snapshot z 

def generate_sub_volume_list(n):
    """generates a list of all the subvolumes"""
    subvolume_list = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                subvolume_list.append([i, j, k])

    return subvolume_list


def load_scsam_halos(base_path, snapshot=99):
    """load sam halos returns a pandas dataframe"""
    sv_list = generate_sub_volume_list(5)
    sam_halos = ilsam.groupcat.load_snapshot_halos(base_path,
      snapshot, sv_list, matches=True)
    mass_fields = [
     'HalopropMass_ejected','HalopropMcooldot','HalopropMhot',
     'HalopropMstar_diffuse','HalopropMvir','HalopropZhot']
    df = pd.DataFrame(sam_halos)
    for f in mass_fields:
        df[f] = 1000_000_000.0 * df[f]

    return df


def load_scsam_galaxies(base_path, snapshot=99):
    """load sam galaxies returns a pandas dataframe"""
    sv_list = generate_sub_volume_list(5)
    sam_gals = ilsam.groupcat.load_snapshot_subhalos(base_path,
      snapshot, sv_list, matches=True)
    pos = sam_gals.pop('GalpropPos')
    vel = sam_gals.pop('GalpropVel')
    df = pd.DataFrame(sam_gals)
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    for i in range(3):
        df['Galprop' + pos_field[i]] = pos[:, i]
        df['Galprop' + vel_field[i]] = vel[:, i]

    mass_fields = ['GalpropMBH','GalpropMH2','GalpropMHI','GalpropMHII',
        'GalpropMbulge','GalpropMcold','GalpropMvir','GalpropMstar',
        'GalpropMstar_merge','GalpropMstrip']
    for f in mass_fields:
        df[f] = scsam_mass_unit * df[f]

    return df


def join_scsam_gals_and_halos(df_gals, df_halos, centrals_only=True):
    """join the sam galaxy and halo properties into 1 df"""
    if centrals_only:
        df_centrals = df_gals.loc[df_gals['GalpropSatType'] == 0]
        df_centrals.pop('GalpropRfric')
    else:
        df_centrals = df_gals
    df = df_centrals.join(df_halos, on='GalpropHaloIndex_Snapshot')
    return df

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

def load_sim_halos(base_path, snapshot=99):
    """create a dataframe of TNG subfind halos"""
    fields = ['GroupFirstSub','GroupBHMass','GroupMass','GroupNsubs',
        'GroupPos','GroupVel','Group_M_Crit200', 'Group_M_TopHat200','Group_R_Crit200',
        'Group_R_TopHat200']
    halos = ilsim.groupcat.loadHalos(base_path, snapshot, fields=fields)
    pos = halos.pop('GroupPos')
    vel = halos.pop('GroupVel')
    df = pd.DataFrame(halos)
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    h = 0.704
    for i in range(3):
        df['Group' + pos_field[i]] = pos[:, i] * 0.001 / h
        df['Group' + vel_field[i]] = vel[:, i]

    mass_fields = ['GroupBHMass','GroupMass','Group_M_Crit200', 'Group_M_TopHat200']
    for f in mass_fields:
        df[f] = df[f] * tng_mass_unit / hubble_value
    radii_fields = ['Group_R_Crit200','Group_R_TopHat200']
    for f in radii_fields:
            df[f] = df[f] / hubble_value

    print(df.columns.values)
    return df


def load_sim_galaxies(base_path, snapshot=99):
    """create a dataframe of TNG subfind galaxies (subhalos)"""
    fields = ['SubhaloBHMass','SubhaloBHMdot','SubhaloGasMetallicity',
        'SubhaloGrNr','SubhaloHalfmassRadType','SubhaloMass',
        'SubhaloMassInMaxRadType','SubhaloParent','SubhaloPos','SubhaloSFRinRad',
        'SubhaloSpin','SubhaloStarMetallicity','SubhaloVmax','SubhaloVmaxRad']
    subs = ilsim.groupcat.loadSubhalos(base_path, snapshot, fields=fields)
    radii = subs.pop('SubhaloHalfmassRadType')
    Rgas = radii[:, 0] / hubble_value
    Rstar = radii[:, 4] / hubble_value
    masses = subs.pop('SubhaloMassInMaxRadType')
    Mgas = masses[:, 0] * tng_mass_unit / hubble_value
    Mstar = masses[:, 4] * tng_mass_unit / hubble_value
    pos = subs.pop('SubhaloPos')
    spins = subs.pop('SubhaloSpin')
    df = pd.DataFrame(subs)
    df['SubhaloRgas'] = Rgas
    df['SubhaloRstar'] = Rstar
    df['SubhaloMgas'] = Mgas
    df['SubhaloMstar'] = Mstar
    pos_field = ['X', 'Y', 'Z']
    spin_field = ['Jx', 'Jy', 'Jz']
    for i in range(3):
        df['Subhalo' + pos_field[i]] = pos[:, i] * 0.001 / h
        df['Subhalo' + spin_field[i]] = spins[:, i] / h

    mass_fields = ['SubhaloBHMass', 'SubhaloMass']
    for f in mass_fields:
        df[f] = df[f] * tng_mass_unit / hubble_value

    print(df.columns.values)
    return df

def load_sim_dmo(base_path,snapshot=99):
    '''load DMO halos from Illustris-TNG simulations (group and subhalo)'''
    #first load group catalog
    fields = ['GroupFirstSub','GroupMass','GroupNsubs','GroupPos','GroupVel',
        'Group_M_Crit200','Group_M_Crit500','Group_M_Mean200','Group_M_TopHat200',
        'Group_R_Crit200','Group_R_Crit500','Group_R_Mean200','Group_R_TopHat200']
    halos = ilsim.groupcat.loadHalos(base_path, snapshot, fields=fields)
    pos = halos.pop('GroupPos')
    vel = halos.pop('GroupVel')
    df_halos = pd.DataFrame(halos)
    pos_field = ['X', 'Y', 'Z']
    vel_field = ['Vx', 'Vy', 'Vz']
    spin_field = ['Jx', 'Jy', 'Jz'] #for use with Subhalos
    for i in range(3):
        df_halos['Group' + pos_field[i]] = pos[:, i] * 0.001 / hubble_value
        df_halos['Group' + vel_field[i]] = vel[:, i]
 
    #second load subhalo catalog
    fields = ['SubhaloHalfmassRad', 'SubhaloMass', 'SubhaloMassInMaxRad', 
        'SubhaloParent', 'SubhaloPos', 'SubhaloSpin', 'SubhaloVel', 'SubhaloVelDisp', 
        'SubhaloVmax', 'SubhaloVmaxRad']
    subs = ilsim.groupcat.loadSubhalos(base_path, snapshot, fields=fields)
    pos = subs.pop('SubhaloPos')
    vel = subs.pop('SubhaloVel')
    spins = subs.pop('SubhaloSpin')
    df_subs = pd.DataFrame(subs)
    for i in range(3):
        df_subs['Subhalo' + pos_field[i]] = pos[:, i] * 0.001 / hubble_value
        df_subs['Subhalo' + vel_field[i]] = vel[:, i]
        df_subs['Subhalo' + spin_field[i]] = spins[:, i] / hubble_value
    #join the two
    df = df_halos.join(df_subs, on='GroupFirstSub', rsuffix='_s')
    mass_fields = ['Group_M_Crit200','Group_M_Crit500','Group_M_Mean200','Group_M_TopHat200',
        'GroupMass', 'SubhaloMass','SubhaloMassInMaxRad']
    radii_fields = ['Group_R_Crit200','Group_R_Crit500','Group_R_Mean200','Group_R_TopHat200']
    for f in mass_fields:
        df[f] = df[f] * tng_mass_unit / hubble_value
    for f in radii_fields:
        df[f] = df[f] / hubble_value
    return df

def join_sim_gals_and_halos(df_subs, df_halos, centrals_only=True):
    """join the sim galaxy and halo properties into 1 df"""
    if centrals_only:
        df_centrals = df_subs.loc[df_subs['SubhaloParent'] == 0]
    else:
        df_centrals = df_gals
    df = df_centrals.join(df_halos, on='SubhaloGrNr', rsuffix='_h')
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
        df_sam_gals = load_sam_galaxies((args.sam), snapshot=(args.s))
        df_sam_halos = load_sam_halos((args.sam), snapshot=(args.s))
        df_sam = join_sam_gals_and_halos(df_sam_gals, df_sam_halos)
        if args.tests:
            print(df_sam.columns.values)
            axs[0][0].hist((df_sam['GalpropMvir'] - df_sam['HalopropMvir']), bins=100, density=True)
        df_sam = remove_columns(df_sam, ['GalpropBirthHaloID','GalpropSatType',
        'GalpropHaloIndex', 'GalpropHaloIndex_Snapshot','GalpropRootHaloID','GalpropMvir',
        'HalopropHaloID','HalopropIndex','HalopropIndex_Snapshot','HalopropRedshift',
        'HalopropRockstarHaloID','HalopropRootHaloID', 'HalopropSnapNum'])
    if args.sim:
        df_sim_halos = load_sim_halos(args.sim)
        df_sim_gals = load_sim_galaxies(args.sim)
        df_sim = join_sim_gals_and_halos(df_sim_gals, df_sim_halos)
        df_sim = remove_columns(df_sim, ['GroupFirstSub', 'SubhaloGrNr', 'SubhaloParent'])
        print('sim catalog: ', df_sim.shape)
        if args.tests:
            print(df_sim.column.values)
            axs[0][1].hist(center_distance(df_sim, type='sim'), bins=100, density=True)
        df_sim.to_hdf(f'tng{args.boxsize}-sim.h5', key='s', mode='w')
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
        df.to_hdf(f'tng{args.boxsize}-lgal.h5',  key='s', mode='w')
    elif args.dmo:
        df_dmo = load_sim_dmo(args.dmo)
        df_dmo = remove_columns(df_dmo, ['GroupFirstSub'])
        df_dmo.to_hdf(f'tng{args.boxsize}-dmo.h5', key='s', mode='w')
    if args.sam:
        if args.sim:
            df = join_sam_and_sim(df_sam, df_sim)
            df = remove_columns(df, ['GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP',
                'HalopropFoFIndex_DM', 'HalopropFoFIndex_FP'])
            df.to_hdf('tng-match.h5', key='s', mode='w')
            if args.tests:
                print(df.columns.values)
                print('max center distance {}'.format(np.max(center_distance(df, type='gals'))))
                print('max center distance {}'.format(np.max(center_distance(df, type='halos'))))
            if args.sam:
                df_sam = remove_columns(df_sam, ['GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP',
                 'HalopropFoFIndex_DM', 'HalopropFoFIndex_FP'])
                df_sam.to_hdf('tng-sam.h5', key='s', mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates pandas datafame from TNG galaxy catalogs.')
    parser.add_argument('--sam', help='Path to the SAM files, will not read if not set')
    parser.add_argument('--sim', help='Path to simulation files, will not read if not set.')
    parser.add_argument('--dmo', help='Path to subfind DMO files, will match with SAM if set')
    parser.add_argument('--lgal',help='Path to Lgalaxies SAM file same as to simulation, DMO must also be set')
    Type = parser.add_mutually_exclusive_group()
    Type.add_argument('--halos', action='store_true', default=True, help='Create dataframe using halos (centrals only)')
    Type.add_argument('--gals', action='store_true', default=False, help='Create dataframe using galaxies (satellites included)')
    parser.add_argument('-s', '--snaphot', default=99, type=int, help='Snapshot to be converted to dataframe (default=99)')
    parser.add_argument('-t', '--tests', action='store_true', default=False, help='Make some plots to check that the matches are correct')
    parser.add_argument('-bs','--boxsize', default=100, type=int, 
                        help='the boxsize (50/100/300) used for labeling output')
    args = parser.parse_args()
    print(args)
    main(args)
