import site, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

site.addsitedir('/Users/ari/Code/')
import illustris_python as ilsim
import illustris_sam as ilsam

def generate_sub_volume_list(n):
    """generates a list of all the subvolumes"""
    subvolume_list = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                subvolume_list.append([i, j, k])

    return subvolume_list


def load_sam_halos(base_path, snapshot=99):
    """load sam halos returns a pandas dataframe"""
    sv_list = generate_sub_volume_list(5)
    sam_halos = ilsam.groupcat.load_snapshot_halos(base_path,
      snapshot, sv_list, matches=True)
    mass_fields = [
     'HalopropMass_ejected','HalopropMcooldot','HalopropMhot',
     'HalopropMstar_diffuse','HalopropMvir','HalopropZhot']
    df = pd.DataFrame(sam_halos)
    for f in mass_fields:
        df[f] = 1000000000.0 * df[f]

    return df


def load_sam_galaxies(base_path, snapshot=99):
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

    mass_fields = [
     'GalpropMBH','GalpropMH2','GalpropMHI','GalpropMHII',
     'GalpropMbulge','GalpropMcold','GalpropMvir','GalpropMstar',
     'GalpropMstar_merge','GalpropMstrip']
    for f in mass_fields:
        df[f] = 1000000000.0 * df[f]

    return df


def join_sam_gals_and_halos(df_gals, df_halos, centrals_only=True):
    """join the sam galaxy and halo properties into 1 df"""
    if centrals_only:
        df_centrals = df_gals.loc[df_gals['GalpropSatType'] == 0]
        df_centrals.pop('GalpropRfric')
    else:
        df_centrals = df_gals
    df = df_centrals.join(df_halos, on='GalpropHaloIndex_Snapshot')
    return df


def load_sim_halos(base_path, snapshot=99):
    """create a dataframe of TNG subfind halos"""
    fields = [
     'GroupFirstSub','GroupBHMass','GroupMass','GroupNsubs',
     'GroupPos','GroupVel','Group_M_Crit200',
     'Group_M_TopHat200','Group_R_Crit200',
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

    mass_fields = [
     'GroupBHMass','GroupMass','Group_M_Crit200',
     'Group_M_TopHat200','Group_R_Crit200','Group_R_TopHat200']
    for f in mass_fields:
        df[f] = df[f] * 10000000000.0 / h

    print(df.columns.values)
    return df


def load_sim_galaxies(base_path, snapshot=99):
    """create a dataframe of TNG subfind galaxies (subhalos)"""
    fields = [
     'SubhaloBHMass','SubhaloBHMdot','SubhaloGasMetallicity',
     'SubhaloGrNr','SubhaloHalfmassRadType','SubhaloMass',
     'SubhaloMassInMaxRadType','SubhaloParent','SubhaloPos','SubhaloSFRinRad',
     'SubhaloSpin','SubhaloStarMetallicity','SubhaloVmax','SubhaloVmaxRad']
    subs = ilsim.groupcat.loadSubhalos(base_path, snapshot, fields=fields)
    h = 0.704
    radii = subs.pop('SubhaloHalfmassRadType')
    Rgas = radii[:, 0] / h
    Rstar = radii[:, 4] / h
    masses = subs.pop('SubhaloMassInMaxRadType')
    Mgas = masses[:, 0] * 10000000000.0 / h
    Mstar = masses[:, 4] * 10000000000.0 / h
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

    mass_fields = [
     'SubhaloBHMass', 'SubhaloMass']
    for f in mass_fields:
        df[f] = df[f] * 10000000000.0 / h

    print(df.columns.values)
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
        p1name = [
         'GalpropX', 'GalpropY', 'GalpropZ']
        p2name = ['GroupX', 'GroupY', 'GroupZ']
    elif type == 'sim':
        p1name = [
         'GroupX', 'GroupY', 'GroupZ']
        p2name = ['SubhaloX', 'SubhaloY', 'SubhaloZ']
    elif type == 'halos':
        p1name = [
         'GalpropX', 'GalpropY', 'GalpropZ']
        p2name = ['SubhaloX', 'SubhaloY', 'SubhaloZ']
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
    """create a dataframe from TNG subfind or TNG-SAM catalogs"""
    if args.tests:
        (f, axs) = plt.subplots(2, 2)
    if args.sam:
        df_sam_gals = load_sam_galaxies((args.sam), snapshot=(args.s))
        df_sam_halos = load_sam_halos((args.sam), snapshot=(args.s))
        df_sam = join_sam_gals_and_halos(df_sam_gals, df_sam_halos)
        if args.tests:
            print(df_sam.columns.values)
            axs[0][0].hist((df_sam['GalpropMvir'] - df_sam['HalopropMvir']), bins=100, density=True)
        df_sam = remove_columns(df_sam, [
         'GalpropBirthHaloID','GalpropSatType','GalpropHaloIndex',
         'GalpropHaloIndex_Snapshot','GalpropRootHaloID','GalpropMvir',
         'HalopropHaloID','HalopropIndex','HalopropIndex_Snapshot',
         'HalopropRedshift','HalopropRockstarHaloID','HalopropRootHaloID',
         'HalopropSnapNum'])
        print(df_sam.columns.values)
    if args.sim:
        df_sim_halos = load_sim_halos(args.sim)
        df_sim_gals = load_sim_galaxies(args.sim)
        df_sim = join_sim_gals_and_halos(df_sim_gals, df_sim_halos)
        df_sim = remove_columns(df_sim, [
         'GroupFirstSub', 'SubhaloGrNr', 'SubhaloParent'])
        print('sim catalog: ', df_sim.shape)
        if args.tests:
            print(list(df_sim))
            axs[0][1].hist(center_distance(df_sim, type='sim'), bins=100, density=True)
        print(df_sim.columns.values)
        df_sim.to_hdf('tng-sim.h5', key='s', mode='w')
    if args.sam:
        if args.sim:
            df = join_sam_and_sim(df_sam, df_sim)
            df = remove_columns(df, [
             'GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP',
             'HalopropFoFIndex_DM', 'HalopropFoFIndex_FP'])
            df.to_hdf('tng-match.h5', key='s', mode='w')
            if args.tests:
                print(df.columns.values)
                axs[1][0].hist(center_distance(df, type='gals'), bins=100, density=True)
                axs[1][1].hist(center_distance(df, type='halos'), bins=100, density=True)
            if args.sam:
                df_sam = remove_columns(df_sam, [
                 'GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP',
                 'HalopropFoFIndex_DM', 'HalopropFoFIndex_FP'])
                df_sam.to_hdf('tng-sam.h5', key='s', mode='w')
        if args.tests:
            plt.savefig('test.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates pandas datafame from TNG galaxy catalogs')
    parser.add_argument('--sam', help='Path to the SAM files, will not read if not set')
    parser.add_argument('--sim', help='Path to simulation files, will not read if not set.')
    parser.add_argument('--dmo', help='Path to subfind DMO files, will match with SAM if set')
    Type = parser.add_mutually_exclusive_group()
    Type.add_argument('--halos', action='store_true', default=True, help='Create dataframe using halos (centrals only)')
    Type.add_argument('--gals', action='store_true', default=False, help='Create dataframe using galaxies (satellites included)')
    parser.add_argument('-s', '-snap', default=99, type=int, help='Snapshot to be converted to dataframe (default=99)')
    parser.add_argument('-t', '--tests', action='store_true', default=False, help='Make some plots to check that the matches are correct')
    args = parser.parse_args()
    print(args)
    main(args)
