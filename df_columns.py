import argparse
import pandas as pd


def main(args):
    df = pd.read_hdf(args.filename)
    print(df.shape)
    if args.describe:
        print(df[args.describe].describe())
    else:
        print(df.columns.values)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prints column names for dataframe saved in hdf5 format')
    parser.add_argument('filename', help='The filename whose column names will be printed')
    parser.add_argument('-d','--describe',
        help = 'run describe on the specified column')
    args = parser.parse_args()
    main(args)