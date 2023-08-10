import time
import numpy as np
import pandas as pd
import matplotlib.plt as pyplot
import matplotlib as mpl
import scipy.stats as sstats
import sklearn as skl
import sklearn.model_selection as sklms
import tqdm.tqdm as tqdm

import pysr.PySRRegressor as PySRRegressor
import galsim

import target_predicting_ML_functions_and_feature_ranking as functions
import RF_target_predicting_and_learning_curves_functions as tp_lc_functions

def fig_corr(df, kind='spearman')
    corr_df = df.corr(kind))
    plt.figure(figsize=(11,8))
    matrix = np.triu(corr_df) # take upper correlation matrix
    sns.heatmap(corr_df, mask=matrix)
    # plt.title("Spearman Correlation of TNG-SAM ", fontsize = 20)
    plt.ylabel("Features", fontsize = 15)
    # plt.savefig('TNG-SAM_Spearman_correlation_matrix_df_normalized_31.jpeg', dpi=500)
    plt.show()