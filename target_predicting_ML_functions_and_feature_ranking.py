#!/usr/bin/env python
# coding: utf-8

# ## Target Predicting Functions: Linear Regression (LR), Support Vector Regressor(SVR) and Random Forest Regressor (RF). RF provides feature ranking using built-in ranking functions, as well as add-feature ranking algorithm. (From TNG-SAM v12 notebook)

# In[ ]:


import time
import numpy as np
import pandas as pd

import galsim #install with conda install -c conda_forge galsim

import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.colors as norm
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline #This allows one to build different steps together
from sklearn.preprocessing import StandardScaler, RobustScaler

from tqdm import tqdm 

# In[ ]:


def target_predicting_LR_function(X, y, params = {}):
    '''This function takes in a dataframe of X(features) and y (a target) and outputs predictions of the 
    target through Linear Regression model'''
    
        
    reg = LinearRegression() # choose the model: Linear Regression
    reg.fit(X, y) # fit the data to the chosen model
    # predict y (galaxy sizes) by using 5-fold cross-validation
    y_pred_LR = cross_val_predict(reg, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10))
    # find prediction scores of each of the cross validation fold
    scores_reg = cross_validate(reg, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10),
                               return_train_score=True)
    print("LR Accuracy: %0.2f (+/- %0.2f)" % (scores_reg['test_score'].mean(), scores_reg['test_score'].std() * 2))
    # Plot the figure
    fig_prediction, ax = plt.subplots(figsize=(5, 5))
    label = 'Train score={} \n Test score={}'.format(round(scores_reg['train_score'].mean(), 2),round(scores_reg['test_score'].mean(), 2) )
    ax = plt.subplot()
    ax.scatter(y, y_pred_LR, s=3, marker='.', alpha=0.7, label=label)
#     ax.set_xlim([0.0,100])
#     ax.set_ylim([0.0,100])
    ax.plot([0.0, 170], [0.0, 170], color = 'black', linewidth = 2)
    ax.set_title('Prediction vs True Value with LR, all features ')
    ax.set_xlabel('True Galaxy Size/Halo Size')
    ax.set_ylabel(' Predicted Galaxy Size/Halo Size')
    ax.legend(loc='lower right')

    fig_prediction.tight_layout()
#         plt.savefig('Linear Regression results.jpg', dpi=500)
    plt.show()
    
    return y_pred_LR, fig_prediction

# In[ ]:

def target_predicting_SVR_function(X, y, params = {}):
    '''This function takes in a dataframe of X(features) and y (a target) and outputs predictions of the 
    target through Support Vector Regressior model'''
    
        
    start_time = time.time()

    piped_model = make_pipeline(StandardScaler(), SVR()) #non linear so I can change the kernel

    piped_model.get_params() #this shows how we can access parameters both for the scaler and the classifier

    #optimizing SVR

    parameters = {'svr__kernel':['rbf'],               'svr__gamma':['scale'], 'svr__C':[1000],               }

    model = GridSearchCV(piped_model, parameters, cv = KFold(n_splits=3, shuffle=True),                      verbose = 3, n_jobs = -1, return_train_score=True)

    model.fit(X, y)

    print('Best params, best score:', "{:.4f}".format(model.best_score_),       model.best_params_)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the GridSearchCV for SVR: {elapsed_time:.3f} seconds")
    
    best_SVR=model.best_estimator_
    best_SVR_trainscore=model.cv_results_['mean_train_score'][model.best_index_]
    scores_SVR = cross_val_score(best_SVR, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10)) 
    y_pred_SVR = cross_val_predict(best_SVR, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10))
    
    print("SVR Accuracy: %0.2f (+/- %0.2f)" % (scores_SVR.mean(), scores_SVR.std() * 2))
    # Plot the figure
    fig_prediction, ax = plt.subplots(figsize=(5, 5))
    label = 'Train score={} \n Test score={}'.format(round(best_SVR_trainscore, 2),round(scores_SVR.mean(), 2) )    
    ax = plt.subplot()
    ax.scatter(y, y_pred_LR, s=3, marker='.', alpha=0.7, label=label)
#     ax.set_xlim([0.0,100])
#     ax.set_ylim([0.0,100])
    ax.plot([0.0, 170], [0.0, 170], color = 'black', linewidth = 2)
    ax.set_title('Prediction vs True Value with SVR & 33 features ')
    ax.set_xlabel('True Galaxy Size/Halo Size')
    ax.set_ylabel(' Predicted Galaxy Size/Halo Size')
    ax.legend(loc='lower right')

    fig_prediction.tight_layout()
#         plt.savefig('Linear Regression results.jpg', dpi=500)
    plt.show()
    
    return y_pred_SVR, fig_prediction


# In[ ]:


def run_random_search_pipeline(X, y):
    
    
#     n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]
#     grid = {'bootstrap': [True],
#                    'max_depth': [5, 10, None],
#     #                'max_features': ['auto', 'sqrt'],
#                    'min_samples_leaf': [1, 2, 4],
#     #                'min_samples_split': [2, 5, 10],
#                    'n_estimators': n_estimators}

    n_estimators = [20,50,100]
    grid = {'bootstrap': [True],
                   'max_depth': [5, 10, None],
    #                'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [2],
    #                'min_samples_split': [2, 5, 10],
                   'n_estimators': n_estimators}
    
#     pipeline = make_pipeline(RandomForestRegressor())
    model=RandomForestRegressor(random_state=0)

    reg_RF = GridSearchCV(model, grid, cv = KFold(n_splits=3, shuffle=True),                          verbose = 3, n_jobs = -1, return_train_score=True)
    reg_RF.fit(X, y)
#     estimator.get_params().keys()
    scores_lim = pd.DataFrame(reg_RF.cv_results_)

    scores_lim = scores_lim[['params','mean_test_score','std_test_score','mean_train_score', 'mean_fit_time']].sort_values(by = 'mean_test_score', ascending = False)
    
    return scores_lim['mean_test_score']


# In[ ]:


def calculate_r_score( X , y ,column_names):
    x_new = X[list(column_names)]
    # print(column_names)
    r_square = run_random_search_pipeline(x_new, y)
    
    return r_square.max()


# In[ ]:


def print_results(max_col_names, max_col_rscores):
    k= 1
    for i, j in zip(max_col_names, max_col_rscores):
        print("\t", k, i,j)
        k +=1


# In[ ]:


def calc_ith_iteration( X, y, max_col_names, max_col_rscores, orj_column_names, i):
    r_score_dict = {}
    for column_names in tqdm(orj_column_names): # orj_column_names = original column names
        # count += 1
        # print(count, max_col_names , column_names)
        feature_list = max_col_names + [column_names]
        r_score = calculate_r_score(X , y ,feature_list)
        r_score_dict[column_names] = r_score
    
    max_col_names.append(max(r_score_dict, key=r_score_dict.get))
    max_col_rscores.append(max(r_score_dict.values()))
    # print("asdfasd", max_col_names[len(max_col_names) - 1])
    
    orj_column_names.remove(max_col_names[len(max_col_names) - 1])
    
    return max_col_names, max_col_rscores, orj_column_names, r_score_dict




# In[ ]:


def target_predicting_RF_function(X, y, params = {}):
    '''This function takes in a dataframe of X(features) and y (a target) and outputs predictions of the 
    target through the Random Forest Regressor ML model'''
    
        
    regr_RF = RandomForestRegressor(random_state=0)

    # number of trees in random forest
    n_estimators = [20,50,100]
    # create random grid
    grid = {'bootstrap': [True],
                   'max_depth': [5, 10, None],
    #                'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
    #                'min_samples_split': [2, 5, 10],
                   'n_estimators': n_estimators}
    # Grid search of parameters
    rfr_grid = GridSearchCV(estimator = regr_RF, param_grid = grid, cv = KFold(n_splits=5, shuffle=True),                          verbose = 1, n_jobs = -1, return_train_score=True)

    rfr_grid.fit(X, y)

    print('Best params, best score:', "{:.4f}".format(rfr_grid.best_score_),         rfr_grid.best_params_)
    # define trainscore according to best model
    best_RF=rfr_grid.best_estimator_ # The index (of the cv_results_ arrays) which corresponds to the best candidate parameter setting.

    best_RF_trainscore=rfr_grid.cv_results_['mean_train_score'][rfr_grid.best_index_]

    # predict y (galaxy sizes) by using 5-fold cross-validation
    y_pred_RF = cross_val_predict(best_RF, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10))

    # find prediction scores of each of the cross validation fold
    scores_RF = cross_val_score(best_RF, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10))
    print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores_RF.mean(), scores_RF.std() * 2))
    
    # Plot the figure
    fig_prediction, ax = plt.subplots(figsize=(5, 5))

    label = 'Train score={} \n Test score={}'.format(round(best_RF_trainscore, 2),round(scores_RF.mean(), 2) )
    ax = plt.subplot()
    ax.scatter(y, y_pred_RF, s=3, marker='.', alpha=0.7, label=label)
#     ax.set_xlim([0.0,100])
#     ax.set_ylim([0.0,100])
    ax.plot([0.0, 170], [0.0, 170], color = 'black', linewidth = 2)
    ax.set_title('Prediction vs True Value with RF & all features ', fontsize=14)
    ax.set_xlabel('True Galaxy Size/Halo Size ', fontsize=12)
    ax.set_ylabel('Predicted Galaxy Size/Halo Size', fontsize=12)
    ax.legend(loc='lower right')

    fig_prediction.tight_layout()
    # plt.savefig('Prediction_vs_True.jpeg', dpi=500)

    plt.show()
        
    # Extract important features using built-in functions
    start_time = time.time()
    importances = rfr_grid.best_estimator_.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfr_grid.best_estimator_.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


    feature_names = X.columns

    forest_importances = pd.Series(importances, index=feature_names)

    fig_perm_import, [ax1, ax2] = plt.subplots(1,2, figsize=(15,5))
    fig_perm_import.suptitle('TNG-SAM Feature Importances  ', fontsize=16, fontweight='bold')

    forest_importances.plot.bar(yerr=std, ax=ax1)
    ax1.set_title("Feature importances using Mean Decrease in Impurity")
    ax1.set_ylabel("Mean decrease in impurity")

    ### Feature Importances Permutation

    start_time = time.time()
    result = permutation_importance(
        rfr_grid.best_estimator_, X, y, n_repeats=10, random_state=0, n_jobs=-1
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    forest_importances.plot.bar(yerr=result.importances_std, ax=ax2)
    ax2.set_title("Feature importances using Permutation Importances ")
    ax2.set_ylabel("Mean accuracy decrease")
    fig_perm_import.tight_layout()
    # plt.savefig("Feature_Importances_v2.jpeg", dpi=500)

    plt.show()

    df_perm_import=pd.DataFrame(forest_importances).sort_values(0, ascending=False)
    df_perm_import.to_csv('RF built in Feature Importances Permutation.csv', index=False)

    # Add-feature Ranking Method
    # Note: This step calls 4 funcitons: run_random_search_pipeline(X,y), calculate_r_score(column_names),
    # print_results(max_col_names, max_col_rscores), calc_ith_iteration(max_col_names, max_col_rscores, orj_column_names, i)

    column_names = X.columns
        
    # init
    
    max_col_names = [] # 
    max_col_rscores = []
    orj_column_names = list(X.columns)


    # main 
    for i in range(1, len(orj_column_names)+1):
        max_col_names, max_col_rscores, orj_column_names, r_score_dict = calc_ith_iteration(X, y, max_col_names, max_col_rscores, orj_column_names, i)

        print(f"{i}. iteration: ")
        print_results(max_col_names, max_col_rscores)
        if i==17: break

    df_max_r_scores = pd.DataFrame({
    'feature_number' : range(1,18),
    'features':max_col_names,
    'r_sq_score': max_col_rscores  })
    # Save the important feature ranking obtained by add-column method
    #df_max_r_scores.to_csv('Max r scores by add column method.csv', index=False)

    fig_add_feature=df_max_r_scores.plot(x='features', y='r_sq_score', rot=90, figsize=(7,5), use_index=True,
                        legend=False, grid=True, 
                        xticks=range(0,18))
    fig_add_feature.set_title(label= 'R square score by add-feature method',fontsize=16)
    fig_add_feature.set_xlabel('Features',fontsize=16)
    fig_add_feature.set_ylabel('R score',fontsize=16)
    
#     fig_add_feature.get_figure().savefig('R score by add-feature method.jpg', dpi=500)

    plt.show()
    
    return y_pred_RF, fig_prediction, fig_perm_import, df_perm_import, df_max_r_scores, fig_add_feature
  

