#!/bin/env python3

#from rdkit import Chem
#import re, tqdm, sys, umap, sklearn, lmfit, copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib
import matplotlib.font_manager

#import plotly.express as px
#import molplotly
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import sklearn.cluster as cluster

## make matplotlib look good
# plt.rc('font', size=8, family="serif")
# plt.rc('axes', titlesize=10, labelsize=10)
# plt.rc(['xtick', 'ytick'], labelsize=8)
# plt.rc('legend', fontsize=10)
# plt.rc('figure', titlesize=12)
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")

##############################################################
# This script is a case study about experimental study of Pd-catalyzed cross-coupling
# reactions between aryl bromide and alkenyl hydrazones, which generates skipped diene
# and vinyl cyclopropane as two major products. Different ligands can have different
# selectivity, reflecting on the ratio of SD:VCP obtained. This script tends to use 
# PCA combined with K-means clustering techinque to identify possible origin behind the 
# selectivity difference, which would be useful for future design. This script only shows
# the optimal result. Before I reached this result, I need to play with the number of clusters
# parameter and different PC combinations. You would also need to check this kind of clustering
# technique on the whole ligand set of kraken database instead of just the Buchwald-type ligand
# subset. For the descriptors, it's also necessary to check molecular fingerprints as the input.
#
# The "descriptos_edit.csv" file is the edited version from published JACS paper of kraken database.
# The "phoss_fliters.csv" file is from an ACS Catalysis paper published later following the original study.
# The "valid_result.csv" file is summarized result from experimental study.
##############################################################

# function to plot test sets of ligands without label
def plot_sns_test(df, xlabel, ylabel, clusterlabel, exemplars, figname, uniq_arr, buch_lgd, df_invent):
    f = plt.figure(figsize=(7,7))
    sns.set_theme(font="Arial", font_scale=1.2, style="ticks")
    colors = ['#D00000', '#90BE6D', '#4FADF0', '#D872EC', '#F48C06', '#43AA8B']

    colors_test = []
    for i in uniq_arr:
        colors_test.append(colors[i])
        
    rest_lgd_buch = [x for x in buch_lgd if x not in exemplars]
    df2 = df.loc[rest_lgd_buch]

    sns.scatterplot(x=xlabel, y=ylabel, data=df, s=100, alpha=0.6, style='class', 
                    hue=clusterlabel, markers={'buch': 'o', 'Thermo': 's', 'Kinetic': '^', 'SD':'s'}, palette=colors)
    
    sns.scatterplot(x=xlabel, y=ylabel, data=df.loc[exemplars], s=df_invent.loc[exemplars,'Ratio'] + 80,
                    alpha=1, style='class', edgecolor='black', linewidth=1.5,
                    hue=clusterlabel, markers={'Thermo': 's', 'Kinetic': '^', 'SD':'s'}, palette=colors_test, legend=False)

    plt.legend().set_visible(False)
    plt.tight_layout()
    
    # save figs
    fig_name = figname + '.png'
    fig_path = './' + fig_name
    plt.savefig(fig_path, dpi=1200)
    plt.show()

# function to plot test sets of ligands with label    
def plot_sns_test_label(df, xlabel, ylabel, clusterlabel, exemplars, figname, uniq_arr, buch_lgd, df_invent, df_kraken_edit):
    f = plt.figure(figsize=(7,7))
    sns.set_theme(font="Arial", font_scale=1.2, style="ticks")
    colors = ['#D00000', '#90BE6D', '#4FADF0', '#D872EC', '#F48C06', '#43AA8B']

    colors_test = []
    for i in uniq_arr:
        colors_test.append(colors[i])
        
    rest_lgd_buch = [x for x in buch_lgd if x not in exemplars]
    df2 = df.loc[rest_lgd_buch]
    
    sns.scatterplot(x=xlabel, y=ylabel, data=df, s=100, alpha=0.6, style='class', 
                    hue=clusterlabel, markers={'buch': 'o', 'Thermo': 's', 'Kinetic': '^', 'SD':'s'}, palette=colors)
    
    sns.scatterplot(x=xlabel, y=ylabel, data=df.loc[exemplars], s=df_invent.loc[exemplars,'Ratio'] + 80,
                    alpha=1, style='class', edgecolor='black', linewidth=1.5,
                    hue=clusterlabel, markers={'Thermo': 's', 'Kinetic': '^', 'SD':'s'}, palette=colors_test, legend=False)
  
    plt.legend().set_visible(False)
    
    for i in exemplars:
        plt.annotate(df_kraken_edit.loc[i, 'ID'], (df_invent.loc[i, xlabel], df_invent.loc[i, ylabel]))
    
    plt.tight_layout()
    
    # save figs
    fig_name = figname + '.png'
    fig_path = './' + fig_name
    plt.savefig(fig_path, dpi=1200)
    plt.show()

# function to plot validation sets of ligands without label    
def plot_sns_valid(df, xlabel, ylabel, clusterlabel, exemplars, figname, buch_lgd, df_invent):
    f = plt.figure(figsize=(7,7))
    sns.set_theme(font="Arial", font_scale=1.2, style="ticks")
    colors = ['#D00000', '#90BE6D', '#4FADF0', '#D872EC', '#F48C06', '#43AA8B']
    colors_valid = colors[1:3] + colors[-1:]
    rest_lgd_buch = [x for x in buch_lgd if x not in exemplars]
    df2 = df.loc[rest_lgd_buch]
    
    sns.scatterplot(x=xlabel, y=ylabel, data=df, s=100, alpha=0.6, style='class', 
                    hue=clusterlabel, markers={'buch': 'o', 'Thermo': 's', 'Kinetic': '^', 'SD':'s'}, palette=colors)
    
    sns.scatterplot(x=xlabel, y=ylabel, data=df.loc[exemplars], s=df_invent.loc[exemplars,'Ratio'] + 80,
                    alpha=1, style='class', edgecolor='black', linewidth=1.5,
                    hue=clusterlabel, markers={'Thermo': 'D', 'Kinetic': '^', 'SD':'D'}, palette=colors_valid, legend=False)
    
    plt.xlim(-6,4)
    plt.ylim(-9,2)
    plt.legend().set_visible(False)
    plt.tight_layout()
    
    # save figs
    fig_name = figname + '.png'
    fig_path = './' + fig_name
    plt.savefig(fig_path, dpi=1200)
    plt.show()


def main(NUM_EXEMPLARS):
    # read monophosphine ligand data from kraken database
    df_kraken = pd.read_csv('descriptors_edit.csv')
    df_kraken_edit = df_kraken.dropna()
    df_kraken_edit = df_kraken_edit.reset_index(drop=True)
    # save only the descriptor part into a new dataframe
    df_desp = df_kraken_edit.drop(['ID','smiles'],axis=1)
    # read ligand data from a filter csv file
    df_phosfilt = pd.read_csv('phoss_filters.csv')
    # extract only the Buchwald type ligands in kraken database
    df_phosfilt_clean = df_phosfilt.dropna(subset=['buch'])
    # get the index for Buchwald ligands
    buch_lgd = []
    for indx in range(df_phosfilt_clean.shape[0]):
        df_value = df_phosfilt_clean.loc[indx, 'buch'].astype(int)
        if(df_value == 1):
            buch_lgd.append(indx)

    df_desp_edit = df_desp.reset_index(drop=True)
    # PCA part for all Buchwald ligands in kraken database
    scaler = StandardScaler()
    X_scaled_buch = scaler.fit_transform(df_desp_edit.iloc[buch_lgd])
    pca = PCA(n_components=4) # use only the first four PCs
    Xpca_buch = pca.fit_transform(X_scaled_buch)
    print(Xpca_buch.shape)
    print(pca.explained_variance_ratio_)
    # save relevant columns into a dataframe later used in plotting
    df_plt_buch = pd.DataFrame(df_kraken_edit.loc[buch_lgd, 'smiles'])
    df_plt_buch["PC1"] = Xpca_buch[:, 0]
    df_plt_buch["PC2"] = Xpca_buch[:, 1]
    df_plt_buch["PC3"] = Xpca_buch[:, 2]
    df_plt_buch["PC4"] = Xpca_buch[:, 3]
    # print loadings of each descriptor within PC
    # change the integer 2 below based on which PC info you want to print
    sorted_list = sorted(enumerate(pca.components_[2]), key=lambda x: abs(x[1]), reverse=True)
    largest_indexes_pc1_top10 = [i for i, _ in sorted_list[:10]]
    for i in largest_indexes_pc1_top10:
        print(str(df_desp_edit.columns[i]) + ': ' + str(pca.components_[2][i]))
    # read csv file containing experimental results
    df_valid = pd.read_csv('valid_result.csv')
    # create list for experimental used ligand IDs
    invent_lgd_ID = [309,314,311,327,329,336,337,307,277,263,333,158,365,47,731,5,4,2,103,1,104,3,368,102,17,68]
    invent_lgd_new = []
    condition = df_kraken_edit['ID'].isin(invent_lgd_ID)
    sliced_df = df_kraken_edit[condition]
    for i in sliced_df.index:
        invent_lgd_new.append(i)
    element_to_remove = [16,67,153,326]
    invent_lgd_buch = [x for x in invent_lgd_new if x not in element_to_remove]
    
    # create dataframe for experimental used ligands only
    for i in df_plt_buch.index:
        df_plt_buch.loc[i,'ID'] = df_kraken_edit.loc[i,'ID']
    df_plt_buch['ID'] = df_plt_buch['ID'].astype(int)
    df_invent = pd.merge(df_plt_buch.loc[invent_lgd_buch], df_valid, on="ID")
    df_invent = df_invent.set_index(df_plt_buch.loc[invent_lgd_buch].index)

    # create list of indexes of traning ligands
    tested_lgd_ID = [1,2,3,4,5,17,68,103,365,314,327,336,104] # change based on experimental result
    tested_lgd = []
    condition_2 = df_kraken_edit['ID'].isin(tested_lgd_ID)
    sliced_df_tested = df_kraken_edit[condition_2]
    for i in sliced_df_tested.index:
        tested_lgd.append(i)
    tested_lgd_buch = [0,1,2,3,4,99,100,307,320,329,354]

    # create list of indexes of ligands leading to SD product exclusively
    sd_lgd_ID = [17,68,314,327,307,5,309,311] # change based on experimental result
    sd_lgd = []
    condition_sd = df_kraken_edit['ID'].isin(sd_lgd_ID)
    sliced_df_sd = df_kraken_edit[condition_sd]
    for i in sliced_df_sd.index:
        sd_lgd.append(i)
    element_to_remove = [16,67,153,326]
    sd_lgd_buch = [x for x in sd_lgd if x not in element_to_remove]

    # create list of indexes of ligands leading to SD:VCP around thermodynamic ratio
    thermo_lgd_ID = [1,2,3,4,102,47,329,263,277,337,368,731] # change based on experimental result
    thermo_lgd = []
    condition_thermo = df_kraken_edit['ID'].isin(thermo_lgd_ID)
    sliced_df_thermo = df_kraken_edit[condition_thermo]
    for i in sliced_df_thermo.index:
        thermo_lgd.append(i)

    # create list of indexes of ligands leading to VCP as the major product
    vcp_lgd_ID = [103,104,336,365] # change based on experimental result
    vcp_lgd = []
    condition_vcp = df_kraken_edit['ID'].isin(vcp_lgd_ID)
    sliced_df_vcp = df_kraken_edit[condition_vcp]
    for i in sliced_df_vcp.index:
        vcp_lgd.append(i)

    # create list of indexes of validation set of ligands 
    valid_lgd = [x for x in invent_lgd_buch if x not in tested_lgd_buch]
    
    # K-means clustering part
    kmeans = KMeans(n_clusters=NUM_EXEMPLARS,random_state=0)
    df_plt_buch["kmeans_cluster"] = kmeans.fit_predict(Xpca_buch[:, [1,2]])
    # assign class label based on ligand type
    df_plt_buch['class'] = 'buch'
    df_plt_buch.loc[sd_lgd_buch, 'class'] = 'SD'
    df_plt_buch.loc[thermo_lgd, 'class'] = 'Thermo'
    df_plt_buch.loc[vcp_lgd, 'class'] = 'Kinetic'
    # find unique indexes
    arr_test = np.array(df_plt_buch.loc[tested_lgd_buch, 'kmeans_cluster'])
    uniq_sort_arr = sorted(np.unique(arr_test), reverse=False)
    # rename PC columns for better plotting
    df_plt_buch = df_plt_buch.rename(columns={'pca_1':'PC1', 'pca_2':'PC2',
                                             'pca_3':'PC3', 'pca_4':'PC4',
                                             'kmeans_cluster':'cluster'})
    df_invent = df_invent.rename(columns={'pca_1':'PC1', 'pca_2':'PC2',
                                             'pca_3':'PC3', 'pca_4':'PC4',
                                             'kmeans_cluster':'cluster'})
    # plotting of traning and validation set of ligands in terms of PCs
    plot_sns_test(df_plt_buch, "PC2", "PC3","cluster", 
                  tested_lgd_buch, "pc2_pc3_train_6c_rg_1_s100_wolabel_SI", uniq_sort_arr, buch_lgd, df_invent)
    plot_sns_test_label(df_plt_buch, "PC2", "PC3","cluster", 
                  tested_lgd_buch, "pc2_pc3_train_6c_rg_1_s100_wlabel_SI", uniq_sort_arr, buch_lgd, df_invent, df_kraken_edit)
    plot_sns_valid(df_plt_buch, "PC2", "PC3", "cluster", 
                   valid_lgd, "pc2_pc3_valid_6c_rg_1_s100_wolabel_v4", buch_lgd, df_invent)

if __name__ == "__main__":
    NUM_EXEMPLARS = 6
    main(NUM_EXEMPLARS)

