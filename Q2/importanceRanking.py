# coding: utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

feature_cols = ['AltCountLineCode', 'CountInput', 'CountLineBlank', 'CountLineCodeDecl', 'CountLineComment',
                'CountLinePreprocessor', 'CountPath', 'CountStmt', 'CountStmtEmpty', 'Cyclomatic', 'CyclomaticStrict',
                'Knots', 'MinEssentialKnots', 'RatioCommentToCode', 'AltCountLineComment', 'CountLine', 'CountLineCode',
                'CountLineCodeExe', 'CountLineInactive', 'CountOutput', 'CountSemicolon', 'CountStmtDecl',
                'CountStmtExe', 'CyclomaticModified', 'Essential', 'MaxEssentialKnots', 'MaxNesting']

def importanceRanking(x, y, dataset):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(x.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_cols[indices[f]], importances[indices[f]]))


    plt.figure()
    plt.title("Feature importances - Dataset: " + dataset)
    plt.bar(range(x.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), indices)
    plt.xlim([-1, x.shape[1]])
    fig = plt.gcf()
    plt.draw()
    fig.savefig('result/' + filename + '-ImportanceRanking.png')
    plt.close()

if __name__ == "__main__":

    files = [
    #'../random_undersampling/xen_data_balanced.csv',
    #'../random_undersampling/glibc_data_balanced.csv',
    # '../random_undersampling/httpd_data_balanced.csv',
    #'../random_undersampling/kernel_data_balanced.csv',
    #'../random_undersampling/mozilla_data_balanced.csv',
    #'../unbalanced/glibc_data.csv',
    #'../unbalanced/httpd_data.csv',
    '../unbalanced/kernel_data.csv',
    '../unbalanced/mozilla_data.csv',
    '../unbalanced/xen_data.csv'
    ]

    for f in files:
        filename_w_ext = os.path.basename(f)
        filename, file_extension = os.path.splitext(filename_w_ext)

        data = pd.read_csv(f)
        x = data[feature_cols]
        y = data.Affected

        print("\n\n - Classificação de importância de features - Dataset: " + filename + "\n")
        importanceRanking(x, y, filename)
