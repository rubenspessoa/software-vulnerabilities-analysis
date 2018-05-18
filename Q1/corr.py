import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
import os

feature_cols = ['AltCountLineCode', 'CountInput', 'CountLineBlank', 'CountLineCodeDecl', 'CountLineComment',
                'CountLinePreprocessor', 'CountPath', 'CountStmt', 'CountStmtEmpty', 'Cyclomatic', 'CyclomaticStrict',
                'Knots', 'MinEssentialKnots', 'RatioCommentToCode', 'AltCountLineComment', 'CountLine', 'CountLineCode',
                'CountLineCodeExe', 'CountLineInactive', 'CountOutput', 'CountSemicolon', 'CountStmtDecl',
                'CountStmtExe', 'CyclomaticModified', 'Essential', 'MaxEssentialKnots', 'MaxNesting']

class_names = ['NEUTRAL', 'VULNERABLE']

files = ['../random_undersampling/xen_data_balanced.csv',
'../random_undersampling/glibc_data_balanced.csv',
'../random_undersampling/httpd_data_balanced.csv',
'../random_undersampling/kernel_data_balanced.csv',
'../random_undersampling/mozilla_data_balanced.csv',
'../unbalanced/glibc_data.csv',
'../unbalanced/httpd_data.csv',
'../unbalanced/kernel_data.csv',
'../unbalanced/mozilla_data.csv',
'../unbalanced/xen_data.csv']

for f in files:
    filename_w_ext = os.path.basename(f)
    filename, file_extension = os.path.splitext(filename_w_ext)

    df = pd.read_csv(f)
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    fig = plt.gcf()
    plt.show()

    plt.draw()
    fig.savefig('corr/' + filename + '.png')
