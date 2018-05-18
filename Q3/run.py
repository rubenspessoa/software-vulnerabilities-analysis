# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from time import sleep
import copy
import os

nfolds = 10
feature_cols = [
  'AltCountLineCode',
  'CountInput',
  'CountLineBlank',
#'CountLineCodeDecl',
#'CountLineComment',
#'CountLinePreprocessor',
#'CountPath',
#'CountStmt',
#'CountStmtEmpty',
#'Cyclomatic',
#'CyclomaticStrict',
#'Knots',
#'MinEssentialKnots',
#'RatioCommentToCode',
#'AltCountLineComment',
  'CountLine',
  'CountLineCode',
  'CountLineCodeExe',
#'CountLineInactive',
  'CountOutput',
  'CountSemicolon',
#'CountStmtDecl'
  'CountStmtExe',
#'CyclomaticModified',
#'Essential',
#'MaxEssentialKnots'
#'MaxNesting'
]

class_names = ['NEUTRAL', 'VULNERABLE']

files = [
#'../random_undersampling/xen_data_balanced.csv',
#'../random_undersampling/glibc_data_balanced.csv',
#'../random_undersampling/httpd_data_balanced.csv',
#'../random_undersampling/kernel_data_balanced.csv'
#'../random_undersampling/mozilla_data_balanced.csv',
#'../unbalanced/glibc_data.csv',
#'../unbalanced/httpd_data.csv',
#'../unbalanced/kernel_data.csv',
#'../unbalanced/mozilla_data.csv',
'../unbalanced/xen_data.csv'
]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
Calcula as métricas Precision, Recall e Fmeasure
"""
def calculate_metrics(y_test, predictions, accuracy=0,model='',sets=''):
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    fscore = f1_score(y_test, predictions, average='macro')
    print(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(fscore) + ',' + model + ',' + sets)
    # print("Accuracy: " + str(accuracy))
    # print("Precision: " + str(precision))
    # print("Recall: " + str(recall))
    # print("F1 Score: " + str(fscore))

print('Accuracy,Precision,Recall,F1-Score,Model,Sets')

classifiers = [
    #KNeighborsClassifier(3),
    #SVC(gamma=2, C=1),
    #DecisionTreeClassifier(max_depth=5),
    #MLPClassifier(alpha=1),
    GaussianNB(),
    ]


for f in files:
    filename_w_ext = os.path.basename(f)
    filename, file_extension = os.path.splitext(filename_w_ext)

    df = pd.read_csv(f)

    x = df[feature_cols]
    y = df.Affected

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=1)

    for model in classifiers:
        best_model = 0
        best_acc = 0
        s = 0
        for train_index, test_index in kf.split(x, y):
            x_train = df.loc[train_index, feature_cols]
            y_train = df.loc[train_index, 'Affected']
            x_test = df.loc[test_index, feature_cols]
            y_test = df.loc[test_index, 'Affected']

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            aux = accuracy_score(y_test, y_pred)
            if aux  > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = aux
            s = s + aux
        # print(s, nfolds)
        acc = s / nfolds
        y_pred = best_model.predict(x_test)
        calculate_metrics(y_test, y_pred, acc, model=model.__class__.__name__, sets=filename)

        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        # plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, title= filename + ":" + model.__class__.__name__)
        fig = plt.gcf()
        # plt.show()
        plt.draw()
        fig.savefig('result/' + filename + '-' + model.__class__.__name__ + '.png')
        plt.close()

    # f, ax = plt.subplots(figsize=(10, 8))
    # corr = df.corr()
    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #             square=True, ax=ax)
    # fig = plt.gcf()
    # plt.show()
    # plt.draw()
    # fig.savefig('result/' + filename + '.png')
