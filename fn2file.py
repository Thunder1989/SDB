'''
write name features to file
'''
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from sklearn import tree
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color
from scikits.statsmodels.tools.tools import ECDF

import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt

input1 = np.genfromtxt('rice_hour_sdh', delimiter=',')
#input1 = np.genfromtxt('sdh_hour_soda', delimiter=',')
#input1 = np.genfromtxt('soda_hour_rice', delimiter=',')
input21 = np.genfromtxt('keti_hour_sum', delimiter=',')
#input21 = np.genfromtxt('rice_hour_soda', delimiter=',')
input3 = np.genfromtxt('sdh_hour_rice', delimiter=',')
input2 = np.vstack((input21,input3))
fd1 = input1[:,0:-1]
fd2 = input2[:,0:-1]
fd3 = input3[:,0:-1]
#train_fd = np.hstack((fd1,fd2))
train_fd = fd1
test_fd = fd2
train_label = input1[:,-1]
#test_label = np.hstack((input2[:,-1],input3[:,-1]))
test_label = input2[:,-1]

input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_sdh').readlines()]
input2 = np.genfromtxt('rice_hour_sdh', delimiter=',')
label = input2[:,-1]
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
cv = CV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()

out = np.hstack([label[:, None], fn])
np.savetxt("fn_rice.csv", out, delimiter=",")
