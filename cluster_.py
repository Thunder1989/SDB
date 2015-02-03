from time import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import math
import random
import pylab as pl

#input1 = [i.strip().split('\\')[-2]+i.strip().split('\\')[-1][:-4] for i in open('sdh_pt_name').readlines()]
input1 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-4] for i in open('rice_pt_forsdh').readlines()]
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
label2 = input2[:,-1]
label = input4[:,-1]
vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#vc = TV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
fn = vc.fit_transform(input3).toarray()
fd = input4[:,[0,1,2,3,5,6,7]]
#n_class = len(np.unique(label))
#print n_class
#print np.unique(label)
print ct(label)

fold = 2
kf = StratifiedKFold(label, n_folds=fold)
#kf = KFold(len(label), n_folds=fold, shuffle=True)
'''
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1
'''

#clf = SVC(kernel='linear')
clf = RFC(n_estimators=50, criterion='entropy')
rounds = 4
print 'total rounds of', rounds
f = open('c_out','w')
acc_sum = []
for train, test in kf:
    ex_dict = dd(list)
    train_label = label[train]
    for i,j in zip(train_label,train):
        ex_dict[i].append(j)
    o_idx = []
    for v in ex_dict.values():
        random.shuffle(v)
        for i in range(rounds):
            if len(v) > i:
                o_idx.append(v[i])

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    for i in o_idx:
        train_id.append(i)
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds)
    acc_sum.append(acc)
print ct(train_label)
print len(train_label)
print 'acc using oracle ex:', np.mean(acc_sum), np.std(acc_sum)

acc_sum = []
for train, test in kf:
    rand_idx = random.sample(xrange(len(train)), len(o_idx))

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    for i in rand_idx:
        train_id.append(train[i])
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds)
    acc_sum.append(acc)
print ct(train_label)
print 'acc using random ex:', np.mean(acc_sum), np.std(acc_sum)
