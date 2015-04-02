import numpy as np
import math
import random
import re
from time import time
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn import tree
from sklearn.preprocessing import normalize

input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
label2 = input2[:,-1]
label = input4[:,-1]
#input3, label = shuffle(input3, label)
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))

vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#vc = TV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
fn = vc.fit_transform(name).toarray()
fd = input4[:,[0,1,2,3,5,6,7]]

fold = 3
kf = StratifiedKFold(label, n_folds=fold, shuffle=True)
#kf = KFold(len(label), n_folds=fold, shuffle=True)
clf = RFC(n_estimators=100, criterion='entropy')
rounds = 1
acc_sum = [[] for i in range(fold)]
for train, test in kf:
    train_fn = fn[train]
    #n_class = len(np.unique(label[train]))
    n_class = 32
    g = GMM(n_components=n_class, covariance_type='spherical', init_params='wmc', n_iter=100)
    g.fit(train_fn)
    #g.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
    preds = g.predict(train_fn)
    #prob = np.sort(g.predict_proba(train_fd))
    acc_sum[0].append(NMI(label[train], preds))

    k = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
    k.fit(train_fn)
    acc_sum[1].append(NMI(label[train], k.labels_))

    d = DPGMM(n_components=50, covariance_type='full')
    d.fit(train_fn)
    print '# of M', d.n_components
    preds = d.predict(train_fn)
    acc_sum[2].append(NMI(label[train], preds))

print np.mean(acc_sum,axis=1)
