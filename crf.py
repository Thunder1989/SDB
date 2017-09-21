'''
CRF based model for moal
'''
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF
from scikits.statsmodels.tools.tools import ECDF
from scipy.stats import t
from scipy.stats import mode
from collections import defaultdict as DD
from collections import Counter as CT

import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt

mapping = {1:'co2',2:'humidity',3:'pressure',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu',30:'pos',31:'power',32:'ctrl',33:'fan spd',34:'timer'}
input1 = np.genfromtxt('rice_hour_sdh', delimiter=',')
#input1 = np.genfromtxt('sdh_hour_soda', delimiter=',')
#input1 = np.genfromtxt('soda_hour_rice', delimiter=',')
input21 = np.genfromtxt('keti_hour_sum', delimiter=',')
#input21 = np.genfromtxt('rice_hour_soda', delimiter=',')
input3 = np.genfromtxt('sdh_hour_rice', delimiter=',')
input2 = np.vstack((input21,input3))
fd1 = input1[:, 0:-1]
fd2 = input2[:, 0:-1]
#fd3 = input3[:, 0:-1]
train_fd = fd1
test_fd = fd2
train_label = input1[:, -1]
test_label = input2[:, -1]
#print np.unique(train_label)
#print np.unique(test_label)
#print train_fd.shape
#print train_label.shape
#print test_fd.shape
#print test_label.shape


#swap the src and tgt
fd_tmp = train_fd
train_fd = test_fd
test_fd = fd_tmp
l_tmp = train_label
train_label = test_label
test_label = l_tmp


#step1 - train base models as noisy 'oracles' from src bldg
rf = RFC(n_estimators=100, criterion='entropy')
svm = SVC(kernel='rbf', probability=True)
lr = LR()
#clf = LinearSVC()
bl = [rf, lr, svm] #set of base learner
for b in bl:
    b.fit(train_fd, train_label) #train each base classifier

'''
step2
node features:
    name features
    confidence of oracles - oracle's prediction matched true y or not
edge features: (for each pair of vertices)
    true y agree or not
    each oracle's predictions agree or not
'''
#input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_soda').readlines()]
input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_sdh').readlines()]
#input1 = [i.strip().split('\\')[-1][:-5] for i in open('soda_pt_rice').readlines()]
label = test_label
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
cv = CV(analyzer='char_wb', ngram_range=(3,4))
test_fn = cv.fit_transform(name).toarray()

k = 3
fold = 10
kf = KFold(len(test_fn), n_folds=fold, shuffle=True)
nb = NN(n_neighbors=k, algorithm='ball_tree', metric='euclidean').fit(test_fn)
distances, indices = nb.kneighbors(test_fn)
acc_ = [] #acc in each run for averaging
oracle_confidence = []
edges = []
v_true_agreement = []
v_oracle_agreement = []
for i in xrange(len(test_fn)):
    #get the confidence of oracles as part of node features
    for f in bl:
        oracle_confidence.append(np.array(f.predict(test_fd[i]) == label[i]).astype(int))
    for idx in indices[i]:
        if [i, idx] not in edges and [idx, i] not in edges:
            edges.append([i, idx])
            v_true_agreement.append(np.array(label[i]==label[idx]).astype(int))
            for f in bl:
                v_oracle_agreement.append(np.array(f.predict(test_fd[i])==f.predict(test_fd[idx])).astype(int))

oracle_confidence = np.array(oracle_confidence)
oracle_confidence = np.reshape(oracle_confidence, (-1,len(bl)))
v_true_agreement = np.array(v_true_agreement)
v_oracle_agreement = np.array(v_oracle_agreement)
v_oracle_agreement = np.reshape(v_oracle_agreement, (-1,len(bl)))

node_features = np.array(np.hstack([test_fn, oracle_confidence]))
edge_features = np.array(np.hstack([v_true_agreement, v_oracle_agreement]))
print node_features.shape
print edges.shape
print edge_features.shape
