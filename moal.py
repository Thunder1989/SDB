'''
multi-oracle acitve learning baseline - kdd'09 method
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
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color
from scikits.statsmodels.tools.tools import ECDF
from scipy.stats import t
from scipy.stats import mode

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


#step1: train base models as 'oracles' from src bldg
'''
input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
input2 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
fd = input2[:,[0,1,2,3,5,6,7]]
label = input2[:,-1]
class_ = np.unique(label)
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
cv = CV(analyzer='char_wb', ngram_range=(3,4))
#fn = cv.fit_transform(name).toarray()
cv.fit(name)

input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
fn = cv.transform(name).toarray()
'''

rf = RFC(n_estimators=100, criterion='entropy')
svm = SVC(kernel='rbf', probability=True)
lr = LR()
#clf = LinearSVC()
R = DD()
bl = [rf, lr, svm] #set of base learner
for b in bl:
    b.fit(train_fd, train_label) #train each base classifier
    #print b
    print b.score(test_fd, test_label)
    R[b] = [1,0] #initial rewards for each 'oracle'

'''
cm_ = CM(test_label, preds)
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10)
cm_cls = np.unique(np.hstack((test_label,preds)))
cls = []
for c in cm_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Mn Confusion matrix (%.3f)'%acc)
pl.show()
'''

#step2: confidence estimation for each oracle and apply to bldg2
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
#test_fd = test_fn

fold = 10
kf = KFold(len(test_fn), n_folds=fold, shuffle=True)
iteration = 100
#lr_ = LR() #clf for use
lr_ = SVC(kernel='linear', probability=True)
CI = DD() #confidence level for each oracle
acc_ = [[] for i in range(iteration)] #acc in each run for averaging
for train, test in kf:
    fd_ = []
    label_ = []
    #TBD: randomly pick two examples from diff classes as starting
    fd_.append(train[0])
    label_.append(test_label[train[0]])
    train = train[1:]
    #needs one more ex from a diff class
    tmp = 0
    for i in train:
        if test_label[i] == label_[0]:
            continue
        else:
            fd_.append(i)
            label_.append(test_label[i])
            tmp = i
            break
    train = train[train!=tmp]

    #pick the most uncertain ex based on f's posterior prob. and ask for 'oracles'
    for itr in range(iteration):
        lr_.fit(test_fn[fd_], label_)
        label_pr = np.sort(lr_.predict_proba(test_fn[train])) #sort in ascending order
        rank = []
        for i,pr in zip(train, label_pr):
            rank.append([i,pr[-1]])
        rank = sorted(rank, key=lambda x: x[-1])
        idx = rank[0][0]
        #compute CI for each oracle
        for b in bl:
            r = R[b]
            n = len(r)
            cv = t.ppf(0.975, n-1)
            CI[b] = np.mean(r) + cv*np.std(r)/np.sqrt(n)

        epsilon = 0.7*max(CI.values())
        preds = []
        for b in bl:
            if CI[b] >= epsilon:
                preds.append(b.predict(test_fd[idx]))
        #print 'predicted label from NO', preds
        y_ = mode(preds, axis=None)[0][0]
        #print 'major', y_
        for b in bl:
            if CI[b] >= epsilon:
                if b.predict(test_fd[idx]) == y_:
                    R[b].append(1)
                else:
                    R[b].append(0)
        fd_.append(idx)
        label_.append(y_)
        lr_.fit(test_fn[fd_], label_)
        acc_[itr].append(lr_.score(test_fn[test], test_label[test])) #sort in ascending order
        train = train[train!=idx]
print 'ave_acc', [np.mean(i) for i in acc_]
print R
print CI
