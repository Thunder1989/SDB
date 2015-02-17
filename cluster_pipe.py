import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import pylab as pl
from time import time
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
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

vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1)
#vc = TV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
fn = vc.fit_transform(name).toarray()
fd = input4[:,[0,1,2,3,5,6,7]]
#n_class = len(np.unique(label))
#print n_class
#print np.unique(label)
#print 'class count from groud truth labels:\n',ct(label)
#kmer = vc.get_feature_names()
#idf = zip(kmer, vc._tfidf.idf_)
#idf = sorted(idf, key=lambda x: x[-1], reverse=True)
#print idf[:20]
#print idf[-20:]
#print vc.get_feature_names()

fold = 2
rounds = 1
clf = SVC(kernel='linear')
#clf = RFC(n_estimators=50, criterion='entropy')
kf = StratifiedKFold(label, n_folds=fold, shuffle=True)
#kf = KFold(len(label), n_folds=fold, shuffle=True)
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
#X = StandardScaler().fit_transform(fn)
acc_sum = []
for train, test in kf:
    print 'class count of true labels on cluster training ex:\n', ct(label[train])
    train_fd = fn[train]
    n_class = len(np.unique(label[train]))
    #print '# of training class', n_class
    c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
    c.fit(train_fd)
    dist = np.sort(c.transform(train_fd))
    ex = dd(list)
    for i,j,k in zip(c.labels_, train, dist):
        ex[i].append([j,k[0]])
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1])
    km_idx = []
    print 'initial exs from k clusters centroid=============================='
    for k,v in ex.items():
        for i in range(rounds):
            if len(v)>i:
                km_idx.append(v[i][0])
                print k,label[v[i][0]],input3[v[i][0]]
    print len(km_idx), 'training examples'
    test_fn = fn[test]
    test_label = label[test]

    for rr in range(n_class):
        train_fn = fn[km_idx]
        train_label = label[km_idx]
        print ct(train_label)
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        preds_c = clf.predict(fn[train]) #predict labels for cluster learning set
        acc = accuracy_score(test_label, preds_fn)
        acc_ = accuracy_score(label[train], preds_c)
        print 'acc on test set', acc
        print 'acc on cluster set', acc_
        print 'class count of predicted labels on cluster learning ex:\n', ct(preds_c)
        acc_sum.append(acc)
        sub_pred = dd(list)
        for i,j in zip(c.labels_, preds_c):
            sub_pred[i].append(j)
        rank = []
        for k,v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i/float(max(count)) for i in count]
            H = np.sum(-math.pi*math.log(p, 2) for p in count if p!=0)
            rank.append([k,len(v),H])
        rank = sorted(rank, key=lambda x: x[-1], reverse=True)
        #print rank
        print 'adding exs itr', r, '>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        idx = rank[0][0] #pick the id of the 1st cluster on the rank
        c_id = [i[0] for i in ex[idx]]
        sub_label = sub_pred[idx]
        sub_fn = fn[c_id]
        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))
        ex_ = dd(list)
        for i,j,k in zip(c_.labels_, c_id, dist):
            ex_[i].append([j,k[0]])
        for i,j in ex_.items():
            ex_[i] = sorted(j, key=lambda x: x[-1])
        for k,v in ex_.items():
            for i in range(rounds):
                if len(v)>i:
                    if v[i][0] not in km_idx:
                        km_idx.append(v[i][0])
                        print '>',k,label[v[i][0]],input3[v[i][0]]
        print len(km_idx), 'training examples'
        '''
        train_fn = fn[km_idx]
        train_label = label[km_idx]
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        preds_train = clf.predict(fn[train])
        acc = accuracy_score(test_label, preds_fn)
        acc_ = accuracy_score(label[train], preds_train)
        print 'acc on test set', acc
        print 'acc on cluster set', acc_
        print 'class count of predicted labels on cluster training ex:\n', ct(preds_train)
        '''
    print '---------------------------------------------'
    print '---------------------------------------------'

#print len(train_label), 'training examples'
print 'class count of clf training ex:', ct(train_label)
print 'average acc:', np.mean(acc_sum), np.std(acc_sum)
cm_ = CM(test_label, preds_fn)
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
cm_cls =np.unique(np.hstack((test_label,preds_fn)))
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
acc_sum = []
for train, test in kf:
    train_label = label[train]
    #print len(np.unique(train_label))
    ex = dd(list)
    oc_idx = []
    for i,j in zip(train_label,train):
        ex[i].append(j)
    for v in ex.values():
        train_fd = fn[v]
        n = 1
        if len(v)>=10:
            #print mapping[k], len(v)
            n = len(v)/10
        c = KMeans(init='k-means++', n_clusters=n, n_init=10)
        c.fit(train_fd)
        rank = dd(list)
        for i,j,k in zip(c.labels_, v, np.sort(c.transform(train_fd))):
            rank[i].append([j,k[0]])
        for k,vv in rank.items():
            dist = sorted(vv, key=lambda x: x[-1])
            for i in range(rounds):
                if len(dist) > i:
                    oc_idx.append(dist[i][0])
                    print k, input3[dist[i][0]]
    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    print '=============================='
    for i in oc_idx:
        print mapping[label[i]],':',input3[i]
        train_id.append(i)
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
    acc_sum.append(acc)
print len(train_label), 'training examples'
print ct(train_label)
print 'acc using oracle centroid ex:', np.mean(acc_sum), np.std(acc_sum)
cm_ = CM(test_label, preds_fn)
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
cm_cls =np.unique(np.hstack((test_label,preds_fn)))
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
