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
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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
n_class = len(np.unique(label))
#print n_class
#print np.unique(label)
print ct(label)
'''
def bench_k_means(estimator, name, data):
    sample_size, feature_size = data.shape
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

fold = 5
skf = StratifiedKFold(label, n_folds=fold)
train, test = next(iter(skf))
x_train = data[train]
y_train = label[train]
x_test = data[test]
y_test = label[test]
n_class = len(np.unique(y_train))
print n_class, 'classes'
'''
fold = 10
#skf = StratifiedKFold(label, n_folds=fold)
kf = KFold(len(label), n_folds=fold, shuffle=True)
'''
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1
'''

#clf = SVC(kernel='linear')
clf = RFC(n_estimators=50, criterion='entropy')

acc_sum = []
g = GMM(n_components=n_class, covariance_type='spherical', init_params='wmc', n_iter=100)
for train, test in kf:
    train_fd = fd[train]
    g.fit(train_fd)
    #g.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
    #print g.means_
    preds = g.predict(train_fd)
    prob = np.sort(g.predict_proba(train_fd))
    #print len(np.unique(preds))
    #print np.unique(preds)

    ex = dd(list)
    for i,j,k in zip(preds, train, prob):
        '''
        if ex[j]:
            if ex[j][-1] < k[-1]:
                ex[j] = [i,k[-1]]
        else:
            ex[j] = [i,k[-1]]
        '''
        ex[i].append([j,k[-1]])
    for i,j in ex.items():
        print i,j
        ex[i] = sorted(j, key=lambda x: x[-1], reverse=True)
    gmm_idx = [j[k][0] for i,j in ex.items() for k in range(2)]

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    for i in gmm_idx:
        train_id.append(i)
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
    acc_sum.append(acc)
print ct(train_label)
print len(gmm_idx)
print 'acc using gmm ex:\n', np.mean(acc_sum), np.std(acc_sum)

acc_sum = []
for train, test in kf:
    rand_idx = random.sample(xrange(len(train)), len(gmm_idx))

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
print 'acc using random ex:\n', np.mean(acc_sum), np.std(acc_sum)

#X = StandardScaler().fit_transform(fn)
c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
acc_sum = []
for train, test in kf:
    c.fit(fn)
#preds = c.predict(x_test)
#print metrics.homogeneity_completeness_v_measure(y_test,preds)
#print 'ARI', metrics.adjusted_rand_score(y_test, preds)
#print 'Sil', metrics.silhouette_score(x_train, c.labels_, metric='euclidean', sample_size=len(test))
#score = metrics.silhouette_samples(fd, c.labels_)
#rank = zip(xrange(len(label)), c.labels_, score)
#rank = sorted(rank, key=lambda x: x[-1])
#print len(rank)
#print rank[:20]
    dist = np.sort(c.transform(fn))
    ex = dd(list)
    for i,j,k in zip(c.labels_, train, dist):
        '''
        if ex[j]:
            if ex[j][-1] > k[0]:
                ex[j] = [i,k[0]]
        else:
            ex[j] = [i,k[0]]
        '''
        ex[i].append([j,k[0]])
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1])
    km_idx = [j[k][0] for i,j in ex.items() for k in range(2)]

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    for i in km_idx:
        train_id.append(i)
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds)
    acc_sum.append(acc)
print ct(train_label)
print 'acc using km ex:\n', np.mean(acc_sum), np.std(acc_sum)

#test_acc = np.mean(preds.ravel() == y_train.ravel())
#test_acc = np.mean(preds.ravel() == y_test.ravel())
#print 'test acc', test_acc

#print(79 * '_')
#print('% 9s' % 'init'
#      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

#bench_k_means(KMeans(init='random', n_clusters=n_class, n_init=10),
#              name="random", data=data)

