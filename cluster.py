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
#clf = SVC(kernel='linear')
clf = RFC(n_estimators=50, criterion='entropy')

g = GMM(n_components=n_class, covariance_type='spherical', init_params='wmc', n_iter=100)
g.fit(fn)
#g.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
#print g.means_
preds = g.predict(fn)
prob = np.sort(g.predict_proba(fn))
ex = dd(list)
acc_sum = []
for i in range(10):
    for i,j,k in zip(xrange(len(fn)),preds, prob):
        if ex[j]:
            if ex[j][-1] < k[-1]:
                ex[j] = [i,k[-1]]
        else:
            ex[j] = [i,k[-1]]
    gmm_idx = [j[0] for i,j in ex.items()]

    train = []
    for i in gmm_idx:
        train.append(i)
        train_data = fn[train]
        train_label = label[train]
        clf.fit(train_data, train_label)
        preds = clf.predict(fn)
        acc = accuracy_score(label, preds)

    acc_sum.append(acc)
#print train_label
#print set(train_label)
print 'acc using gmm ex:\n', np.mean(acc_sum), np.std(acc_sum)

acc_sum = []
for i in range(10):
    rand_idx = random.sample(xrange(len(label)), len(gmm_idx))
    train = []
    for i in rand_idx:
        train.append(i)
        train_data = fn[train]
        train_label = label[train]
        clf.fit(train_data, train_label)
        preds = clf.predict(fn)
        acc = accuracy_score(label, preds)

    acc_sum.append(acc)
#print train_label
#print set(train_label)
print 'acc using random ex:\n', np.mean(acc_sum), np.std(acc_sum)

#X = StandardScaler().fit_transform(fn)
c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
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
acc_sum = []
for i in range(10):
    for i,j,k in zip(xrange(len(label)), c.labels_, dist):
        if ex[j]:
            if ex[j][-1] > k[0]:
                ex[j] = [i,k[0]]
        else:
            ex[j] = [i,k[0]]
    km_idx = [j[0] for i,j in ex.items()]

    train = []
    for i in km_idx:
        train.append(i)
        train_data = fn[train]
        train_label = label[train]
        clf.fit(train_data, train_label)
        preds = clf.predict(fn)
        acc = accuracy_score(label, preds)

    acc_sum.append(acc)
#print train_label
#print set(train_label)
print 'acc using km ex:\n', np.mean(acc_sum), np.std(acc_sum)

#test_acc = np.mean(preds.ravel() == y_train.ravel())
#test_acc = np.mean(preds.ravel() == y_test.ravel())
#print 'test acc', test_acc

#print(79 * '_')
#print('% 9s' % 'init'
#      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

#bench_k_means(KMeans(init='random', n_clusters=n_class, n_init=10),
#              name="random", data=data)

