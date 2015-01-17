from time import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict as dd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
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
label1 = input2[:,-1]
label2 = input4[:,-1]
vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#vc = TV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
data = vc.fit_transform(input1).toarray()
#data = input4[:,[0,1,2,3,5,6,7]]

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
skf = StratifiedKFold(label1, n_folds=fold)
train, test = next(iter(skf))
x_train = data[train]
y_train = label1[train]
x_test = data[test]
y_test = label1[test]
n_class = len(np.unique(y_train))
print n_class, 'classes'
c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
c.fit(x_train)
preds = c.predict(x_test)
#print metrics.homogeneity_completeness_v_measure(y_test,preds)
#print 'ARI', metrics.adjusted_rand_score(y_test, preds)
#print 'Sil', metrics.silhouette_score(x_train, c.labels_, metric='euclidean', sample_size=len(test))
score = metrics.silhouette_samples(x_train, c.labels_)
rank = zip(train, y_train, c.labels_, score)
rank = sorted(rank, key=lambda x: x[-1])
#print len(rank)
#print rank[:20]

g = GMM(n_components=n_class, covariance_type='spherical', init_params='wmc', n_iter=100)
g.fit(data)
#g.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
#print g.means_
preds = g.predict(data)
prob = np.sort(g.predict_proba(data))
ex = dd(list)
for i,j,k in zip(label1, preds, prob):
    if ex[j]:
        if ex[j][-1] < k[-1]:
            ex[j] = [i,k[-1]]
    else:
        ex[j] = [i,k[-1]]
ex_clx = [j[0] for i,j in ex.items()]

ex_rand = []
for i in random.sample(xrange(0, len(label1)-1), len(ex)):
    ex_rand.append(label1[i])
print len(ex)
print ex
print set(ex_clx)
print ex_rand
print set(ex_rand)

#test_acc = np.mean(preds.ravel() == y_train.ravel())
#test_acc = np.mean(preds.ravel() == y_test.ravel())
#print 'test acc', test_acc

#print(79 * '_')
#print('% 9s' % 'init'
#      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

#bench_k_means(KMeans(init='random', n_clusters=n_class, n_init=10),
#              name="random", data=data)

