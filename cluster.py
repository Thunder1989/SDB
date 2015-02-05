from time import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pylab as pl
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
rounds = 1
print 'total rounds of', rounds
#f = open('c_out2','w')
acc_sum = []
for train, test in kf:
    train_fd = fn[train]
    n_class = len(np.unique(label[train]))
    #print '# of training class', n_class
    g = GMM(n_components=n_class*2, covariance_type='spherical', init_params='wmc', n_iter=100)
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
        #print i,j
        ex[i] = sorted(j, key=lambda x: x[-1], reverse=True)
    #gmm_idx = [j[k][0] for i,j in ex.items() for k in range(2)]
    gmm_idx = []
    for i in range(rounds):
        for v in ex.values():
            if len(v)>i:
                gmm_idx.append(v[i][0])
    #print len(gmm_idx)

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    pre_sum = np.array([])
    rec_sum = np.array([])
    for i in gmm_idx:
        train_id.append(i)
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        #pre = precision_score(test_label, preds_fn, average=None)
        #rec = recall_score(test_label, preds_fn, average=None)
        #pre_sum = np.append(pre_sum, pre)
        #rec_sum = np.append(rec_sum, rec)
    '''
    t_class = len(pre)
    f.write('-------------gmm-------------\n')
    pre_sum = np.reshape(pre_sum, (len(pre_sum)/t_class,t_class)).T.tolist()
    rec_sum = np.reshape(rec_sum, (len(rec_sum)/t_class,t_class)).T.tolist()
    f.write('-------------precision-------------\n')
    f.writelines('%s;\n'%repr(i) for i in pre_sum)
    f.write('-------------recall-------------\n')
    f.writelines('%s;\n'%repr(i) for i in rec_sum)
    f.write('ex in each itr:'+repr(train_label)+'\n')
    f.write(repr(np.unique(test_label))+'\n')
    '''
    acc_sum.append(acc)
print len(train_label), 'training examples'
#print ct(train_label)
print 'acc using gmm ex:', np.mean(acc_sum), np.std(acc_sum)
#f.write('acc using gmm: %s\n'%(repr(acc_sum)))
#f.write('acc using gmm ex: %s-%s\n'%(str(np.mean(acc_sum)), str(np.std(acc_sum))))

mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
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
test_cls =np.unique(np.hstack((train_label, test_label)))
cls = []
for c in test_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Mn Confusion matrix (%.3f)'%acc)
pl.show()

#X = StandardScaler().fit_transform(fn)
acc_sum = []
for train, test in kf:
    train_fd = fn[train]
    n_class = len(np.unique(label[train]))
    #print '# of training class', n_class
    c = KMeans(init='k-means++', n_clusters=n_class*2, n_init=10)
    c.fit(train_fd)
#preds = c.predict(x_test)
#print metrics.homogeneity_completeness_v_measure(y_test,preds)
#print 'ARI', metrics.adjusted_rand_score(y_test, preds)
#print 'Sil', metrics.silhouette_score(x_train, c.labels_, metric='euclidean', sample_size=len(test))
#score = metrics.silhouette_samples(fd, c.labels_)
    dist = np.sort(c.transform(train_fd))
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
    #km_idx = [j[k][0] for i,j in ex.items() for k in range(2)]
    km_idx = []
    for i in range(rounds*2):
        if len(km_idx)>len(gmm_idx):
            break
        for v in ex.values():
            if len(v)>i and len(km_idx)<len(gmm_idx):
                km_idx.append(v[i][0])
    #print len(km_idx)

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    pre_sum = np.array([])
    rec_sum = np.array([])
    for i in km_idx:
        train_id.append(i)
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        #pre = precision_score(test_label, preds_fn, average=None)
        #rec = recall_score(test_label, preds_fn, average=None)
        #pre_sum = np.append(pre_sum, pre)
        #rec_sum = np.append(rec_sum, rec)
    #t_class = len(np.unique(test_label))
    '''
    t_class = len(pre)
    pre_sum = np.reshape(pre_sum, (len(pre_sum)/t_class,t_class)).T.tolist()
    rec_sum = np.reshape(rec_sum, (len(rec_sum)/t_class,t_class)).T.tolist()
    f.write('-------------km-------------\n')
    f.write('-------------precision-------------\n')
    f.writelines('%s;\n'%repr(i) for i in pre_sum)
    f.write('-------------recall-------------\n')
    f.writelines('%s;\n'%repr(i) for i in rec_sum)
    f.write('ex in each itr:'+repr(train_label)+'\n')
    f.write(repr(np.unique(test_label))+'\n')
    '''
    acc_sum.append(acc)
print len(train_label), 'training examples'
#print ct(train_label)
print 'acc using km ex:', np.mean(acc_sum), np.std(acc_sum)
#f.write('acc using km: %s\n'%(repr(acc_sum)))
#f.write('acc using km ex: %s-%s\n'%(str(np.mean(acc_sum)), str(np.std(acc_sum))))
#f.close()
#test_acc = np.mean(preds.ravel() == y_train.ravel())
#test_acc = np.mean(preds.ravel() == y_test.ravel())
#print 'test acc', test_acc
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
test_cls =np.unique(np.hstack((train_label, test_label)))
cls = []
for c in test_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Mn Confusion matrix (%.3f)'%acc)
pl.show()

acc_sum = []
for train, test in kf:
    ex_dict = dd(list)
    train_label = label[train]
    for i,j in zip(train_label,train):
        ex_dict[i].append(j)
    o_idx = []
    for v in ex_dict.values():
        random.shuffle(v)
        for i in range(len(gmm_idx)/len(ex_dict.keys())):
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
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
    acc_sum.append(acc)
print len(train_label)
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
test_cls =np.unique(np.hstack((train_label, test_label)))
cls = []
for c in test_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Mn Confusion matrix (%.3f)'%acc)
pl.show()
print 'acc using oracle ex:', np.mean(acc_sum), np.std(acc_sum)


acc_sum = []
for train, test in kf:
    rand_idx = random.sample(xrange(len(train)), len(gmm_idx))
    n_class = len(np.unique(label[train]))

    test_fn = fn[test]
    test_label = label[test]
    train_id = []
    pre_sum = np.array([])
    rec_sum = np.array([])
    for i in rand_idx:
        train_id.append(train[i])
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        #pre = precision_score(test_label, preds_fn, average=None)
        #rec = recall_score(test_label, preds_fn, average=None)
        #pre_sum = np.append(pre_sum, pre)
        #rec_sum = np.append(rec_sum, rec)
    #t_class = len(np.unique(test_label))
    '''
    t_class = len(pre)
    pre_sum = np.reshape(pre_sum, (len(pre_sum)/t_class,t_class)).T.tolist()
    rec_sum = np.reshape(rec_sum, (len(rec_sum)/t_class,t_class)).T.tolist()
    f.write('-------------random-------------\n')
    f.write('-------------precision-------------\n')
    f.writelines('%s;\n'%repr(i) for i in pre_sum)
    f.write('-------------recall-------------\n')
    f.writelines('%s;\n'%repr(i) for i in rec_sum)
    f.write('ex in each itr:'+repr(train_label)+'\n')
    f.write(repr(np.unique(test_label))+'\n')
    '''
    acc_sum.append(acc)
print len(train_label)
#print ct(train_label)
print 'acc using random ex:', np.mean(acc_sum), np.std(acc_sum)
#f.write('acc using random: %s\n'%(repr(acc_sum)))
#f.write('acc using random ex: %s-%s\n'%(str(np.mean(acc_sum)), str(np.std(acc_sum))))
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
test_cls =np.unique(np.hstack((train_label, test_label)))
cls = []
for c in test_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Mn Confusion matrix (%.3f)'%acc)
pl.show()

#print(79 * '_')
#print('% 9s' % 'init'
#      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

#bench_k_means(KMeans(init='random', n_clusters=n_class, n_init=10),
#              name="random", data=data)

