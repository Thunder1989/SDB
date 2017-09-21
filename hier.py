import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
from scipy import stats
from time import time
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize

input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_rice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_sdh').readlines()]
input4 = np.genfromtxt('rice_hour_sdh', delimiter=',')
input5 = [i.strip().split('_')[-1][:-5] for i in open('soda_pt_new').readlines()]
input6 = np.genfromtxt('soda_45min_new', delimiter=',')
label1 = input2[:,-1]
label = input4[:,-1]
label1 = input6[:,-1]
#input3 = input3 #quick run of the code using other building
#input3, label = shuffle(input3, label)
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))

cv = CV(analyzer='char_wb', ngram_range=(3,4))
#tv = TV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()
#fn = cv.fit_transform(input1).toarray()
#print cv.vocabulary_
print 'class count of true labels of all ex:\n', ct(label)
#n_class = len(np.unique(label))
#print n_class
#print np.unique(label)
#print 'class count from groud truth labels:\n',ct(label)

fold = 10
rounds = 100
#clf = LR()
#clf = LinearSVC()
#clf = SVC(kernel='linear', probability=True)
clf = RFC(n_estimators=100, criterion='entropy')

#kf = StratifiedKFold(label, n_folds=fold, shuffle=True)
kf = KFold(len(label), n_folds=fold, shuffle=True)
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
#X = StandardScaler().fit_transform(fn)
p_acc = [] #pseudo label acc
acc_sum = [[] for i in xrange(rounds)]
acc_ave = dd(list)
tao = 0
alpha_ = 1

p1 = []
p5 = []
p10 = []
run = 0
n_clu = 30
for train, test in kf:
    run += 1
    print 'fold', run
    #print 'class count of true labels on cluster training ex:\n', ct(label[train])
    train_fn = fn[train]
    test_fn = fn[test]
    test_label = label[test]
    hc = AC(n_clusters=n_clu, linkage="average")
    hc.fit(train_fn)
    #n_class = len(np.unique(label[train]))
    #ii = itertools.count(train_fn.shape[0])
    #T = dd()
    #[{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in hc.children_]
    ex = dd(list)
    ex_N = dd() #example id for each C
    for i,j in zip(hc.labels_, train):
        ex[i].append(int(j))
    for i,j in ex.items():
        ex_N[i] = len(ex[i])

    c_L = dd()
    for i in ex.keys():
        c_L[i] = []
    train_id = []
    for r in xrange(rounds):
        H_rank = []
        for k in ex.keys():
            v = c_L[k]
            if not v:
                H = 0
            else:
                count = ct(v).values()
                count[:] = [i/float(max(count)) for i in count]
                H = np.sum(-p*math.log(p,2) for p in count if p!=0)
            H_rank.append([k,H*ex_N[k]])
        H_rank  = sorted(H_rank, key=lambda x: x[-1], reverse=True)
        flag = 1
        ctr = 0
        while(flag):
            c_id = H_rank[ctr][0]
            ex_ = ex[c_id]
            random.shuffle(ex_)
            for e in ex_:
                if e not in train_id:
                    train_id.append(e)
                    c_L[c_id].append(label[e])
                    flag=0
                    break
            ctr+=1

        print c_L
        train_fn = fn[train_id]
        train_label = label[train_id]
        clf.fit(train_fn, train_label)
        preds = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds)
        acc_sum[r].append(acc)

    print '----------------------------------------------------'
    print '----------------------------------------------------'
    #ss = raw_input()
#print 'class count of clf training ex:', ct(train_label)
print 'average acc:', [np.mean(i) for i in acc_sum]
#print 'average p label acc:', np.mean(p_acc)

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
