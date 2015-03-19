import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
from scikits.statsmodels.tools.tools import ECDF
from scipy import stats
from scipy.optimize import curve_fit
from time import time
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
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
from sklearn.svm import LinearSVC
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
input5 = [i.strip().split('_')[-1][:-5] for i in open('soda_pt_new').readlines()]
input6 = np.genfromtxt('soda_45min_new', delimiter=',')
label1 = input2[:,-1]
label = input4[:,-1]
label1 = input6[:,-1]
#input3 = input1 #for quick run of the code using other building
#input3, label = shuffle(input3, label)
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))

cv = CV(analyzer='char_wb', ngram_range=(3,4))
#tv = TV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()
#fn = cv.fit_transform(input1).toarray()
fd = input4[:,[0,1,2,3,5,6,7]]
print 'class count of true labels of all ex:\n', ct(label)
#n_class = len(np.unique(label))
#print n_class
#print np.unique(label)
#print 'class count from groud truth labels:\n',ct(label)
#kmer = cv.get_feature_names()
#idf = zip(kmer, cv._tfidf.idf_)
#idf = sorted(idf, key=lambda x: x[-1], reverse=True)
#print idf[:20]
#print idf[-20:]
#print cv.get_feature_names()

fold = 10
rounds = 5
clf = LinearSVC()
#clf = SVC(kernel='linear')
#clf = RFC(n_estimators=100, criterion='entropy')

clf.fit(fn, label)
coef = abs(clf.coef_)
weight = np.max(coef, axis=0)
#weight = np.mean(coef,axis=0)
feature_rank = []
for i,j in zip(weight, xrange(len(weight))):
    feature_rank.append([i,j])
feature_rank = sorted(feature_rank,key=lambda x: x[0],reverse=True)
feature_idx=[]
for i in feature_rank:
    if i[0]>=0.05:
        feature_idx.append(i[1])
#print 'feature num', len(feature_idx)
fn = fn[:, feature_idx]

#fn = fd
#fn = StandardScaler().fit_transform(fn)
'''
n_class = 16
c = KMeans(init='k-means++', n_clusters=2*n_class, n_init=10)
c.fit(fn)
ex_id = dd(list) #example id for each C
for i,j in zip(c.labels_, xrange(len(fn))):
    ex_id[i].append(j)

same = []
diff = []
all = []
for i in xrange(len(fn)):
    for j in xrange(0,i):
        d = np.linalg.norm(fn[i]-fn[j])
#for v in ex_id.values():
#    pair = list(itertools.combinations(v,2))
#    for p in pair:
#        d = np.linalg.norm(fn[p[0]]-fn[p[1]])
        all.append(d)
        if label[i] == label[j]:
        #if label[p[0]] == label[p[1]]:
            same.append(d)
        else:
            diff.append(d)
#print len(same)
#print len(diff)
#t,p = stats.ttest_ind(same, diff, equal_var=False)
#print t,p
#y1 = np.mean(same)
#y2 = np.mean(diff)
#s1 = np.var(same)
#s2 = np.var(diff)
#n1 = len(same)
#n2 = len(diff)
#T = (y1-y2)/np.sqrt(s1/n1 + s2/n2)
#print T

#src = same
#ecdf = ECDF(src)
#plt.plot(ecdf.x, ecdf.y, 'b--', label='within')

x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'r--', label='within')
src = diff
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'b--', label='across')

src = all
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
#plt.plot(x, y, 'k--', label='all')
#plt.legend(loc='lower right')
#plt.grid(axis='y')
#plt.show()
#s = raw_input()
'''

#kf = StratifiedKFold(label, n_folds=fold, shuffle=True)
kf = KFold(len(label), n_folds=fold, shuffle=True)
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
#X = StandardScaler().fit_transform(fn)
acc_sum = [[] for i in xrange(rounds)]
acc_ave = dd(list)
tao = 0
for train, test in kf:
    print 'class count of true labels on cluster training ex:\n', ct(label[train])
    train_fd = fn[train]
    #n_class = len(np.unique(label[train]))
    n_class = 16
    #print '# of training class', n_class
    c = AC(n_clusters=n_class, affinity='cosine', linkage='average')
    c.fit(train_fd)
    tmp = dd(list)
    for i,j in zip(c.labels_, train):
        tmp[i].append([label[j], input3[j]])
    for k,v in tmp.items():
        for vv in v:
            pass
            #print k, vv
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    c = KMeans(init='k-means++', n_clusters=2*n_class, n_init=10)
    c.fit(train_fd)
    dist = np.sort(c.transform(train_fd))
    ex = dd(list) #example id, distance to centroid
    ex_id = dd(list) #example id for each C
    debug = dd(list) #example true label, point name
    for i,j,k in zip(c.labels_, train, dist):
        ex[i].append([j,k[0]])
        ex_id[i].append(int(j))
        #debug[i].append([label[j],k[0],k[1],input3[j]])
        debug[i].append([label[j],input3[j]])
    for k,v in debug.items():
        for vv in v:
            pass
            #print k, vv
    #ss=raw_input()
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1])
    km_idx = []
    p_idx = []
    p_label = []
    print 'initial exs from k clusters centroid=============================='
    for k,v in ex.items():
        for i in range(1):
            if len(v)<=i:
                continue
            idx = v[i][0]
            km_idx.append(idx)
            print k,label[idx],input3[idx]
    #print len(km_idx), 'training examples'

    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k*(x-x0)))
        return y

    #compute the all pair distribution, set tao to min_X
    fit_dist = []
    fit_same = []
    fit_diff = []
    pair = list(itertools.combinations(km_idx,2))
    for p in pair:
        d = np.linalg.norm(fn[p[0]]-fn[p[1]])
        fit_dist.append(d)
        if label[p[0]] == label[p[1]]:
            fit_same.append(d)
        else:
            fit_diff.append(d)
    src = fit_dist
    ecdf = ECDF(src)
    xdata = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
    ydata = ecdf(xdata)
    tao = min(xdata)
    '''
    #popt, pcov = curve_fit(sigmoid, xdata, ydata)
    #print popt
    #x_p = np.linspace(min(src), max(src), 100)
    #y_p = sigmoid(x_p, popt[0],popt[1])
    plt.plot(x, y, 'r', label='all_true')
    #plt.plot(x_p, y_p, 'k--', label='fit')
    plt.plot(xdata, ydata, 'k', label='all_prx')
    src = fit_same
    ecdf = ECDF(src)
    xdata = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
    ydata = ecdf(xdata)
    plt.plot(xdata, ydata, 'r--', label='same_prx')
    src = fit_diff
    ecdf = ECDF(src)
    xdata = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
    ydata = ecdf(xdata)
    plt.plot(xdata, ydata, 'k--', label='diff_prx')
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.show()
    '''
    #with tao, excluding the exs near initial C centroid
    for k,idx in zip(ex.keys(),km_idx):
        tmp = []
        for e in ex_id[k]:
            if e == idx:
                continue
            d = np.linalg.norm(fn[e]-fn[idx])
            if d<tao:
                p_idx.append(e)
                p_label.append(label[idx])
            else:
                tmp.append(e)
        if not tmp:
            ex_id.pop(k)
        else:
            ex_id[k] = tmp

    test_fn = fn[test]
    test_label = label[test]

    '''
    #find neighbors for each ex within each cluster
    neighbor = dd(list)
    for v in ex.values():
        idx = [vv[0] for vv in v]
        pair = list(itertools.combinations(idx,2))
        for p in pair:
            d = np.linalg.norm(fn[p[0]]-fn[p[1]])
            if d>=6:
                neighbor[p[0]].append([p[1],d])
                neighbor[p[1]].append([p[0],d])
    for k,v in neighbor.items():
        neighbor[k] = sorted(v, key=lambda x: x[-1])
    '''
    acc_itr= []
    cl_id = []
    ex_al = [] #the ex added in each itr
    ex_num = [] #num of training ex in each itr
    #for rr in range(n_class):
    for rr in range(rounds):
        ex_num.append(len(km_idx))
        train_fn = fn[np.hstack((km_idx, p_idx))]
        train_label = np.hstack((label[km_idx], p_label))
        #train_fn = fn[km_idx]
        #train_label = label[km_idx]
        print 'ct on traing label', ct(train_label)
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        sub_pred = dd(list) #Mn predicted labels for each cluster
        for k,v in ex_id.items():
            sub_pred[k] = clf.predict(fn[v]) #predict labels for cluster learning set
        #print sub_pred.values()
        acc = accuracy_score(test_label, preds_fn)
        #acc_ = accuracy_score(label[train_], preds_c)
        print 'acc on test set', acc
        #print 'acc on cluster set', acc_
        #print 'class count of predicted labels on cluster learning ex:\n', ct(preds_c)
        acc_sum[rr].append(acc)
        acc_itr.append(acc)
        '''
        for k in ex.keys():
            prev = ex_cur[k]
            nb = neighbor[prev]
            for n in nb:
                if n[0] not in km_idx:
                    km_idx.append(n[0])
                    ex_cur[k] = n[0]
                    ex_al.append([rr,k,label[n[0]],input3[n[0]]])
                    break
        '''
        '''
        #the original H based cluster selection
        rank = []
        for k,v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i/float(max(count)) for i in count]
            H = np.sum(-p*math.log(p,2) for p in count if p!=0)
            #H /= len(v)/float(len(train))
            rank.append([k,len(v),H])
            #if rr+1 == 3*n_class:
            #print k,'---',len(v), H
        #rank = sorted(rank, key=lambda x: x[-1], reverse=True)
        #print rank
        '''
        '''
        #for debug
        ss = raw_input('')
        while ss!='+':
            l = debug[int(ss)]
            for ll in l:
                print '<<', ss, ll
            ss = raw_input('')
        '''
        print 'iteration', rr, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        #idx = rank[0][0] #pick the id of the 1st cluster on the rank
        #cl_id.append(idx) #track cluster id on each iteration
        '''
        l = debug[idx]
        for ll in l:
            print '<<', idx, ll
        '''
        for cc,ll in sub_pred.items():
            #print 'cluster',cc,'# of ex.', len(ll),'# predicted L', len(np.unique(ll))
            c_id = ex_id[cc] #example id of the cluster picked
            #sub_label = sub_pred[idx]
            sub_label = ll
            sub_fn = fn[c_id]
            #name_ = []
            #for cc in c_id:
            #    name_.append(name[cc])
            #sub_fn = tv.fit_transform(name_).toarray()

            #sub-clustering the each cluster
            c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
            c_.fit(sub_fn)
            c_sub = dd(list)
            for i,j in zip(c_.labels_, c_id):
                c_sub[i].append(input3[j])
            dist = np.sort(c_.transform(sub_fn))

            ex_ = dd(list)
            for i,j,k,l in zip(c_.labels_, c_id, dist, sub_label):
                ex_[i].append([j,l,k[0]])
            for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
                ex_[i] = sorted(j, key=lambda x: x[-1])
            for k,v in ex_.items():
                if v[0][0] not in km_idx:
                    idx = v[0][0]
                    tmp = []
                    for e in ex_id[cc]:
                        if e == idx:
                            continue
                        d = np.linalg.norm(fn[e]-fn[idx])
                        if d<tao:
                            p_idx.append(e)
                            p_label.append(label[idx])
                        else:
                            tmp.append(e)
                    if not tmp:
                        ex_id.pop(cc)
                    else:
                        ex_id[cc] = tmp
                    km_idx.append(idx)
                    #ex_cur[k] = idx
                    ex_al.append([rr,cc,v[0][-2],label[idx],input3[idx]])
                    print cc,label[idx],input3[idx]
                    break

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
    for e in ex_al:
        print e
    #print cl_id
    print 'psudo label acc', sum(label[p_idx]==p_label)/float(len(p_label))
    print 'x=',ex_num
    print 'y=',repr(acc_itr)
    for i,j in zip(ex_num, acc_itr):
        acc_ave[i].append(j)
    print '---------------------------------------------'
    print '---------------------------------------------'
    ss = raw_input()
#print len(train_label), 'training examples'
print 'class count of clf training ex:', ct(train_label)
print 'average acc:', [np.mean(i) for i in acc_sum]
tmp = []
for i,j in acc_ave.items():
    tmp.append([i,np.mean(j)])
tmp = sorted(tmp, key=lambda x: x[0])
x = [i[0] for i in tmp]
y = [i[1] for i in tmp]

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
