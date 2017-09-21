'''
run transfer learning first, then aplly active learning on the remaining unlabeled population
the weight in TL is the product of confidence and sim, where confidence estimation will be updated in each iteration
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

from scikits.statsmodels.tools.tools import ECDF
from scipy import stats
from scipy.stats import t
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.linear_model import LogisticRegression as LR

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as F1
from sklearn.metrics import confusion_matrix as CM

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.preprocessing import normalize

input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_rice').readlines()]
input21 = np.genfromtxt('keti_hour_sum', delimiter=',')
input22 = np.genfromtxt('sdh_hour_rice', delimiter=',')
input2 = np.vstack((input21,input22))
#input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_sdh').readlines()]
input4 = np.genfromtxt('rice_hour_sdh', delimiter=',')
input5 = [i.strip().split('_')[-1][:-5] for i in open('soda_pt_new').readlines()]
input6 = np.genfromtxt('soda_45min_new', delimiter=',')
label1 = input2[:,-1]
label = input4[:,-1]
label1 = input6[:,-1]
#print 'class count of true labels of all ex:\n', ct(label)
#input3 = input3 #quick run of the code using other building
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))

cv = CV(analyzer='char_wb', ngram_range=(3,4))
#tv = TV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()
#fn = cv.fit_transform(input1).toarray()
#print cv.vocabulary_
#fd = input4[:,[0,1,2,3,5,6,7]]
#kmer = cv.get_feature_names()
#idf = zip(kmer, cv._tfidf.idf_)
#idf = sorted(idf, key=lambda x: x[-1], reverse=True)
#print idf[:20]
#print idf[-20:]
#print cv.get_feature_names()

fold = 10
rounds = 100
clf = LinearSVC()
#clf = SVC(kernel='linear', probability=True)
#clf = RFC(n_estimators=100, criterion='entropy')
#kf = StratifiedKFold(label, n_folds=fold, shuffle=True)
kf = KFold(len(label), n_folds=fold, shuffle=True)
p_acc = [] #pseudo label acc
acc_sum = [[] for i in xrange(rounds)]
acc_ave = dd(list)
tao = 0
alpha_ = 1
R = dd()
CI = dd()
rf = RFC(n_estimators=100, criterion='entropy')
svm = SVC(kernel='rbf', probability=True)
lr = LR()
bl = [rf, lr, svm] #set of base learner
for b in bl:
    R[b] = [1,0]
    CI[b] = 1

for train, test in kf:
    #insert TL before running AL
    input1 = np.genfromtxt('rice_hour_sdh', delimiter=',')
    input2 = np.genfromtxt('keti_hour_sum', delimiter=',')
    input21 = np.genfromtxt('sdh_hour_rice', delimiter=',')
    input3 = np.genfromtxt('soda_hour_rice', delimiter=',')
    input2 = np.vstack((input2,input21))
    fd1 = input1[:,0:-1]
    fd2 = input2[:,0:-1]
    fd3 = input3[:,0:-1]
    train_fd = fd1
    test_fd = fd2
    train_label = input1[:,-1]
    test_label = input2[:,-1]

    #switch src and tgt
    fd_tmp = train_fd
    train_fd = test_fd
    test_fd = fd_tmp
    l_tmp = train_label
    train_label = test_label
    test_label = l_tmp

    for b in bl:
        b.fit(train_fd, train_label) #train each base classifier
        #print b

    test_fn = fn
    label = test_label
    class_ = np.unique(train_label)
    n_class = 32/2
    c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
    c.fit(test_fn)
    dist = np.sort(c.transform(test_fn))
    ex = dd(list) #example id, distance to centroid
    ex_id = dd(list) #example id for each C
    for i,j,k in zip(c.labels_, xrange(len(test_fn)), dist):
        ex[i].append([j,k[0]])
        ex_id[i].append(int(j))
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1]) #sort ex in each C by dist to centroid
    nb_c = dd()
    for exx in ex_id.values():
        exx = np.asarray(exx)
        for e in exx:
            nb_c[e] = exx[exx!=e] #create a dict of nb by C for each ex
    nb_f = [dd(), dd(), dd()]
    for b,n in zip(bl, nb_f):
        preds = b.predict(test_fd)
        ex_ = dd(list)
        for i,j in zip(preds, xrange(len(test_fd))):
            ex_[i].append(int(j))
        for exx in ex_.values():
            exx = np.asarray(exx)
            for e in exx:
                n[e] = exx[exx!=e] #create a dict of nb by f for each ex

    delta = 0.6
    pred = []
    l_id = []
    sim_ = dd() #sim score by each f for each ex
    for i in xrange(len(test_fn)):
        w = []
        v_c = set(nb_c[i])
        for n in nb_f:
            v_f = set(n[i])
            cns = len(v_c & v_f) / float(len(v_c | v_f)) #original count based sim
            inter = v_c & v_f
            union = v_c | v_f
            d_i = 0
            d_u = 0
            for it in inter:
                d_i += np.linalg.norm(test_fn[i]-test_fn[it])
            for u in union:
                d_u += np.linalg.norm(test_fn[i]-test_fn[u])
            sim = cns
            if d_i != 0:
                sim = 1 - (d_i/d_u)/cns
                #sim = (d_i/d_u)/cns
            w.append(sim)
            sim_[i] = w
            if sim<0:
                pass
                #print 'bug case',d_i, d_u, len(inter), len(union)
        if np.mean(w) > delta:
            w[:] = [float(j)/sum(w) for j in w]
            pred_pr = np.zeros(len(class_))
            for wi, b in zip(w,bl):
                pr = b.predict_proba(test_fd[i])
                pred_pr = pred_pr + wi*pr
            tmp = class_[np.argmax(pred_pr)]
            pred.append(tmp)
            l_id.append(i)
    print 'tl label #', len(l_id)
    print 'tl acc', accuracy_score(pred, label[l_id])

    #remove ex labeled by TL from AL training set
    p_idx = []
    p_label = []
    km_idx = []
    p_dist = dd()
    for i,l in zip(l_id,pred):
        if i in train:
            p_idx.append(i)
            p_label.append(l)
            train = train[train!=i]
            p_dist[i] = 0

    #training set for AL
    train_fd = fn[train]
    test_fn = fn[test]
    test_label = label[test]

    train_fn = fn[p_idx]
    train_label = p_label
    if len(np.unique(train_label))>1:
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        acc_sum[0].append(acc)

    c = KMeans(init='k-means++', n_clusters=32/2, n_init=10)
    c.fit(train_fd)
    '''
    c = DPGMM(n_components=50, covariance_type='diag', alpha=1)
    c.fit(train_fd)
    c_labels = c.predict(train_fd)
    print '# of GMM', len(np.unique(c_labels))
    mu = c.means_
    cov = c._get_covars()
    c_inv = []
    for co in cov:
        c_inv.append(np.linalg.inv(co))
    e_pr = np.sort(c.predict_proba(train_fd))
    '''
    dist = np.sort(c.transform(train_fd))
    ex = dd(list) #ex id, distance to centroid
    ex_id = dd(list) #ex id for each C
    ex_N = [] #num of ex for each C
    #for i,j,k in zip(c_labels, train, e_pr):
    for i,j,k in zip(c.labels_, train, dist):
        ex[i].append([j,k[0]])
        ex_id[i].append(int(j))
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1])
        ex_N.append([i,len(ex[i])])
    ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True)
    #print 'initial exs from k clusters centroid=============================='

    #'''
    #ordered by density on the first batch of exs
    ctr = 0
    for ee in ex_N:
        key = ee[0] #C id
        idx = ex[key][0][0] #ex id
        km_idx.append(idx)
        ctr += 1
        if ctr<3:
            continue

        for b in bl: #update reward table for f_i
            if b.predict(test_fd[idx]) == label[idx]:
                R[b].append(1)
            else:
                R[b].append(0)
        for b in bl: #update confidence for f_i
            r = R[b]
            n = len(r)
            cv = t.ppf(0.975, n-1)
            CI[b] = np.mean(r) + cv*np.std(r)/np.sqrt(n)

        #'''
        fit_diff = []
        pair = list(itertools.combinations(km_idx,2))
        for p in pair:
            if label[p[0]] != label[p[1]]:
                d = np.linalg.norm(fn[p[0]]-fn[p[1]])
                fit_diff.append(d)
        src = fit_diff
        tao = alpha_*min(src)/2

        #exclude exs
        tmp = [] #buffer for all unlabeled ex
        #re-visit exs removed on previous itr with the new tao
        idx_tmp = []
        label_tmp = []
        for i,j in zip(p_idx,p_label):
            if p_dist[i]<tao:
                idx_tmp.append(i)
                label_tmp.append(j)
            else:
                p_dist.pop(i)
                tmp.append(i)
        p_idx = idx_tmp
        p_label = label_tmp

        if ctr==3:
            #make up for p_labels for the first 2 itrs
            #TBD
            pass

        for e in ex_id[key]: #visit each ex in C, propagate label if d(ex, centroid) < tao
            if e == idx:
                continue
            d = np.linalg.norm(fn[e]-fn[idx])
            if d<tao:
                p_dist[e] = d
                p_idx.append(e)
                p_label.append(label[idx])
            else:
                tmp.append(e)
        if not tmp:
            ex_id.pop(key)
        else:
            ex_id[key] = tmp
        #'''
        if not p_idx:
            train_fn = fn[km_idx]
            train_label = label[km_idx]
        else:
            train_fn = fn[np.hstack((km_idx, p_idx))]
            train_label = np.hstack((label[km_idx], p_label))
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        acc_sum[ctr-2].append(acc)
    #'''

    cl_id = []
    ex_al = [] #ex added in each itr
    test_fn = fn[test]
    test_label = label[test]
    tl_ctr = 0
    for rr in range(ctr-1, rounds):
    #for rr in range(rounds):
        delta = 0.3
        pred = []
        l_id = []
        u_id = []
        for i in xrange(len(fn)):
            w = [j*CI[b] for b,j in zip(bl,sim_[i])]
            if np.mean(w) > delta:
                w[:] = [float(j)/sum(w) for j in w]
                pred_pr = np.zeros(len(class_))
                for wi, b in zip(w,bl):
                    pr = b.predict_proba(test_fd[i])
                    pred_pr = pred_pr + wi*pr
                tmp = class_[np.argmax(pred_pr)]
                pred.append(tmp)
                l_id.append(i)
            else:
                u_id.append(i)

        #update the training set for AL based on new TL results
        for i,l in zip(l_id,pred):
            if i in train and i not in p_idx:
                p_idx.append(i)
                p_label.append(l)
                tl_ctr += 1
                #print 'new tl label'
                train = train[train!=i]
                p_dist[i] = 0
                for k,v in ex_id.items():
                    if type(v) is int:
                        if v==i:
                            ex_id.pop(k)
                    else:
                        if i in v:
                            ex_id[k] = v[v!=i]
        if not p_idx:
            train_fn = fn[km_idx]
            train_label = label[km_idx]
        else:
            train_fn = fn[np.hstack((km_idx, p_idx))]
            train_label = np.hstack((label[km_idx], p_label))
        #print 'ct on traing label', ct(train_label)
        clf.fit(train_fn, train_label)
        sub_pred = dd(list) #Mn predicted labels for each cluster
        for k,v in ex_id.items():
            sub_pred[k] = clf.predict(fn[v]) #predict labels for cluster learning set

        #acc_ = accuracy_score(label[train_], preds_c)
        #print 'acc on test set', acc
        #print 'acc on cluster set', acc_
        #acc_sum[rr].append(acc)
        #print 'iteration', rr, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'

        #the original H based cluster selection
        rank = []
        for k,v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i/float(max(count)) for i in count]
            H = np.sum(-p*math.log(p,2) for p in count if p!=0)
            #H /= len(v)/float(len(train))
            rank.append([k,len(v),H])
        rank = sorted(rank, key=lambda x: x[-1], reverse=True)
        if not rank:
            break
        idx = rank[0][0] #pick the id of the 1st cluster on the rank
        cl_id.append(idx) #track cluster id on each iteration
        cc = idx #id of the cluster picked by H
        c_id = ex_id[cc] #example id of the cluster picked
        sub_label = sub_pred[idx]#used when choosing cluster by H
        sub_fn = fn[c_id]

        #sub-clustering the cluster
        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))
        '''
        n=0
        if len(c_id)>=5:
            n = 5
        else:
            n = len(c_id)
        c_ = DPGMM(n_components=n, covariance_type='diag', alpha=1)
        c_.fit(sub_fn)
        cc_labels = c_.predict(sub_fn)
        #print '# of sub-GMM', len(np.unique(cc_labels))
        mu = c_.means_
        cov = c_._get_covars()
        c_inv = []
        for co in cov:
            c_inv.append(np.linalg.inv(co))
        e_pr = np.sort(c_.predict_proba(sub_fn))
        '''
        ex_ = dd(list)
        #for i,j,k,l in zip(cc_labels, c_id, e_pr, sub_label):
        for i,j,k,l in zip(c_.labels_, c_id, dist, sub_label):
            ex_[i].append([j,l,k[0]])
        for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
            ex_[i] = sorted(j, key=lambda x: x[-1])
        for k,v in ex_.items():
            if v[0][0] not in km_idx:
                idx = v[0][0]
                km_idx.append(idx)

                for b in bl: #update reward table for f_i
                    if b.predict(test_fd[idx]) == label[idx]:
                        R[b].append(1)
                    else:
                        R[b].append(0)
                for b in bl: #update confidence for f_i
                    r = R[b]
                    n = len(r)
                    cv = t.ppf(0.975, n-1)
                    CI[b] = np.mean(r) + cv*np.std(r)/np.sqrt(n)

                #'''
                #update tao then remove ex<tao
                fit_diff = []
                pair = list(itertools.combinations(km_idx,2))
                for p in pair:
                    if label[p[0]] != label[p[1]]:
                        d = np.linalg.norm(fn[p[0]]-fn[p[1]])
                        fit_diff.append(d)
                src = fit_diff #set tao be the min(inter-class pair dist)/2
                tao = alpha_*min(src)/2

                tmp = []
                #re-visit exs removed on previous itr with the new tao
                idx_tmp = []
                label_tmp = []
                for i,j in zip(p_idx,p_label):
                    if p_dist[i]<tao:
                        idx_tmp.append(i)
                        label_tmp.append(j)
                    else:
                        p_dist.pop(i)
                        tmp.append(i)
                p_idx = idx_tmp
                p_label = label_tmp

                #tmp = []
                for e in ex_id[cc]:
                    if e == idx:
                        continue
                    d = np.linalg.norm(fn[e]-fn[idx])
                    if d<tao:
                        #print 'added ex with d',d
                        p_dist[e] = d
                        p_idx.append(e)
                        p_label.append(label[idx])
                    else:
                        tmp.append(e)
                if not tmp:
                    ex_id.pop(cc)
                else:
                    ex_id[cc] = tmp
                #ex_al.append([rr,cc,v[0][-2],label[idx],input3[idx]])
                #print cc,label[idx],input3[idx]
                #'''
                break

        #print len(km_idx), 'training examples'
        if not p_idx:
            train_fn = fn[km_idx]
            train_label = label[km_idx]
        else:
            train_fn = fn[np.hstack((km_idx, p_idx))]
            train_label = np.hstack((label[km_idx], p_label))
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        acc_sum[rr].append(acc)

    #for e in ex_al: #print the example detail added on each itr
    #    print e
    #print len(p_idx)
    print '# of new tl label', tl_ctr
    print '# of p label', len(p_label)
    print cl_id
    if not p_label:
        print 'pseudo label acc', 0
        p_acc.append(0)
    else:
        print 'pseudo label acc', sum(label[p_idx]==p_label)/float(len(p_label))
        p_acc.append(sum(label[p_idx]==p_label)/float(len(p_label)))
    print CI
    print '----------------------------------------------------'
    print '----------------------------------------------------'
#print 'class count of clf training ex:', ct(train_label)
print 'average acc:', [np.mean(i) for i in acc_sum]
print 'average p label acc:', np.mean(p_acc)

tmp = []
for i,j in acc_ave.items():
    tmp.append([i,np.mean(j)])
tmp = sorted(tmp, key=lambda x: x[0])
x = [i[0] for i in tmp]
y = [i[1] for i in tmp]

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
