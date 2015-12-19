'''
buildsys15's method:
local weighted transfer learning btw buildings
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
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from sklearn import tree
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color
from scikits.statsmodels.tools.tools import ECDF

import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt

input1 = np.genfromtxt('rice_hour_sdh', delimiter=',')
#input1 = np.genfromtxt('sdh_hour_soda', delimiter=',')
#input1 = np.genfromtxt('soda_hour_rice', delimiter=',')
input21 = np.genfromtxt('keti_hour_sum', delimiter=',')
#input21 = np.genfromtxt('rice_hour_soda', delimiter=',')
input3 = np.genfromtxt('sdh_hour_rice', delimiter=',')
input2 = np.vstack((input21,input3))
fd1 = input1[:,0:-1]
fd2 = input2[:,0:-1]
fd3 = input3[:,0:-1]
#train_fd = np.hstack((fd1,fd2))
train_fd = fd1
test_fd = fd2
train_label = input1[:,-1]
#test_label = np.hstack((input2[:,-1],input3[:,-1]))
test_label = input2[:,-1]
#print np.unique(train_label)
#print np.unique(test_label)
#print train_fd.shape
#print train_label.shape
#print test_fd.shape
#print test_label.shape

fd_tmp = train_fd
train_fd = test_fd
test_fd = fd_tmp
l_tmp = train_label
train_label = test_label
test_label = l_tmp

rf = RFC(n_estimators=100, criterion='entropy')
#rf = SVC(kernel='poly')
rf.fit(train_fd, train_label) #train each base classifier
#print rf.feature_importances_
pred = rf.predict(test_fd) #train each base classifier
for i,j,k in zip(np.ravel(pred), np.ravel(test_label), xrange(len(test_label))):
    if i!=j:
        pass
        #print i,j,input12[k]
        #print k+30
print 'test acc',rf.score(test_fd, test_label)

mapping = {1:'co2',2:'humidity',3:'pressure',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu',30:'pos',31:'power',32:'ctrl',33:'fan spd',34:'timer'}
cm_ = CM(test_label, pred)
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap=Color.YlOrBr)
fig.colorbar(cax)
for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=9)
cm_cls =np.unique(np.hstack((test_label, pred)))
cls = []
for c in cm_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Confusion Matrix (%.3f)'%(rf.score(test_fd, test_label)))
pl.show()

'''
step1: train base models from bldg1
'''
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
bl = [rf, lr, svm] #set of base learner
for b in bl:
    b.fit(train_fd, train_label) #train each base classifier
    #print b

'''
step2: TL with name feature on bldg2
'''
#input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_soda').readlines()]
input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_sdh').readlines()]
#input1 = [i.strip().split('\\')[-1][:-5] for i in open('soda_pt_rice').readlines()]
label = test_label
class_ = np.unique(train_label)
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
cv = CV(analyzer='char_wb', ngram_range=(3,4))
test_fn = cv.fit_transform(name).toarray()
for b in bl:
    print b.score(test_fd,label)

n_class = 32/2
c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
c.fit(test_fn)
dist = np.sort(c.transform(test_fn))
ex = DD(list) #example id, distance to centroid
ex_id = DD(list) #example id for each C
ex_N = [] #number of examples for each C
for i,j,k in zip(c.labels_, xrange(len(test_fn)), dist):
    ex[i].append([j,k[0]])
    ex_id[i].append(int(j))
for i,j in ex.items():
    ex[i] = sorted(j, key=lambda x: x[-1])
    ex_N.append([i,len(ex[i])])
ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True) #sort cluster by density
nb_c = DD()
for exx in ex_id.values():
    exx = np.asarray(exx)
    for e in exx:
        nb_c[e] = exx[exx!=e]
nb_f = [DD(), DD(), DD()]
for b,n in zip(bl, nb_f):
    preds = b.predict(test_fd)
    ex_ = DD(list)
    for i,j in zip(preds, xrange(len(test_fd))):
        ex_[i].append(int(j))
    for exx in ex_.values():
        exx = np.asarray(exx)
        for e in exx:
            n[e] = exx[exx!=e]

#find k NN for each ex without considering clustering
k = 21
nb = NN(n_neighbors=k, algorithm='ball_tree', metric='euclidean').fit(test_fn)
distances, indices = nb.kneighbors(test_fn)

preds = np.array([999 for i in xrange(len(test_fd))])
acc_ = []
cov_ = []
#for delta in np.linspace(0.4, 0.4, 1):
for delta in xrange(1, k):
    print 'delta =', delta
    ct=0
    ct_=0
    t=0
    true = []
    pred = []
    l_id = []
    mean_t = []
    mean_f = []
    h_t = []
    h_f = []
    output = DD()
    for i in xrange(len(test_fn)):
        '''
        #the origianl C v.s. f similiarity
        w = []
        v_c = set(nb_c[i])
        for n in nb_f:
            v_f = set(n[i])
            cns = len(v_c & v_f) / float(len(v_c | v_f)) #original count based weight
            #print cns
            #print input1[i],
            #print 'sim', cns, len(v_c & v_f), len(v_c | v_f),
            inter = v_c & v_f
            union = v_c | v_f
            d_i = 0
            d_u = 0
            for it in inter:
                d_i += np.linalg.norm(test_fn[i]-test_fn[it])
            for u in union:
                d_u += np.linalg.norm(test_fn[i]-test_fn[u])
            sim = cns
            if len(inter) != 0:
                sim = 1 - (d_i/d_u)/cns
                #sim = (d_i/d_u)/cns
            if i==None:
                print '==============================================='
                print name[i], sim, d_i/len(inter), d_u/len(union)
                print '\t --> common nb', len(inter)
                for it in inter:
                    print name[it], np.linalg.norm(test_fn[i]-test_fn[it])
                print '\t --> other f nb', len(union)
                for u in union:
                    if u not in inter:
                        print name[u], np.linalg.norm(test_fn[i]-test_fn[u])
                s = raw_input('pause...')
            if i in output:
                output[i].extend(['%s/%s'%(len(inter), len(union)), 1-sim])
            else:
                output[i] = ['%s/%s'%(len(inter), len(union)), 1-sim]
            w.append(sim)
        output[i].append(np.mean(w))
        #H = np.sum(-p*math.log(abs(p),2) for p in w if p!=0)
        #H = np.max(w)
        #m = np.mean(w)

        if np.mean(w) >= delta:
            w[:] = [float(j)/sum(w) for j in w]
            pred_pr = np.zeros(len(class_))
            for wi, b in zip(w,bl):
                pr = b.predict_proba(test_fd[i])
                pred_pr = pred_pr + wi*pr
            preds[i] = class_[np.argmax(pred_pr)]
            true.append(label[i])
            pred.append(preds[i])
            ct+=1
            l_id.append(i)
            if preds[i]==label[i]:
                t+=1
                #mean_t.append(m)
                #h_t.append(H)
            else:
                pass
                #mean_f.append(m)
                #h_f.append(H)
        else:
            #ct_+=1
            w_ = w
            w[:] = [float(j)/sum(w) for j in w]
            pred_pr = np.zeros(len(class_))
            tmp = []
            tmp_pr = []
            for wi, b in zip(w,bl):
                pr = b.predict_proba(test_fd[i])
                tmp.append(int(b.predict(test_fd[i])))
                tmp_pr.append(pr)
                pred_pr = pred_pr + wi*pr
            pred_ = class_[np.argmax(pred_pr)]
            #if pred_ == label[i]:
                #print '--->',
                #h_t.append(H)
            #else:
                #print '===>',
                #h_f.append(H)
            #print w_, tmp_pr, tmp, pred_, label[i], input1[i]
            '''
        #new kNN based approach
        '''
        print '======================='
        print name[i]
        idx = indices[i]
        for ii in idx:
            print name[ii]
        '''
        p = rf.predict(test_fd[indices[i]])
        #print p
        #s = raw_input('pause...')
        if len(np.unique(p)) <= delta:
            ct+=1
            l_id.append(i)
            p_tmp = rf.predict(test_fd[i])
            if p_tmp ==label[i]:
                t+=1
            pred.append(rf.predict(test_fd[i]))

    #print 'ct' , ct
    #print '# l_id', len(l_id)
    #print FS(true, pred, average='weighted')
    #s = raw_input('pause...')
    acc_.append(float(t)/ct)
    cov_.append(float(ct)/len(label))

    '''
    tl_label = DD()
    for i,j in zip(l_id,pred):
        tl_label[i] = j
    for k,v in ex_id.items():
        print '\t ===================================='
        ctr = 0
        N = len(v)
        for vv in v:
            if vv in l_id:
                ctr+=1
                print '\/', vv, name[vv], label[vv], tl_label[vv]
            else:
                print '><', vv, name[vv], label[vv], [int(b.predict(test_fd[vv])) for b in bl]
            print '\t -->', output[vv]

        print '\t %.3f of %s labeled in C-%s'%(ctr/float(N), N, k)
    '''

'''
src = h_t
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.figure(figsize=(8,5))
plt.plot(x, y, 'k--', label='intra-class', linewidth=2.0)
src = h_f
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'r--', label='intra-class', linewidth=2.0)
plt.show()
'''

print 'y1=',acc_,';'
print 'y2=',cov_,';'
s = raw_input('end of TL...')

#print 'FN', fn
'''
src = mean_t
ecdf = ECDF(src)
#x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
#y = ecdf(x)
plt.step(ecdf.x, ecdf.y, 'k', label='mean for trues')
src = mean_f
ecdf = ECDF(src)
plt.step(ecdf.x, ecdf.y, 'r', label='mean for falses')
plt.legend(loc='lower right')
plt.xlabel('mean of weight')
plt.title('CDF of mean weight distribution for T/F')
plt.grid(axis='y')
plt.show()

src = h_t
ecdf = ECDF(src)
plt.step(ecdf.x, ecdf.y, 'k', label='H for trues')
src = h_f
ecdf = ECDF(src)
plt.step(ecdf.x, ecdf.y, 'r', label='H for falses')
plt.legend(loc='lower right')
plt.xlabel('entropy of weight')
plt.title('CDF of weight entropy distribution for T/F')
plt.grid(axis='y')
plt.show()
'''
pair = list(itertools.combinations(l_id,2))
dist = []
for p in pair:
    if preds[p[0]] != preds[p[1]]:
        dist.append(np.linalg.norm(test_fn[p[0]]-test_fn[p[1]]))
#r = np.percentile(dist,5)/2
#print 'estimated r', r

cm_ = CM(true, pred)
cm_sum = np.sum(cm_, axis=1)
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
fig = pl.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#cax = ax.matshow(cm, cmap=Color.YlOrBr)
cax = ax.matshow(cm, cmap=Color.Blues)
#fig.colorbar(cax)
for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        #ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
        ax.annotate(str("%d"%cm_[x][y]), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16)
cm_cls = np.unique(np.hstack((true,pred)))
cls_x = []
cls_y = []
for c, count in zip(cm_cls,cm_sum):
    cls_x.append(mapping[c]+'\n%.3f'%(float(label_sum[c])/len(label)))
    if label_sum[c] == 0:
        cls_y.append(mapping[c]+'\n%.3f'%(float(0)))
    else:
        cls_y.append(mapping[c]+'\n%.3f'%(float(count)/label_sum[c]))
pl.yticks(range(len(cls_y)), cls_y,fontsize=16)
pl.ylabel('True label',fontsize=16)
pl.xticks(range(len(cls_x)), cls_x,fontsize=16)
pl.xlabel('Predicted label',fontsize=16)
#pl.title('Confusion Matrix')
#pl.title('Confusion Matrix (%.3f on %0.3f\%, threshold=%s)'%(float(t)/ct,float(ct)/len(label),delta))
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('cm_single.pdf')
pp.savefig(dpi = 300)
pp.close()
pl.show()

#knn pass
knn = KNN(n_neighbors=1, weights='distance', metric='euclidean')
knn.fit(test_fn[l_id],pred)
#knn.fit(test_fn,preds)
ct = 0
t = 0
true = np.array(true)
for i in xrange(len(test_fn)):
    if preds[i] != 999:
        continue
    tmp = knn.predict(test_fn[i])
    dis, idx = knn.kneighbors(test_fn[i],3)
    if tmp == 999:
        continue
    else:
        preds[i] = tmp
        #print '--->', tmp, label[i], input1[i], true[idx], dis
        ct+=1
        if preds[i]==label[i]:
            t+=1
print '# of Y by knn', ct
if ct!=0:
    print 'knn acc' , float(t)/ct
    print 'knn percent', float(ct)/len(label)

ctr = 0
ct = 0
t = 0
em = 0
for k,v in ex.items():
    '''
    #propagate the majority label in the cluster
    l = preds[v]
    if np.mean(l)==999:
        idx = ex[k][0][0]
        m = label[idx]
        ctr += 1
    else:
        rank = CT(l).keys()
        m = rank[0]
        if m==999:
            m=rank[1]
        for vv in v:
            if preds[vv]==999:
                preds[vv] = m
                ct+=1
                if preds[vv]==label[vv]:
                    t+=1
    '''
    v = [vv[0] for vv in v]
    if np.sum(preds[np.array(v)]!=999)==0:
        em += 1
    for j,vv in enumerate(v):
        if preds[vv] == 999:
            continue
        #print 'working on', len(v), j, input1[vv]
        for v_ in v:
            if v_ == vv:
                continue
            #label propagation based on the radius
            d = np.linalg.norm(test_fn[v_]-test_fn[vv])
            if preds[v_] == 999:
                #print 'u-ex', label[v_], input1[v_]
                if d<=r:
                    preds[v_] = preds[vv]
                    true.append(label[v_])
                    pred.append(preds[v_])
                    print len(v), j,'--', preds[v_], label[v_], input1[v_], input1[vv]
                    ct+=1
                    if preds[v_]==label[v_]:
                        t+=1
print 'no labeled cluster', em
print 'propogated #', ct
if ct!=0:
    print 'propogate acc' , float(t)/ct
    print 'propogate percent', float(ct)/len(label)
#print '# of manual label', ctr
#print 'acc by LWE', accuracy_score(preds, label)
cm_ = CM(true, pred)
cm_sum = np.sum(cm_, axis=1)
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap=Color.YlOrBr)
fig.colorbar(cax)
for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10)
cm_cls = np.unique(np.hstack((true,pred)))
cls_x = []
cls_y = []
for c, count in zip(cm_cls,cm_sum):
    cls_x.append(mapping[c]+'\n%.3f'%(float(label_sum[c])/len(label)))
    cls_y.append(mapping[c]+'\n%.3f'%(float(count)/label_sum[c]))
pl.yticks(range(len(cls_y)), cls_y)
pl.ylabel('True label')
pl.xticks(range(len(cls_x)), cls_x)
pl.xlabel('Predicted label')
pl.title('Confusion Matrix (%.3f on %0.3f, threshold=%s)'%(float(t)/ct,float(ct)/len(label),delta))
pl.show()


'''
============
old block
============
'''
iteration = 40
fold = 10
clx = 13
kf = KFold(len(label), n_folds=fold, shuffle=True)
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1

acc_sum = [[] for i in range(iteration)]
acc_train = [[] for i in range(iteration)]
acc_Md = []
acc_type = [[] for i in range(clx)]
#acc_type = [[[] for i in range(iteration)] for i in range(6)]
#clf = RFC(n_estimators=100, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = SVC(kernel='linear')
mn = LinearSVC()

for train, test in kf:
#for fd in range(1):
    '''
    print 'running AL on new bldg - fold', fd
    train = np.hstack((folds[(fd+x)%fold] for x in range(1)))
    validate = np.hstack((folds[(fd+x)%fold] for x in range(1,fold/2)))
    #cut train to one example
    validate = np.append(validate,train[1:])
    train = train[:1]

    test = np.hstack((folds[(fd+x)%fold] for x in range(fold/2,fold)))
    '''
    test_data = fn[test]
    test_label = label[test]
    train_data = fn[train]
    preds = md.predict(data1[train])
    train_label = DD()
    for i,j in zip(train, preds):
        train_label[i] = j
    #acc_Md.append(accuracy_score(test_label, label1[test]))
    mn.fit(train_data, preds)
    acc = mn.score(test_data, test_label)
    acc_sum[0].append(acc)

    c_fn = fn[train]
    #n_class = len(np.unique(label[train]))
    n_class = 20
    c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
    c.fit(c_fn)
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
    dist = np.sort(c.transform(c_fn))
    ex = DD(list) #example id, distance to centroid
    ex_id = DD(list) #example id for each C
    ex_N = [] #number of examples for each C
    #for i,j,k in zip(c_labels, train, e_pr):
    for i,j,k in zip(c.labels_, train, dist):
        ex[i].append([j,k[0]])
        ex_id[i].append(int(j))
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1])
        ex_N.append([i,len(ex[i])])
    ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True) #sort cluster by density

    #confidence of training ex
    label_pr = np.sort(md.predict_proba(data1[train]))
    cf_d = DD()
    for i,pr in zip(train, label_pr):
        if len(pr)<2:
            margin = 1
        else:
            margin = pr[-1]-pr[-2]
        cf_d[i] = margin
    #cf_d = sorted(cf_d, key=lambda x: x[-1])

    for itr in range(1,n_class+1):
        print 'running itr', itr
        #train_data = fn[train]
        #train_label = md.predict(train_data)
        #validate_data = data1[validate]
        #validate_label = label1[validate]

        #kNN based voting on Md labels
        knn = KNN(n_neighbors=3, weights='distance', metric='euclidean')
        c_id = ex_N[itr-1][0]
        e_id = np.asarray([i[0] for i in ex[c_id]]) #voting starts from the centroid
        '''
        sub_cf = [[i, cf_d[i]] for i in e_id]
        sub_cf = sorted(sub_cf, key=lambda x: x[-1])
        e_id = np.asarray([i[0] for i in sub_cf]) #voting starts frim min_cf by Md
        '''
        for i in e_id:
            tmp = e_id[e_id!=i]
            X = fn[tmp]
            Y = []
            for t in tmp:
                Y.append(train_label[t])
            knn.fit(X, Y)
            train_label[i] = int(knn.predict(fn[i]))

        Y = []
        for t in train:
            Y.append(train_label[t])
        mn.fit(train_data, Y)
        #print mn.classes_
        acc = mn.score(test_data, test_label)
        acc_sum[itr].append(acc)
        acc_train[itr].append(accuracy_score(label[train], Y))

        '''
        #entropy based example selection block
        #compute entropy for each instance and rank
        label_pr = np.sort(clf.predict_proba(validate_data)) #sort in ascending order
        preds = clf.predict(validate_data)
        res = []
        for h,i,pr in zip(validate,preds,label_pr):
            entropy = np.sum(-p*math.log(p,clx) for p in pr if p!=0)
            if len(pr)<2:
                margin = 1
            else:
                margin = pr[-1]-pr[-2]
            cf = cf_d[h][0][-1]
            res.append([h,i,entropy,cf/(margin+1)])

        res = sorted(res, key=lambda x: x[-1], reverse=True)
        elmt = res[idx][0]
        ex.extend([itr+1, elmt, label1[elmt], label_gt[elmt]])
        train = np.append(train, elmt)
        validate = validate[validate!=elmt]
        '''
#print 'acc from Md', np.mean(acc_Md)
ave_acc = [np.mean(acc) for acc in acc_sum]
ave_train = [np.mean(acc) for acc in acc_train]
#acc_std = [np.std(acc) for acc in acc_sum]
print 'overall acc:', repr(ave_acc)
print 'overall acc on train:', repr(ave_train)
#print 'acc std:', repr(acc_std)
#print 'acc by type', repr(acc_type)
#f = open('pipe_out','w')
#f.writelines('%s;\n'%repr(i) for i in acc_type)
#f.write('ex in each itr:'+repr(ex)+'\n')
#f.write(repr(np.unique(test_label)))
#f.close()
#for i in acc_type:
    #print 'a = ', repr(i), '; plot(a\');'
#print repr(ex)

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
