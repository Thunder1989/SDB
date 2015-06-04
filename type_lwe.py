'''
transfer learning btw buildings
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from sklearn import tree
from collections import defaultdict as DD
from collections import Counter as CT

import numpy as np
import re
import math
import random
import pylab as pl

#cross building data clx
input1 = np.genfromtxt('rice_day_wpeak', delimiter=',')
input2 = np.genfromtxt('rice_hour_sum', delimiter=',')
input3 = np.genfromtxt('rice_diu_sum', delimiter=',')
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
input5 = np.genfromtxt('sdh_hour_sum', delimiter=',')
input6 = np.genfromtxt('sdh_day_sum', delimiter=',')
input7 = np.genfromtxt('sdh_diu_sum', delimiter=',')
input8 = np.genfromtxt('keti_hour_sum', delimiter=',')
input9 = np.genfromtxt('keti_day_sum', delimiter=',')
input10 = np.genfromtxt('keti_diu_sum', delimiter=',')
input11 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
fd1 = input1[:,0:-1]
fd2 = input2[:,0:-1]
fd3 = input5[:,0:-1]
fd4 = input6[:,0:-1]
fd5 = input8[:,0:-1]
fd6 = input9[:,0:-1]
train_fd = np.hstack((fd1,fd2))
fd21 = np.hstack((fd3,fd4))
fd22 = np.hstack((fd5,fd6))
test_fd = np.vstack((fd21,fd22))
train_label = input4[:,-1]
test_label = input11[:,-1]
print train_fd.shape
print train_label.shape
print test_fd.shape
print train_label.shape

rf = RFC(n_estimators=100, criterion='entropy')
rf.fit(train_fd, train_label) #train each base classifier
print fd.score(test_fd, test_label)




'''
step1: train base models from bldg1
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
input1 = [i.strip().split('\\')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
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

rf = RFC(n_estimators=100, criterion='entropy')
svm = SVC(kernel='rbf', probability=True)
lr = LR()
#clf = LinearSVC()
bl = [lr, rf, svm] #set of base classifier
for b in bl:
    b.fit(fd, label) #train each base classifier
    print b
'''
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
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

'''
step2: TL with name feature on bldg2
'''
#input1 = [i.strip().split('\\')[-2]+i.strip().split('\\')[-1][:-4] for i in open('sdh_pt_name').readlines()]
input1 = [i.strip().split('\\')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
fd = input2[:,[0,1,2,3,5,6,7]]
#input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
#input2 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
#input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_new_part').readlines()]
#input2 = np.genfromtxt('sdh_45min_part', delimiter=',')
#input1 = [i.strip().split('_')[-1][:-5] for i in open('soda_pt_part').readlines()]
#input2 = np.genfromtxt('soda_45min_part', delimiter=',')
label = input2[:,-1]
#label2 = input4[:,-1]
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
#cv = CV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.transform(name).toarray()
#fd = fn
for b in bl:
    print b.score(fd,label)

n_class = 10
c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
c.fit(fd)
dist = np.sort(c.transform(fd))
ex = DD(list) #example id, distance to centroid
ex_id = DD(list) #example id for each C
ex_N = [] #number of examples for each C
for i,j,k in zip(c.labels_, xrange(len(fd)), dist):
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
    preds = b.predict(fd)
    ex_ = DD(list)
    for i,j in zip(preds, xrange(len(fd))):
        ex_[i].append(int(j))
    for exx in ex_.values():
        exx = np.asarray(exx)
        for e in exx:
            n[e] = exx[exx!=e]

preds = np.array([999 for i in xrange(len(fd))])
delta = 0.5
ct=0
t=0
true = []
pred = []
for i in xrange(len(fd)):
    w = []
    v_c = set(nb_c[i])
    for n in nb_f:
        v_f = set(n[i])
        sim = len(v_c & v_f) / float(len(v_c | v_f))
        w.append(sim)
    if np.mean(w) > delta:
        w[:] = [float(j)/sum(w) for j in w]
        pred_pr = np.zeros(len(class_))
        for wi, b in zip(w,bl):
            pr = b.predict_proba(fd[i])
            pred_pr = pred_pr + wi*pr
        preds[i] = class_[np.argmax(pred_pr)]
        true.append(label[i])
        pred.append(preds[i])
        ct+=1
        if preds[i]==label[i]:
            t+=1
print 'part acc' , float(t)/ct
print 'percent', float(ct)/len(label)
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
cm_ = CM(true, pred)
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
                    fontsize=8)
cm_cls =np.unique(np.hstack((true,pred)))
cls = []
for c in cm_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Confusion Matrix (%.3f)'%(float(t)/ct))
pl.show()
ctr = 0
ct = 0
t = 0
for k,v in ex_id.items():
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
if ct!=0:
    print 'propogate acc' , float(t)/ct
    print 'propogate percent', float(ct)/len(label)
print '# of manual label', ctr
print 'acc by LWE', accuracy_score(preds, label)

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
