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
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from sklearn import tree
from collections import defaultdict as DD

import numpy as np
import re
import math
import random
import pylab as pl

'''
step1: apply a data model from bldg1 to bldg2
'''
input1 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
data1 = input1[:,[0,1,2,3,5,6,7]]
label1 = input1[:,-1]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
data2 = input2[:,[0,1,2,3,5,6,7]]
label2 = input2[:,-1]
train_data = data2
train_label = label2
test_data = data1
test_label = label1
md = RFC(n_estimators=100, criterion='entropy')
#clf = LinearSVC()
md.fit(train_data, train_label)
#print 'class in Md as training:\n', clf.classes_
preds = md.predict(test_data)
acc = md.score(test_data, test_label)
print 'Md acc', acc

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
#pl.show()

'''
step2: AL with name feature on bldg2
'''
#input1 = [i.strip().split('\\')[-2]+i.strip().split('\\')[-1][:-4] for i in open('sdh_pt_name').readlines()]
input1 = [i.strip().split('\\')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
input2 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
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

cv = CV(analyzer='char_wb', ngram_range=(3,4))
#vc = CV(token_pattern='[a-z]{2,}')
fn = cv.fit_transform(name).toarray()
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
