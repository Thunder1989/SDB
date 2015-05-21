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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
from collections import defaultdict
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
train_data = data1
train_label = label1
test_data = data2
test_label = label2
clf = RFC(n_estimators=100, criterion='entropy')
#clf = LinearSVC()
clf.fit(train_data, train_label)
#print 'class in Md as training:\n', clf.classes_
preds = clf.predict(test_data)
acc = clf.score(test_data, test_label)
print 'Md acc', acc

#compute 'confidence' for each example in the new bldg
label_pr = np.sort(clf.predict_proba(test_data)) #sort each prob vector in ascending order
cf_d = defaultdict(list)
for h,i,pr in zip(range(len(test_data)),preds,label_pr):
    #entropy = np.sum(-p*math.log(p,2) for p in pr if p!=0)
    if len(pr)<2:
        margin = 1
    else:
        margin = pr[-1]-pr[-2]
    cf_d[h].append([i,margin])

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
cm_cls =np.unique(np.hstack((test_label,preds)))
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
step2: AL with string feature on bldg2
'''
#input1 = [i.strip().split('\\')[-2]+i.strip().split('\\')[-1][:-4] for i in open('sdh_pt_name').readlines()]
#input1 = [i.strip().split('\\')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
#input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
input2 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
#input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_new_part').readlines()]
#input2 = np.genfromtxt('sdh_45min_part', delimiter=',')
#input1 = [i.strip().split('_')[-1][:-5] for i in open('soda_pt_part').readlines()]
#input2 = np.genfromtxt('soda_45min_part', delimiter=',')
label1 = input2[:,-1]
#label2 = input4[:,-1]

name = []
for i in input2:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
label_gt = input2[:,-1]
label1 = preds

iteration = 120
fold = 5
clx = 13
kf = KFold(len(label1), n_folds=fold, shuffle=True)
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1

acc_sum = [[] for i in range(iteration)]
acc_Md = []
acc_type = [[] for i in range(clx)]
#acc_type = [[[] for i in range(iteration)] for i in range(6)]
clf = RFC(n_estimators=50, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = SVC(kernel='linear')

vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#vc = CV(token_pattern='[a-z]{2,}')
data1 = vc.fit_transform(input1).toarray()
ex = []
for fd in range(1):
    print 'running AL on new bldg - fold', fd
    train = np.hstack((folds[(fd+x)%fold] for x in range(1)))
    validate = np.hstack((folds[(fd+x)%fold] for x in range(1,fold/2)))
    #cut train to one example
    validate = np.append(validate,train[1:])
    train = train[:1]

    test = np.hstack((folds[(fd+x)%fold] for x in range(fold/2,fold)))
    test_data = data1[test]
    test_label = label_gt[test]
    acc_Md.append(accuracy_score(test_label, label1[test]))

    for itr in range(iteration):
        train_data = data1[train]
        train_label = label1[train]
        validate_data = data1[validate]
        validate_label = label1[validate]

        clf.fit(train_data, train_label)
        print clf.classes_
        acc = clf.score(test_data, test_label)
        acc_sum[itr].append(acc)

        preds = clf.predict(test_data)
        cm = CM(test_label,preds)
        cm = normalize(cm.astype(np.float), axis=1, norm='l1')
        k=0
        while k<clx:
            acc_type[k].append(cm[k,k])
            k += 1

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

print 'acc from Md', np.mean(acc_Md)
ave_acc = [np.mean(acc) for acc in acc_sum]
acc_std = [np.std(acc) for acc in acc_sum]

print 'overall acc:', repr(ave_acc)
#print 'acc std:', repr(acc_std)
#print 'acc by type', repr(acc_type)
f = open('pipe_out','w')
f.writelines('%s;\n'%repr(i) for i in acc_type)
f.write('ex in each itr:'+repr(ex)+'\n')
f.write(repr(np.unique(test_label)))
f.close()
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
