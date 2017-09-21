'''
doing the type classification with point name
'''
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
from sklearn.utils import shuffle
import numpy as np
import re
import math
import pylab as pl

input1 = [i.strip().split('+')[-1][:-5] for i in open('rice_pt_soda').readlines()]
input2 = np.genfromtxt('rice_hour_soda', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-5] for i in open('soda_pt_rice').readlines()]
input4 = np.genfromtxt('soda_hour_rice', delimiter=',')
label1 = input2[:,-1]
label2 = input4[:,-1]
#input3, label = shuffle(input3, label)

name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
#print name

'''
fold = 10
#clx = len(np.unique(label))
clx = 10
skf = StratifiedKFold(label, n_folds=fold, shuffle=True)
acc_sum = []
indi_acc =[[] for i in range(clx)]
'''
clf = RFC(n_estimators=100, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = SVC(kernel='linear', probability=True)
#clf = GNB()

#vc = CV(token_pattern='[a-z]{2,}')
#vc = TV(analyzer='char_wb', ngram_range=(3,4))
vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#fn = vc.fit_transform(name).toarray()
#fn = vc.fit_transform(input3).toarray()
#print vc.get_feature_names()
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
vc.fit(name)
data1 = vc.transform(name).toarray()
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))
data2 = vc.transform(name).toarray()
train_data = data2
train_label = label2
test_data = data1
test_label = label1
clf.fit(train_data, train_label)
preds = clf.predict(test_data)
print accuracy_score(test_label, preds)

for train_idx, test_idx in skf:
#for test_idx,train_idx in skf:
    '''
    because we want to do inverse k-fold XV
    aka, use 1 fold to train, k-1 folds to test
    so the indexing is inversed
    '''
    train_data = fn[test_idx]
    train_label = label[test_idx]
    test_data = fn[train_idx]
    test_label = label[train_idx]
    #train_data = data1
    #train_label = label1
    #test_data = data2
    #test_label = label2
    clf.fit(train_data, train_label)
    preds = clf.predict(test_data)
    acc = accuracy_score(test_label, preds)
    acc_sum.append(acc)

    cm_ = CM(test_label,preds)
    cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
    k=0
    while k<clx:
        indi_acc[k].append(cm[k,k])
        k += 1

    for i,j,k in zip(test_label, preds, train_idx):
        if i==1 and j==1:
                pass
            #print name[k]

    '''
    #debug co2 instances
    print '===================='
    importance += clf.feature_importances_
    for i,j in zip(train_label, test_idx):
        if i==1:
                print 'train id:', j+1
    #for i,j,k in zip(test_label, preds, train_idx):
    for i,j,k in zip(test_label, preds, range(len(test_label))):
        if i==1 and i!=j:
            print '%d-%d'%(k+1,j)

    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    for x in xrange(len(cm)):
        for y in xrange(len(cm)):
            ax.annotate(str("%.3f(%d)"%(cm[x][y],cm_[x][y])), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center')


    cls = ['co2','humidity','rmt','stpt','flow','other_t']
    pl.xticks(range(len(cm)),cls)
    pl.yticks(range(len(cm)),cls)
    pl.title('Confusion matrix (%.3f)'%acc)
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()
    '''

#print importance/run
indi_ave_acc = [np.mean(i) for i in indi_acc]
#indi_ave_acc_std = [np.std(i) for i in indi_acc]
print 'ave acc/type:', indi_ave_acc
#print 'acc std/type:', indi_ave_acc_std

'''
log = open('log_r_s','w')
pr = clf.predict_proba(test_data)
k=1
for i,j,pr in zip(test_label,preds,pr):
    entropy = np.sum(-p*math.log(p,6) for p in pr if p!=0)
    log.write('[%d]-%d:%d'%(k,i,j))
    log.write('-%s'%pr)
    log.write('-%.3f\n'%entropy)
    k += 1
log.close()
'''
print 'ave acc:', np.mean(acc_sum)
#print 'std:', np.std(acc_sum)


cm_ = CM(test_label,preds)
#print cm_.shape
cm = normalize(cm.astype(np.float), axis=1, norm='l1')
#print cm
#cm /= cm.astype(np.float).sum(axis=1)
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)

for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f(%d)"%(cm[x][y],cm_[x][y])), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10)

#mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
#cls_id =np.unique(test_label)
#cls = []
#for c in cls_id:
#   cls.append(mapping[c])

mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
test_cls =np.unique(np.hstack((train_label, test_label)))
#print len(test_cls)
cls = []
for c in test_cls:
    cls.append(mapping[int(c)])
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix (%.3f)'%acc)
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()
