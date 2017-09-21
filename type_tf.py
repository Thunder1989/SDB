'''
doing the type classification with point name
'''
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
from collections import defaultdict
import numpy as np
import math
import operator
import pylab as pl

input1 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-4] for i in open('rice_pt_forsdh').readlines()]
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
data2 = input2[:,[0,1,2,3,5,6,7]]
data1 = input4[:,[0,1,2,3,5,6,7]]
label2 = input2[:,-1]
label1 = input4[:,-1]

fold = 2
clx = 15
skf = StratifiedKFold(label1, n_folds=fold)
acc_sum = []
indi_acc =[[] for i in range(clx)]
#clf = ETC(n_estimators=10, criterion='entropy')
clf = RFC(n_estimators=50, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = Ada(n_estimators=100)
#clf = SVC(kernel='linear')
#clf = GNB()

vc = CV(token_pattern='[a-z]{2,}')
#vc = TV(token_pattern='[a-z]{2,}')
#vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#data1 = vc.fit_transform(input3).toarray()
#vc.fit(input1)
#data1 = vc.transform(input1).toarray()
#data2 = vc.transform(input3).toarray()
data = defaultdict(list)
for train_idx, test_idx in skf:
    '''
    because we want to do inverse k-fold XV
    aka, use 1 fold to train, k-1 folds to test
    so the indexing is inversed
    '''
    train_data = data1[test_idx]
    train_label = label1[test_idx]
    test_data = data1[train_idx]
    test_label = label1[train_idx]
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
    input3 = np.asarray(input3)
    for i,j in zip(preds,input3[train_idx]):
        data[i].append(j)

out = open('tmp_out','w')
k = 0
for i,j in data.items():
    v = vc.fit_transform(j).toarray()
    f = v.sum(axis=0)
    t = vc.get_feature_names()
    count = zip(t,f)
    count = sorted(count, key=operator.itemgetter(1), reverse=True)
    #print '=== %s (%s) ==='%(i, data.shape[0])
    out.write('=== %s (%d, %.3f) ===\n'%(i, v.shape[0], np.mean(indi_acc[k])))
    out.writelines('%s\n'%repr(c) for c in count)
    #for c in count:
        #print c
    #d = sorted(d.items(), key=operator.itemgetter(1))
    #print i,'===', d
    k+=1

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

print 'ave acc:', np.mean(acc_sum)
#print 'std:', np.std(acc_sum)

cm_ = CM(test_label,preds)
#print cm
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

cls = ['co2','humidity','rmt','status','stpt','flow','HW sup','HW ret','CW sup','CW ret','SAT','RAT','MAT','C enter','C leave','occu']
#cls = ['co2','humidity','rmt','stpt','flow','other T']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix (%.3f)'%acc)
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
#pl.show()
