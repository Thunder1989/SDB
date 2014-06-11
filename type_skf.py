from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import pylab as pl

input1 = np.genfromtxt('rice_45min', delimiter=',')
data2 = input1[:,0:-1]
label2 = input1[:,-1]
input2 = np.genfromtxt('sdh_45min', delimiter=',')
data1 = input2[:,0:-1]
label1 = input2[:,-1]

'''
loo = LeaveOneOut(len(data))
for train_idx, test_idx in loo:
    pass
'''

ctr = 0
preds = []
fold = 2
skf = StratifiedKFold(label1, n_folds=fold)
sum = []
for train_idx, test_idx in skf:
    '''
    because we want to do inverse k-fold XV
    aka, use 1 fold to train, k-1 folds to test
    so the indexing is inversed
    '''
    train_data = data1[test_idx]
    train_label = label1[test_idx]
    #test_data = data[train_idx]
    #test_label = label[train_idx]
    test_data = data2
    test_label = label2
    #clf = ETC(n_estimators=10, criterion='entropy')
    clf = RFC(n_estimators=100, criterion='entropy')
    #clf = DT(criterion='entropy', random_state=0)
    #clf = Ada(n_estimators=100)
    clf.fit(train_data, train_label)
    #out = tree.export_graphviz(clf, out_file='tree.dot')
    preds = clf.predict(test_data)
    acc = accuracy_score(test_label, preds)
    sum.append(acc)
    print acc

print 'ave acc:', np.mean(sum)
print 'std:', np.std(sum)

cm = CM(test_label,preds)
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
        ax.annotate(str("%.3f"%cm[x][y]), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center')


cls = ['co2','humidity','rmt','stpt','flow','other_t']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix')
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()
