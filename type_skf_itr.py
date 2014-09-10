from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import math
import random
import pylab as pl

input1 = np.genfromtxt('rice_45min', delimiter=',')
data1 = input1[:,0:-1]
label1 = input1[:,-1]
input2 = np.genfromtxt('sdh_45min', delimiter=',')
data2 = input2[:,0:-1]
label2 = input2[:,-1]

'''
loo = LeaveOneOut(len(data))
for train_idx, test_idx in loo:
    pass
'''

ctr = 0
preds = []
fold = 10
skf = StratifiedKFold(label1, n_folds=fold)
acc_sum = []
indi_acc =[[] for i in range(6)]
#clf = ETC(n_estimators=10, criterion='entropy')
clf = RFC(n_estimators=50, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = Ada(n_estimators=100)
#clf = SVC(kernel='linear')
for train_idx, test_idx in skf:
    '''
    because we want to do inverse k-fold XV
    aka, use 1 fold to train, k-1 folds to test
    so the indexing is inversed
    '''
    for ctr in range(20):
        print len(test_idx), 'training exs'
        print len(train_idx), 'testing exs'
        train_data = data1[test_idx]
        train_label = label1[test_idx]
        test_data = data1[train_idx]
        test_label = label1[train_idx]
        clf.fit(train_data, train_label)
        preds = clf.predict(test_data)

        #overall acc
        acc = accuracy_score(test_label, preds)
        acc_sum.append(acc)
        #print acc

        #acc by type
        cm = CM(test_label,preds)
        cm = normalize(cm.astype(np.float), axis=1, norm='l1')
        k=0
        while k<6:
            indi_acc[k].append(cm[k,k])
            k += 1

        #compute entropy for each instance and rank
        label_pr = clf.predict_proba(test_data)
        correct = []
        wrong = []
        for h,i,j,pr in zip(train_idx,test_label,preds,label_pr):
            entropy = np.sum(-p*math.log(p,6) for p in pr if p!=0)
            if i==j:
                correct.append([h,i,j,entropy])
            else:
                wrong.append([h,i,j,entropy])
        '''
        #sort and pick the 1st one with largest H
        wrong = sorted(wrong, key=lambda x: x[3], reverse=True)
        idx = 0

        #randomly pick one
        idx = random.randint(0,len(wrong)-1)
        '''

        #pick the one with H most close to 0.5
        for i in wrong:
            i[-1] = abs(i[-1]-0.5)
        wrong = sorted(wrong, key=lambda x: x[3])
        idx = 0

        elmt = wrong[idx][0]
        print 'ex H:', wrong[idx][-1]
        print '================='
        #remove the item on the top of the ranked wrong list from the training set
        #add it to test set
        test_idx = np.append(test_idx, elmt)
        train_idx = train_idx[train_idx!=elmt]

    break
indi_ave_acc = [np.mean(i) for i in indi_acc]
#indi_ave_acc_std = [np.std(i) for i in indi_acc]
#print 'ave acc/type:', repr(indi_ave_acc)
#print 'acc std/type:', indi_ave_acc_std
print 'overall acc:', repr(acc_sum)
#print 'ave cc :', np.mean(acc_sum)
#print 'acc std:', np.std(acc_sum)
