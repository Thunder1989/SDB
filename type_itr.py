from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import math
import random
import pylab as pl

input1 = np.genfromtxt('rice_45min', delimiter=',')
data1 = input1[:,[0,1,2,3,5,6,7]]
label1 = input1[:,-1]
input2 = np.genfromtxt('sdh_45min', delimiter=',')
data2 = input2[:,[0,1,2,3,5,6,7]]
label2 = input2[:,-1]

iteration = 50
init = 20
fold = 60
#loo = LeaveOneOut(len(data))
#skf = StratifiedKFold(label1, n_folds=fold)
kf = KFold(len(label1), n_folds=fold, shuffle=True)
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1

acc_sum = [[] for i in range(iteration)]
acc_type = [[[] for i in range(iteration)] for i in range(6)]
precision_type = [[[] for i in range(iteration)] for i in range(6)]
recall_type = [[[] for i in range(iteration)] for i in range(6)]
#clf = ETC(n_estimators=10, criterion='entropy')
clf = RFC(n_estimators=50, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = Ada(n_estimators=100)
#clf = SVC(kernel='linear')

for fd in range(fold):
    train = np.hstack((folds[(fd+x)%fold] for x in range(1)))
    validate = np.hstack((folds[(fd+x)%fold] for x in range(1,30)))
    test = np.hstack((folds[(fd+x)%fold] for x in range(30,fold)))
    test_data = data1[test]
    test_label = label1[test]

    for itr in range(iteration):
        #print 'running fold %d iter %d'%(fd, itr)
        train_data = data1[train]
        train_label = label1[train]
        validate_data = data1[validate]
        validate_label = label1[validate]

        clf.fit(train_data, train_label)
        preds = clf.predict(test_data)
        acc = accuracy_score(test_label, preds)
        acc_sum[itr].append(acc)

        #acc by type
        cm_ = CM(test_label,preds)
        cm = normalize(cm_.astype(np.float), axis=1, norm='l1')

        '''
        #for debugging
        if itr==0 or itr==iteration-1:
            print cm_
            pre = precision_score(test_label, preds, average=None)
            rec = recall_score(test_label, preds, average=None)
            print pre
            print rec
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

        pre = precision_score(test_label, preds, average=None)
        rec = recall_score(test_label, preds, average=None)
        k=0
        while k<6:
            acc_type[k][itr].append(cm[k,k])
            precision_type[k][itr].append(pre[k])
            recall_type[k][itr].append(rec[k])
            k += 1

        #compute entropy for each instance and rank
        label_pr = clf.predict_proba(validate_data)
        preds = clf.predict(validate_data)
        correct = []
        wrong = []
        for h,i,j,pr in zip(validate,validate_label,preds,label_pr):
            entropy = np.sum(-p*math.log(p,6) for p in pr if p!=0)
            if i==j:
                correct.append([h,i,j,entropy])
            else:
                wrong.append([h,i,j,entropy])
        #print 'preds size', len(preds)
        #print 'iter', itr, 'wrong #', len(wrong)


        #H-based, sort and pick the 1st one with largest H
        wrong = sorted(wrong, key=lambda x: x[3], reverse=True)
        idx = 0
        '''

        #randomly pick one
        idx = random.randint(0,len(wrong)-1)


        #E-based, pick the one with H most close to 0.5
        for i in wrong:
            i[-1] = abs(i[-1]-0.5)
        wrong = sorted(wrong, key=lambda x: x[3])
        idx = 0
        '''

        elmt = wrong[idx][0]
        #print 'ex H:', wrong[idx][-1]
        #remove the item on the top of the ranked wrong list from the training set
        #add it to test set
        train = np.append(train, elmt)
        validate = validate[validate!=elmt]
        #train_idx.append(elmt)
        #test_idx.remove(elmt)

ave_acc = [np.mean(acc) for acc in acc_sum]
acc_std = [np.std(acc) for acc in acc_sum]
ave_acc_type = [[] for i in range(6)]
ave_pre = [[] for i in range(6)]
ave_rec = [[] for i in range(6)]
for i in range(6):
    ave_acc_type[i] = [np.mean(a) for a in acc_type[i]]
    ave_pre[i] = [np.mean(p) for p in precision_type[i]]
    ave_rec[i] = [np.mean(r) for r in recall_type[i] ]

print 'overall acc:', repr(ave_acc)
print 'acc std:', repr(acc_std)
print '=================================='
print 'acc by type:', repr(ave_acc_type)
print '=================================='
print 'precision by type:', repr(ave_pre)
print '=================================='
print 'recall by type:', repr(ave_rec)
