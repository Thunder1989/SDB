from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
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
data1 = input1[:,[0,1,2,3,5,6,7]]
label1 = input1[:,-1]
input2 = np.genfromtxt('sdh_45min', delimiter=',')
data2 = input2[:,[0,1,2,3,5,6,7]]
label2 = input2[:,-1]

run = 10
iteration = 15
init = 20
fold = 10
#loo = LeaveOneOut(len(data))
#skf = StratifiedKFold(label1, n_folds=fold)
kf = KFold(len(label1), n_folds=fold, shuffle=True)
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1

acc_sum = [[] for i in range(run)]
indi_acc =[[] for i in range(6)]
#clf = ETC(n_estimators=10, criterion='entropy')
clf = RFC(n_estimators=50, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = Ada(n_estimators=100)
#clf = SVC(kernel='linear')

for itr in range(fold):
    train = np.hstack((folds[(itr+x)%10] for x in range(3)))
    validate = np.hstack((folds[(itr+x)%10] for x in range(3,6)))
    test = np.hstack((folds[(itr+x)%10] for x in range(6,fold)))

    test_data = data1[test]
    test_label = label1[test]

    for ctr in range(iteration):
        #print 'running fold %d iter %d'%(itr, ctr)
        train_data = data1[train]
        train_label = label1[train]
        validate_data = data1[validate]
        validate_label = label1[validate]

        clf.fit(train_data, train_label)
        preds = clf.predict(test_data)
        acc = accuracy_score(test_label, preds)
        acc_sum[itr].append(acc)

        '''
        #acc by type
        cm = CM(validate_label,preds)
        cm = normalize(cm.astype(np.float), axis=1, norm='l1')

        k=0
        while k<6:
            indi_acc[k].append(cm[k,k])
            k += 1
        '''

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
        print 'worng #', len(wrong)

        '''
        #sort and pick the 1st one with largest H
        wrong = sorted(wrong, key=lambda x: x[3], reverse=True)
        idx = 0

        '''
        #randomly pick one
        idx = random.randint(0,len(wrong)-1)

        '''
        #pick the one with H most close to 0.5
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

ave_acc = []
acc_std = []
for i in range(iteration):
    l = [acc[i] for acc in acc_sum]
    ave_acc.append(np.mean(l))
    acc_std.append(np.std(l))

#indi_ave_acc = [np.mean(i) for i in indi_acc]
#indi_ave_acc_std = [np.std(i) for i in indi_acc]
#print 'ave acc/type:', repr(indi_ave_acc)
#print 'acc std/type:', indi_ave_acc_std
print 'overall acc:', repr(ave_acc)
#print 'ave cc :', np.mean(acc_sum)
print 'acc std:', repr(acc_std)
