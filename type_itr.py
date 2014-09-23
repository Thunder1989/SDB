from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.svm import SVC
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
#label = [1,2,4,6,7,8]

iteration = 80
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
    #cut train to one example
    validate = np.append(validate,train[2:])
    train = train[:2]

    test = np.hstack((folds[(fd+x)%fold] for x in range(30,fold)))
    test_data = data1[test]
    test_label = label1[test]

    for itr in range(iteration):
        #if itr%10==0:
        #    print 'running fold %d iter %d'%(fd, itr)
        train_data = data1[train]
        train_label = label1[train]
        validate_data = data1[validate]
        validate_label = label1[validate]

        clf.fit(train_data, train_label)
        preds = clf.predict(test_data)
        acc = clf.score(test_data, test_label)
        acc_sum[itr].append(acc)

        #plot confusion matrix, for debugging
        cm_ = CM(test_label,preds)
        cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
        #if itr==0 or itr==iteration-1:
        if True:
            pre = precision_score(test_label, preds, average=None)
            rec = recall_score(test_label, preds, average=None)
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

        #statistics by type
        pre = precision_score(test_label, preds, average=None)
        rec = recall_score(test_label, preds, average=None)
        k=0
        while k<6:
            acc_type[k][itr].append(cm[k,k])
            precision_type[k][itr].append(pre[k])
            recall_type[k][itr].append(rec[k])
            k += 1


        #entropy based example selection block
        #compute entropy for each instance and rank
        label_pr = np.sort(clf.predict_proba(validate_data)) #sort in ascending order
        preds = clf.predict(validate_data)
        res = []
        for h,i,j,pr in zip(validate,validate_label,preds,label_pr):
            entropy = np.sum(-p*math.log(p,6) for p in pr if p!=0)
            if len(pr)<2:
                margin = 1
            else:
                margin = pr[-1]-pr[-2]
            res.append([h,i,j,entropy,margin])
        #print 'iter', itr, 'wrong #', len(wrong)

        '''
        #Entropy-based, sort and pick the one with largest H
        res = sorted(res, key=lambda x: x[-2], reverse=True)
        idx = 0

        '''
        #Margin-based, sort and pick the one with least margin
        res = sorted(res, key=lambda x: x[-1])
        idx = 0
        '''

        #least confidence based
        tmp = sorted(label_pr, key=lambda x: x[-1])
        idx = 0


        #Expectation-based, pick the one with H most close to 0.5
        for i in res:
            i[-2] = abs(i[-2]-0.5)
        res = sorted(res, key=lambda x: x[3])
        idx = 0


        #randomly pick one
        idx = random.randint(0,len(res)-1)
        '''

        elmt = res[idx][0]
        print 'running fold %d iter %d'%(fd, itr)
        print label1[elmt]

        '''
        #minimal future expected error
        loss = []
        label_pr = clf.predict_proba(validate_data)
        clx = clf.classes_
        for i, pr in zip(validate, label_pr):
            #print 'validate ex#', i
            #print 'pr vector', pr
            new_train = np.append(train,i)
            new_validate = validate[validate!=i]

            new_train_data = data1[new_train]
            new_train_label = label1[new_train]
            new_validate_data = data1[new_validate]
            new_validate_label = label1[new_validate]

            err = 0
            for j in range(len(pr)):
                #compute the sum of confidence for the rest of examples in validate set
                #on the new re-trained model after each possbile labeling (x,y_i) of i is added to the train set
                if pr[j]==0:
                    continue

                new_train_label[-1] = clx[j]
                clf.fit(new_train_data, new_train_label)
                confidence_sum = np.sum(1-np.sort(clf.predict_proba(new_validate_data))[:,-1])
                err += pr[j]*confidence_sum

            loss.append([i,err])

        loss = sorted(loss, key=lambda x: x[-1])
        #print loss

        elmt = loss[0][0]
        '''

        #remove the item from validate set
        #add it to train set
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
'''
print '=================================='
print 'acc by type:', repr(ave_acc_type)
print '=================================='
print 'precision by type:', repr(ave_pre)
print '=================================='
print 'recall by type:', repr(ave_rec)
'''
