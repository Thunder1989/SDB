from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import math
import pylab as pl

#input1 = np.genfromtxt('rice_45min_raw_sliding', delimiter=',')
input1 = np.genfromtxt('rice_day_wpeak', delimiter=',')
input3 = np.genfromtxt('rice_hour_sum', delimiter=',')
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
input5 = np.genfromtxt('rice_45min', delimiter=',')
input6 = np.genfromtxt('rice_diu_sum', delimiter=',')
input2 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
#data1 = input1[:,[0,1,2,3,5,6,7,9,10,11]]
#fd = input1[:,[0,1,2,3,5,6,7]]
fd1 = input3[:,0:-1]
fd2 = input1[:,0:-1]
fd3 = input6[:,0:-1]
fd = np.hstack((fd1,fd2,fd3))
ipt = np.array(
[ 0.28176179,0.16062254,0.19488627,0.24487843,0.30255501,0.35443289
    ,0.31540133,0.24201478,0.31132315,0.36955541,0.32088347,0.35540954
    ,0.29804991,0.39299528,0.31813591,0.28859625,0.03791034,0.16494693
    ,0.5002103, 0.15306943,0.07011232,0.13198681,0.01576,0.02375381
    ,0.03632773,0.13501897,0.01659925,0.02392811,0.31466717,0.21641054
    ,0.2536133, 0.31012377,0.31670794,0.38888536,0.33204075,0.30747534
    ,0.03674301,0.2202327, 0.52015362,0.19612566,0.17695897,0.1192621
    ,0.11778389,0.11168993])
ipt = np.array(
[ 0.1005068, 0.02616343,0.04972112,0.02907851,0.08717033,0.090715
    ,0.07524639,0.02660152,0.09332957,0.11589702,0.0748438, 0.02153562
    ,0.08392234,0.12446451,0.07793186,0.02234248,0.01374349,0.03109567
    ,0.13628661,0.03589373,0.0082536, 0.01123168,0.00406529,0.00722405
    ,0.00329813,0.00938755,0.00436952,0.00775867,0.10933382,0.04171875
    ,0.05652619,0.03300635,0.08361292,0.12047689,0.08167406,0.02133249
    ,0.01501659,0.031198,0.17034366,0.03114427,0.01860756,0.05220669
    ,0.03376522,0.0492824, 0.1021459, 0.0414551, 0.071813,0.04910903
    ,0.07994753,0.09302449,0.1120499, 0.03989825,0.06550381,0.08258047
    ,0.07684329,0.04734925,0.09362827,0.11146772,0.09932905,0.05286112
    ,0.23371136,0.04629134,0.01745012,0.04364592,0.03342541,0.02552217
    ,0.03417656,0.04324286,0.03880639,0.04164729,0.15718462,0.17438275
    ,0.08040356,0.05459401,0.05814264,0.09459399,0.08483518,0.12513027
    ,0.08457465,0.03199699,0.281243,0.02805951,0.06668088,0.02087521
    ,0.017792,0.01575326,0.02681766,0.01752114,0.16420835,0.14096078
    ,0.1611314, 0.12041089,0.08189578,0.1395546, 0.21022586,0.16214429
    ,0.09528004,0.11260615,0.04431597,0.05815294,0.07876316,0.02922815
    ,0.05258999,0.03132234,0.05308002,0.12650086,0.0872874, 0.09561468
    ,0.05228488,0.05885853,0.06762833,0.10577131,0.06700939,0.1427806
    ,0.11528041,0.0322758, 0.08850744,0.0203266, 0.02052234,0.02069731
    ,0.01624093,0.01752225,0.0250615, 0.07739528,0.02538702,0.02123008
    ,0.07031688,0.06293785,0.05781509,0.03261409,0.04829227,0.02160311
    ,0.06143983,0.11019975,0.07740351,0.09048957,0.16319507,0.0198428
    ,0.02132689,0.01824602,0.02613648,0.02717668,0.02714104,0.01616273
    ,0.01289072,0.04900295,0.03783948,0.0387874, 0.07344496,0.16149891
    ,0.01980653,0.01016751,0.03515208,0.04905213,0.01072137,0.01041158])
fd = fd[:,ipt>float(10)/156]
#fd = fd[:,ipt>0.15]
#index = data1.shape[1]*2/3
#data1 = input1[:,0:index]
#fd = input5[:,[0,1,2,3,5,6,7]]
label = input4[:,-1]
input3 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input4 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_forrice').readlines()]
fd1 = input3[:,[0,1,2,3,5,6,7]]
label1 = input3[:,-1]

fold = 10
clx = 9
#skf = KFold(len(label), n_folds=fold, shuffle=True)
skf = StratifiedKFold(label, n_folds=fold, shuffle=True)
acc_sum = []
indi_acc =[[] for i in range(clx)]
clf1 = RFC(n_estimators=100, criterion='entropy')
#pca = PCA(n_components=100)
#pca.fit(fd)
#print pca.explained_variance_ratio_
#print pca.n_components_
#model = SVR(kernel="linear")
#selector = RFE(model, 500)
#selector = selector.fit(fd, label)
#fd = fd[:,selector.support_]
#print fd.shape
#clf = DT(criterion='entropy', random_state=0)
clf2 = SVC(kernel='linear')
vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
fn = vc.fit_transform(input4).toarray()

loop = 0
run = fold
importance = np.zeros(fd.shape[1])
a_r = 0
a_w = 0
d_md_r = 0
d_mn_r = 0
fi = np.zeros(fd.shape[1])
for loop in xrange(10):
    for train_idx, test_idx in skf:
        '''
        because we want to do inverse k-fold XV
        aka, use 1 fold to train, k-1 folds to test
        so the indexing is inversed
        '''
        train_fd = fd[train_idx]
        train_fn = fn[train_idx]
        train_label = label[train_idx]
        test_fd = fd[test_idx]
        test_fn = fn[test_idx]
        test_label = label[test_idx]
        #test_data = data2
        #test_label = label2
        clf1.fit(train_fd, train_label)
        fi += clf1.feature_importances_
        clf2.fit(train_fn, train_label)
        #print clf.classes_
        #print clf.feature_importances_
        preds_fd = clf1.predict(test_fd)
        preds_fn = clf2.predict(test_fn)
        acc = accuracy_score(test_label, preds_fd)
        acc_sum.append(acc)
        #importance += clf.feature_importances_
        #print acc

        #compute the statistics per model on rights and wrongs
        for i,j,k in zip(preds_fd, preds_fn, test_label):
            if i==j:
                if i==k:
                    a_r += 1
                else:
                    a_w += 1
            else:
                if i==k:
                    d_md_r += 1
                else:
                    d_mn_r += 1

        '''
        cm_ = CM(test_label,preds)
        cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
        k=0
        while k<clx:
            indi_acc[k].append(cm[k,k])
            k += 1

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

num = len(test_label)
#print float(a_r)/(run*num)
#print float(a_w)/(run*num)
#print float(d_md_r)/(run*num)
#print float(d_mn_r)/(run*num)

#for i,j,k in zip(test_label, preds, train_idx):
#    if j==4 and i!=j:
#        print '%d-%d'%(k+1,i)

#print importance/run
#indi_ave_acc = [np.mean(i) for i in indi_acc]
#indi_ave_acc_std = [np.std(i) for i in indi_acc]
#print 'ave acc/type:', indi_ave_acc
#print 'acc std/type:', indi_ave_acc_std
print 'ave acc:', np.mean(acc_sum)
#print 'std:', np.std(acc_sum)
print fi/10

cm_ = CM(test_label,preds_fd)
#print cm
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
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

mapping = {1:'co2',2:'humidity',3:'pressure',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',14:'pressure',15:'timer',17:'MAT',18:'C enter',19:'C leave',20: 'dew pt',21:'occu'}
cls_id = np.unique(np.hstack((test_label,preds_fd)))
cls = []
for c in cls_id:
    cls.append(mapping[c])
#cls = ['co2','humidity','pressure','rmt','status','stpt','flow','other T','occu']
#cls = ['rmt','pos','stpt','flow','other_t','ctrl','spd','sta','pressure','tmr'] #soda
#cls = ['rmt','pos','stpt','flow','other_t','pwr','ctrl','occu','spd','sta'] #sdh
#cls = ['co2','humidity','rmt','status','stpt','flow','HW sup','HW ret','CW sup','CW ret','SAT','RAT','MAT','C enter','C leave','occu']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix (%.3f)'%acc)
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()

