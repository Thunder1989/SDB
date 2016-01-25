'''
generate node features for CRF and output to file
===format=====
line1: #ofInstance, #ofFeature, #ofClass
line2~N:
x_i (implicit), Y_ij, name features, delta(oracle_k(x_i), Y_ij)
'''
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color
from scikits.statsmodels.tools.tools import ECDF

import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt

input1 = np.genfromtxt('rice_hour_sdh', delimiter=',')
#input1 = np.genfromtxt('sdh_hour_soda', delimiter=',')
#input1 = np.genfromtxt('soda_hour_rice', delimiter=',')
input21 = np.genfromtxt('keti_hour_sum', delimiter=',')
#input21 = np.genfromtxt('rice_hour_soda', delimiter=',')
input3 = np.genfromtxt('sdh_hour_rice', delimiter=',')
input2 = np.vstack((input21,input3))
fd1 = input1[:,0:-1]
fd2 = input2[:,0:-1]
#train_fd = np.hstack((fd1,fd2))
train_fd = fd1
test_fd = fd2
train_label = input1[:,-1]
#test_label = np.hstack((input2[:,-1],input3[:,-1]))
test_label = input2[:,-1]

#switch src and tgt
fd_tmp = train_fd
train_fd = test_fd
test_fd = fd_tmp
l_tmp = train_label
train_label = test_label
test_label = l_tmp

input1 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_sdh').readlines()]
label = test_label
name = []
for i in input1:
    s = re.findall('(?i)[a-z]{2,}', i)
    name.append(' '.join(s))
cv = CV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()

rf = RFC(n_estimators=100, criterion='entropy')
svm = SVC(kernel='rbf', probability=True)
lr = LR()
bl = [rf, lr, svm] #set of base learner
for b in bl:
    b.fit(train_fd, train_label) #train each base classifier

O_xy = np.empty((0,len(bl)),int)
u_label, remap = np.unique(test_label, return_inverse=True)
for i in xrange(len(test_fd)):
    O_tmp = np.zeros([len(u_label),len(bl)])
    for b,j in zip(bl,xrange(len(bl))):
        o_label = b.predict(test_fd[i])
        O_tmp[u_label==o_label,j] = 1
    O_xy = np.append(O_xy, O_tmp, axis=0)

name_feature = np.repeat(fn,len(u_label),axis=0)
y_index = np.tile(range(len(u_label)),(1,len(fn))).T
#print y_index.shape
#print O_xy.shape
#print name_feature.shape
node_feature = np.hstack([y_index, name_feature, O_xy])
print node_feature.shape

f = open('fn_rice.txt','w')
f.write("%s,%s,%s\n"%(fn.shape[0],fn.shape[1],len(u_label)))
f.writelines(",".join(str(y) for y in remap))
f.write('\n')
np.savetxt(f, node_feature, delimiter=",", fmt='%d')
f.close()

#out = np.hstack([label[:, None], fn])
#np.savetxt("fn_rice.csv", out, delimiter=",")
