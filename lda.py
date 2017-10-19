import numpy as np
import matplotlib.pyplot as plt

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_topics = 10

t0 = time()
tf = np.genfromtxt('tf.txt', delimiter=',')
print "feature laoded in %0.3fs." % (time() - t0)

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
#lda.fit(tf)
#doc_topic = lda.transform(tf)
#print "lda done in %0.3fs." % (time() - t0)


tfidf = np.genfromtxt('tfidf.txt', delimiter=',')
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
doc_topic = nmf.transform(tfidf)
plt.imshow(doc_topic, cmap='hot', interpolation='nearest')
plt.show()
