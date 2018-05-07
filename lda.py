import numpy as np
import matplotlib.pyplot as plt
import sys

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

def get_topics(n_topics):

    t0 = time()
    tf = np.genfromtxt('tf.txt', delimiter=',')
    print "feature laoded in %0.3fs." % (time() - t0)

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf.T)
    doc_topic = lda.transform(tf.T)
    print "lda done in %0.3fs." % (time() - t0)

    #tfidf = np.genfromtxt('tfidf.txt', delimiter=',')
    #nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    #doc_topic = nmf.transform(tfidf)
    plt.imshow(doc_topic, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    n_topics = int(sys.argv[1])
    get_topics(n_topics)
