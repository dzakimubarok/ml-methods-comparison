import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('traindata/trained_data.csv')
# print(df['text'])

vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 1),
                             min_df=1,
                             max_df=0.95,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors_tfidf = vectorizer.fit_transform(df['text'])
save_vec = open("tfidf.p", "wb")
pickle.dump(train_vectors_tfidf, save_vec, pickle.HIGHEST_PROTOCOL)
save_vec.close()

neigh = KNeighborsClassifier(n_neighbors=6, metric="cosine")
neigh.fit(train_vectors_tfidf, df['sentiment'])
save_clf2 = open("6nnsent.p", "wb")
pickle.dump(neigh, save_clf2, pickle.HIGHEST_PROTOCOL)
save_clf2.close()


def svmclassifier(train_vectors_tfidf, train_label):
    clf = SVC(kernel="linear")
    linear_clf = clf.fit(train_vectors_tfidf, train_label)
    save_clf = open("svmsent.p", "wb")
    pickle.dump(clf, save_clf, pickle.HIGHEST_PROTOCOL)
    save_clf.close()
    return linear_clf

svmclassifier(train_vectors_tfidf, df['sentiment'])