import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from sklearn import metrics

df = pd.read_csv('testdata/testdata.csv')
tweets = df['text']
sentimen = df['sentiment']

vec_pickle = open('Model/tfidf.p', 'rb')
vec = pickle.load(vec_pickle)
vec_pickle.close()

clf_pickle = open('Model/svmsent.p', 'rb')
clf_svm = pickle.load(clf_pickle)
clf_pickle.close()

clf2_pickle = open('Model/6nnsent.p', 'rb')
clf_knn = pickle.load(clf2_pickle)
clf2_pickle.close()

###################################################################################

train_vectors_tfidf = vec.transform(tweets)
prediction = clf_svm.predict(train_vectors_tfidf)
prediction2 = clf_knn.predict(train_vectors_tfidf)
# print('Jumlah prediksi: ' + str(len(prediction)))
# print('Jumlah prediksi: ' + str(len(prediction2)))

###################################################################################

# Simpan hasil sentimen
# hasilsvm = []
# for sentiment in prediction:
#     hasilsvm.append(sentiment)
#
# file1 = open('Hasil/hasilsvm.txt', 'w')
#
# for sentiment1 in hasilsvm:
#     file1.write(sentiment1 + "\n")
#
# hasilknn = []
# for sentiment in prediction2:
#     hasilknn.append(sentiment)
#
# file2 = open('Hasil/hasil10nn.txt', 'w')
#
# for sentiment2 in hasilknn:
#     file2.write(sentiment2 + "\n")
#
# print("Bot: Hasil analisis sentimen telah disimpan.")
# print("Me: Maacih qaqa.")

###################################################################################

'''Untuk menghitung akurasi classifier'''

# Scoring SVM
acc_svm = metrics.accuracy_score(sentimen, prediction)
print("SVM accuracy:    %0.3f" %acc_svm)
rec_svm = metrics.recall_score(sentimen, prediction, average='weighted')
print("SVM recall:    %0.3f" %rec_svm)
pre_svm = metrics.precision_score(sentimen, prediction, average='weighted')
print("SVM precision:    %0.3f" %pre_svm)
f1_svm = metrics.f1_score(sentimen, prediction, average='weighted')
print("SVM F1 Score:    %0.3f" %f1_svm)

# Scoring KNN
acc_knn = metrics.accuracy_score(sentimen, prediction2)
print("6NN accuracy:    %0.3f" %acc_knn)
rec_knn = metrics.recall_score(sentimen, prediction2, average='weighted')
print("6NN recall:    %0.3f" %rec_knn)
pre_knn = metrics.precision_score(sentimen, prediction2, average='weighted')
print("6NN precision:    %0.3f" %pre_knn)
f1_knn = metrics.f1_score(sentimen, prediction2, average='weighted')
print("6NN F1 Score:    %0.3f" %f1_knn)

###################################################################################

import time
start_time = time.time()
#
# clf_pickle = open('Model/svmsent.p', 'rb')
# clf_svm = pickle.load(clf_pickle)
# clf_pickle.close()
#
# acc_svm = metrics.accuracy_score(sentimen, prediction)
# print("SVM accuracy:    %0.3f" %acc_svm)
# rec_svm = metrics.recall_score(sentimen, prediction, average='weighted')
# print("SVM recall:    %0.3f" %rec_svm)
# pre_svm = metrics.precision_score(sentimen, prediction, average='weighted')
# print("SVM precision:    %0.3f" %pre_svm)
# f1_svm = metrics.f1_score(sentimen, prediction, average='weighted')
# print("SVM F1 Score:    %0.3f" %f1_svm)
#
end_time = time.time()
print("Waktu proses: %s second." %(end_time - start_time))

###################################################################################

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

clf_svm = SVC(kernel="linear")
clf_knn = KNeighborsClassifier(n_neighbors=2, metric="cosine")

def crossval(clf, X, y, K):
    validator = KFold(n_splits=K, shuffle=True, random_state=0)
    score = cross_val_score(clf, X, y, cv=validator, scoring='accuracy')
    score1 = cross_val_score(clf, X, y, cv=validator, scoring='precision_weighted')
    score2 = cross_val_score(clf, X, y, cv=validator, scoring='recall_weighted')
    score3 = cross_val_score(clf, X, y, cv=validator, scoring='f1_weighted')
    print("Hasil akurasi: ", score)
    print("Rata-rata akurasi: ", np.mean(score))
    print("Hasil precision: ", score1)
    print("Rata-rata precision: ", np.mean(score1))
    print("Hasil recall: ", score2)
    print("Rata-rata recall: ", np.mean(score2))
    print("Hasil f1: ", score3)
    print("Rata-rata f1: ", np.mean(score3))

# crossval(clf_svm, train_vectors_tfidf, sentimen, 10)
# crossval(clf_knn, train_vectors_tfidf, sentimen, 10)

##################################################################################
import time
start_time = time.time()
kfcv_SVM = crossval(clf_svm, train_vectors_tfidf, sentimen, 10)
end_time = time.time()
print("SVM: %s second." %(end_time - start_time))

start_time = time.time()
kfcv_SVM = crossval(clf_knn, train_vectors_tfidf, sentimen, 10)
end_time = time.time()
print("KNN: %s second." %(end_time - start_time))