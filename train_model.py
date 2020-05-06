# import the necessary packages
from time import time
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
#modeller icin kutuphaneler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())
#data1 = pickle.loads(open("output/embeddings_train.pickle", "rb").read())
#data2 = pickle.loads(open("output/embeddings_test.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
#labels_train = le.fit_transform(data1["names"])
#labels_test = le.fit_transform(data2["names"])

'''
scaler = MinMaxScaler(feature_range=(0, 1)).fit(data1["embeddings"])
data["embeddings"] = scaler.transform(data["embeddings"])
data1["embeddings"] = scaler.transform(data1["embeddings"])
data2["embeddings"] = scaler.transform(data2["embeddings"])
'''

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
t0 = time()


# modeller
# logistic regression
reg = LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=1, fit_intercept=True, intercept_scaling=1,
                         class_weight='balanced', random_state=None, solver='lbfgs', max_iter=300,
                         multi_class='multinomial',
                         verbose=0, warm_start=False)

# K-nn KNeighborsClassifier

nc = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                          metric='minkowski', metric_params=None, n_jobs=None)

# Support vector machine
svm = svm.SVC(C=1, decision_function_shape='ovr',max_iter=-1, class_weight=None,
                 verbose=False, gamma='scale', coef0=0.0, cache_size=200, degree=3, random_state=None
                 , shrinking=True, tol=0.0001, kernel="linear", probability=True)

# Decision tree
dtree = tree.DecisionTreeClassifier(criterion ='gini', random_state=None, max_depth = 100,
                                         min_samples_split=2 ,max_leaf_nodes = None, min_samples_leaf =1,
                                         class_weight = None)

# Naive bayes
nb = GaussianNB(priors=None, var_smoothing=1e-09)

# Gradient Boosting Classifier
gbc = ensemble.GradientBoostingClassifier()

# Random Forest
rf = RandomForestClassifier(n_estimators='warn', max_depth=None,min_samples_split=2, min_samples_leaf=1,class_weight=None, random_state=3)

# Linear Discrinant Analysis
recognizer = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                 solver='svd', store_covariance=False, tol=0.3)

svm.fit(data["embeddings"], labels)
print("Train Score: ",svm.score(data["embeddings"], labels))
#print("Test Score: ",svm.score(data2["embeddings"], labels_test))
cognizer_predict = svm.predict(data2["embeddings"])

'''
#hatalari bulma
imagePaths = list(paths.list_images("test"))
count = 0
for i in range(len(recognizer_predict)):
    if (count == 4):
        count = 0
    if(recognizer_predict[i] != labels_test[i]):
        print(imagePaths[i])
    count += 1
'''
print("Egitim Suresi %0.3fs" % (time() - t0))


#print(precision_score(labels_test, recognizer_predict, average='weighted'))
#print(recall_score(labels_test, recognizer_predict, average='weighted'))
#print(f1_score(labels_test, recognizer_predict, average='weighted'))



#scores = cross_val_score(svm, data1["embeddings"], labels_train, cv=10)
#print("Cross Validation Score:\n",scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(svm))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()