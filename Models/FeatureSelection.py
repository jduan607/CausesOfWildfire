from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import sklearn.metrics
import collections

# ANOVA feature selection for numeric/categorical input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

def load_data():
  X = np.load('drive/Shareddrives/CS760/ProcessedData/X_processed_numpy_10class.npz')['arr_0']
  y = np.load('drive/Shareddrives/CS760/ProcessedData/y_processed_numpy_10class.npz')['arr_0']

  print('X shape: ', X.shape)
  print('y shape: ', y.shape)

  return X,y


def split_data(X, y, method = 'random'):
  if method == 'random':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

  elif method == 'year':
    X_train = X[X[:,0] <= 2014][:,1:]
    y_train = y_2[X[:,0] <= 2014]

    #X_val = X[X[:,0] == 2014][:,1:]
    #y_val = y_2[X[:,0] == 2014]

    X_test = X[X[:,0] > 2014][:,1:]
    y_test = y_2[X[:,0] > 2014]
  else:
    print('Error: method not found!')
    return None

  print('X train shape: ', X_train.shape)
  print('y train shape: ',y_train.shape)
  print('X test shape: ',X_test.shape)
  print('y test shape: ',y_test.shape)

  return X_train, X_test, y_train, y_test


def normalize_X(X):
  X_mean = X.mean(axis = 0)
  X_mean.shape

  X_std = X.std(axis = 0)
  X_std.shape

  X_normal = (X - X_mean) / X_std

  return X_normal


def resample(X, sample_per_label = -1):
  Xs = np.array( [ np.array(X[y_3 == i]) for i in range(3)] )

  if sample_per_label == -1:
    sample_per_label = min([len(i) for i in Xs])

  print('Sample_per_label = ', sample_per_label)

  X_sample = []
  y_sample = []
  for i in range(3):
    X0 = np.random.choice(len(Xs[i]), size= int(sample_per_label), replace = False)
    X0 = Xs[i][X0]
    y0 = np.ones(int(sample_per_label)) * i
    X_sample.append(X0)
    y_sample.append(y0)

  X_sample = np.concatenate((X_sample))
  y_sample = np.concatenate((y_sample))

  print('X_sample shape:', np.shape(X_sample))
  print('y_sample shape:', np.shape(y_sample))

  return X_sample,y_sample

def class4(X,y):
  choosed = [0,1,4,6]
  np.random.seed(0)
  Xs = []
  ys = []
  for i in choosed:
    index = [y == i]
    Xs.append(X[index])
    ys.append(y[index])

  sampled_Xs = np.zeros((0,28))
  ys = np.zeros(0)
  for cnt,i in enumerate(Xs):
    sampled_X = np.random.choice(len(i), size = 20000)
    ys = np.concatenate((ys, np.ones(20000)* cnt))
    sampled_Xs = np.concatenate((sampled_Xs,i[sampled_X]))

  sampled_ys = ys.astype(int)

  return sampled_Xs, sampled_ys

def print_result(model, X_tv, y_tv, X_test, y_test, name):
  model.fit(X_tv, y_tv)
  
  print('Train Report')
  y_tv_pred = model.predict(X_tv)
  print(sklearn.metrics.classification_report(y_tv, y_tv_pred))
  #print(sklearn.metrics.confusion_matrix(y_tv, y_train_pred))
  print(model.score(X_tv, y_tv))

  print('Test Report')
  y_test_pred = model.predict(X_test)
  print(sklearn.metrics.classification_report(y_test, y_test_pred))
  #print(sklearn.metrics.confusion_matrix(y_test, y_test_pred))
  print(model.score(X_test, y_test))


  import seaborn as sns
  train_cfm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)
  #categories = [i[3:] for i in onehot_encoder.get_feature_names()]
  plt.figure(figsize = (7,4))
  sns.heatmap(train_cfm/np.sum(train_cfm), annot=True, 
              fmt='.1%', cmap='Blues')
  plt.title(name+' ' + str(len(np.unique(y_tv))) + '-Class Testing Result')
  plt.show()

#### analysis
X,y = load_data()

# Mutual information (slower)
mi = mutual_info_classif(X,y,discrete_features=[0,1,2,6],n_neighbors=20)
print(mi)
indices_m = np.argsort(mi)[::-1]
print(indices_m)

# F-test (linear dependency)
ftest = f_classif(X,y)
print(ftest)
indices_f = np.argsort(ftest[0])[::-1]
print(indices_f)

X_train, X_test, y_train, y_test = split_data(X, y, method = 'random')

table1 = np.zeros((2,28),dtype=float)

for s in range(1,29,1):
    Xm_train = X_train[:,indices_m[:s]]
    NaiveBayes = GaussianNB()

    cv_results = cross_validate(NaiveBayes, Xm_train, y_train, cv=5)
    table1[0,s-1] = np.mean(cv_results['test_score'])
    #print(np.mean(cv_results['test_score']))

    Xf_train = X_train[:,indices_f[:s]]
    NaiveBayes = GaussianNB()

    cv_results = cross_validate(NaiveBayes, Xf_train, y_train, cv=5)
    table1[1,s-1] = np.mean(cv_results['test_score'])
    #print(np.mean(cv_results['test_score']))
   
X4, y4= class4(X,y)  

# Mutual information (slower)
mi4 = mutual_info_classif(X4,y4,discrete_features=[0,1,2,6],n_neighbors=4)
print(mi4)
indices_m4 = np.argsort(mi4)[::-1]
print(indices_m4)

# F-test (linear dependency)
ftest4 = f_classif(X4,y4)
print(ftest4)
indices_f4 = np.argsort(ftest4[0])[::-1]
print(indices_f4)

X4_train, X4_test, y4_train, y4_test = split_data(X4, y4, method = 'random')

table2 = np.zeros((2,28),dtype=float)

for s in range(1,29,1):
    Xm_train = X4_train[:,indices_m4[:s]]
    NaiveBayes = GaussianNB()

    cv_results = cross_validate(NaiveBayes, Xm_train, y4_train, cv=5)
    table2[0,s-1] = np.mean(cv_results['test_score'])
    #print(np.mean(cv_results['test_score']))

    Xf_train = X4_train[:,indices_f4[:s]]
    NaiveBayes = GaussianNB()

    cv_results = cross_validate(NaiveBayes, Xf_train, y4_train, cv=5)
    table2[1,s-1] = np.mean(cv_results['test_score'])
    #print(np.mean(cv_results['test_score']))

plt.plot(range(1,29),table1[0,:])
plt.title("11-Class Mutual Information Accuracy")
plt.xlabel("Number of selected features")
plt.ylabel("Validation Accuracy")
plt.show()

plt.plot(range(1,29),table1[1,:])
plt.title("11-Class ANOVA F-test Accuracy")
plt.xlabel("Number of selected features")
plt.ylabel("Validation Accuracy")
plt.show()

plt.plot(range(1,29),table2[0,:])
plt.title("4-Class Mutual Information Accuracy")
plt.xlabel("Number of selected features")
plt.ylabel("Validation Accuracy")
plt.show()

plt.plot(range(1,29),table2[1,:])
plt.title("4-Class ANOVA F-test Accuracy")
plt.xlabel("Number of selected features")
plt.ylabel("Validation Accuracy")
plt.show()

plt.figure(figsize=(16,4))
plt.plot(range(1,29),indices_f.argsort(),label="11-Class ANOVA")
plt.plot(range(1,29),indices_m.argsort(),label="11-Class Mutual Information")
plt.plot(range(1,29),indices_f4.argsort(),label="4-Class ANOVA")
plt.plot(range(1,29),indices_m4.argsort(),label="4-Class Mutual Information")
plt.xlabel("Feature")
plt.ylabel("Rank of significance")
plt.legend()
plt.show()


#11-class
NaiveBayes = GaussianNB()

Xf_train = X_train[:,indices_f[:2]] 
Xf_test = X_test[:,indices_f[:2]]

print_result(NaiveBayes, Xf_train, y_train, Xf_test, y_test, "Naive Bayes")

#4-class
NaiveBayes = GaussianNB()

Xf4_train = X4_train[:,indices_f4[:13]] 
Xf4_test = X4_test[:,indices_f4[:13]]

print_result(NaiveBayes, Xf4_train, y4_train, Xf4_test, y4_test, "Naive Bayes")