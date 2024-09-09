from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import sklearn.metrics
import collections

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

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


def print_result(model, X_tv, y_tv, X_test, y_test, name):
  model.fit(X_tv, y_tv)

  y_tv_pred = model.predict(X_tv)
  print('Train Report')
  print(sklearn.metrics.classification_report(y_tv, y_tv_pred))
  #print(sklearn.metrics.confusion_matrix(y_tv, y_train_pred))
  print(model.score(X_tv, y_tv))

  y_test_pred = model.predict(X_test)
  print('Test Report')
  print(sklearn.metrics.classification_report(y_test, y_test_pred))
  print(sklearn.metrics.confusion_matrix(y_test, y_test_pred))
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
X_normal = normalize_X(X)

counter = collections.Counter(y)
height = [counter[i] if i in counter else 0 for i in range(12)]
#height was manually deleted with an empty column
height = [61155, 32670, 9849, 20399, 110874, 3318, 48926, 10086, 3893, 8714, 1710]

plt.figure(figsize=(10,2))
plt.bar(range(11), height)
plt.title('11-Class Label Distribution')
plt.xlabel("Feature Index")
plt.ylabel('Sample counts')
plt.show()

#### resample X with only 4 class

#class index
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
  sampled_X = np.random.choice(len(i), size = 20000, replace = False)
  ys = np.concatenate((ys, np.ones(20000)* cnt))
  sampled_Xs = np.concatenate((sampled_Xs,i[sampled_X]))

sampled_ys = ys.astype(int)

#4class
X_tv4, X_test4, y_tv4, y_test4 = split_data(sampled_Xs, sampled_ys, method = 'random')
#11class
X_tv, X_test, y_tv, y_test = split_data(X, y, method = 'random')

## Find the best depth from cross validation
scores = []
for i in range(1,20):
  #print(i)
  DecisionTree = DecisionTreeClassifier(max_depth= i)
  cv_results = cross_validate(DecisionTree, X_tv, y_tv, cv=5)
  #print(np.mean(cv_results['test_score']))
  scores.append(np.mean(cv_results['test_score']))

plt.plot(range(1,20), scores)
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')

plt.title('Decision Tree Depth-Accuracy')
plt.show()

## Find the best depth from cross validation
scores = []
for i in range(1,40,2):
  #print(i)
  DecisionTree = DecisionTreeClassifier(max_depth= i)
  cv_results = cross_validate(DecisionTree, X_tv4, y_tv4, cv=5)
  #print(np.mean(cv_results['test_score']))
  scores.append(np.mean(cv_results['test_score']))

plt.plot(range(1,40,2), scores)
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')

plt.title('Decision Tree Depth-Accuracy')
plt.show()

## Best training accuracy from cross-validation
print('11-class: ')
DecisionTree = DecisionTreeClassifier(max_depth= 13)
cv_results = cross_validate(DecisionTree, X_tv, y_tv, cv=5)
print(np.mean(cv_results['test_score']))

print('4-class: ')
DecisionTree = DecisionTreeClassifier(max_depth = 9)
cv_results = cross_validate(DecisionTree, X_tv4, y_tv4, cv=5)
print(np.mean(cv_results['test_score']))

## Print result
DecisionTree = DecisionTreeClassifier(max_depth=13)
print_result(DecisionTree, X_tv, y_tv, X_test, y_test, 'DecisionTree')

DecisionTree = DecisionTreeClassifier(max_depth=9)
print_result(DecisionTree, X_tv4, y_tv4, X_test4, y_test4, 'DecisionTree')

from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver='full')
X_new = pca.fit_transform(X_normal)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
plt.figure(figsize=(10,10))
plt.scatter(X_new[:,0], X_new[:,1], s = 0.1, color=colors[y.tolist()])

plt.show()