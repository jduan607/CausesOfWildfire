from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import sklearn.metrics
import collections
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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
    sampled_X = np.random.choice(len(i), size = 20000, replace = False)
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

def find_best_lambda(X_tv, y_tv):

  lambdas = np.ones(X_tv.shape[1])
  old_lambdas = [1.e+10,1.e+00 ,1.e+00 ,1.e-05 ,1.e+00 ,1.e+00, 1.e+15, 1.e+00, 1.e+00 ,1.e+00, 1.e+00 ,
           1.e+00 ,0.e+00 ,0.e+00 ,0.e+00 ,1.e+00 ,1.e+00 ,1.e+00 ,1.e+00 ,0.e+00, 0.e+00 ,0.e+00 ,
           0.e+00 ,1.e+00, 1.e+00, 1.e+00 ,1.e+00, 0.e+00] 
  k = 20

  #loop over to find the best lambda
  for i in range(len(lambdas)):
      #best records
      best_value = old_lambdas[i]
      best_acc = None
      #iterate through all test_value as weight
      for power in range(-4,5):

          if old_lambdas[i] == 0:
            test_value = 10**(power - 25)
          else:
            test_value = 10**power * old_lambdas[i]

          lambdas[i] = test_value
          print(f'Finding Risk at lambda:{lambdas}')
          
          
          X_current = X_tv.copy()
          X_current[:, i] = X_current[:, i] * test_value

          if i != 0:
            for j in range(i):
              X_current[:, j] = X_current[:, j] * lambdas[j]


          Knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance', p = 2, n_jobs= -1)
          cv_results = cross_validate(Knn, X_current, y_tv, cv=5)
          score = np.mean(cv_results['test_score'])
          

          if best_acc == None or score > best_acc :
              best_value = test_value
              best_acc = score
              print('Best acc', best_acc)

              
      lambdas[i] = best_value
      
  print('Best Lambda:', lambdas,'                 ')


  return lambdas

def find_best_k(start, end, X_tv, y_tv):
  scores = []
  for i in range(start,end):
    Knn = KNeighborsClassifier(n_neighbors = i, p = 2, n_jobs= -1)

    model_X_train = X_tv
    cv_results = cross_validate(Knn, model_X_train, y_tv, cv=5)

    scores.append(np.mean(cv_results['test_score']))

  plt.plot(range(start,end), scores)
  plt.xlabel('k')
  plt.ylabel('Validation Accuracy')
  plt.title(str(len(np.unique(y_tv))) + '-Class k-Accuracy for KNN')
  plt.show()

  return scores


#### analysis
X,y = load_data()
X_sampled, y_sampled = class4(X,y)

#this value was calculated by find_best_lambda() and saved it here
lambdas = [1.e+07, 1.e+00, 1.e+00, 1.e-01, 1.e+02 ,1.e+02, 1.e+11 ,1.e+00, 1.e-01, 1.e-04,1.e-01,
           1.e-04, 1.e-29, 1.e-29, 1.e-29, 1.e+00, 1.e+00, 1.e+00, 1.e+00, 1.e-29,1.e-29, 1.e-29,
           1.e-29, 1.e+00, 1.e+00, 1.e-01, 1.e+00, 1.e-29]   
scaled_X = X * lambdas
scaled_X_sampled = X_sampled * lambdas

#raw data
X_tv, X_test, y_tv, y_test = split_data(X, y, method = 'random')
#scaled raw data
sX_tv, sX_test, sy_tv, sy_test = split_data(scaled_X, y, method = 'random')
#4-class data
X_tv4, X_test4, y_tv4, y_test4 = split_data(X_sampled, y_sampled, method = 'random')
#scaled 4-class data
sX_tv4, sX_test4, sy_tv4, sy_test4 = split_data(scaled_X_sampled, y_sampled, method = 'random')

scores = find_best_k(1, 30, X_tv, y_tv)
scores = find_best_k(1, 30, X_tv4, y_tv4)

plt.bar(range(len(lambdas)), np.log10(lambdas))
plt.xlabel('Feature Index')
plt.ylabel('Log10(w_i)')
plt.title('Log of weights')
plt.show()

Knn = KNeighborsClassifier(n_neighbors = 20, p = 2, n_jobs= -1)
print_result(Knn, X_tv, y_tv, X_test, y_test, 'KNN')

Knn = KNeighborsClassifier(n_neighbors = 20, p = 2, n_jobs= -1)
print_result(Knn, sX_tv, sy_tv, sX_test, sy_test, 'KNN')

Knn = KNeighborsClassifier(n_neighbors = 4, p = 2, n_jobs= -1)
print_result(Knn, X_tv4, y_tv4, X_test4, y_test4, 'KNN')

Knn = KNeighborsClassifier(n_neighbors = 20, p = 2, n_jobs= -1)
print_result(Knn, sX_tv4, sy_tv4, sX_test4, sy_test4, 'KNN')