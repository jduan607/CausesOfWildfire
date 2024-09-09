from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import sklearn.metrics
import collections

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

def create_one_hot(y):
  Y = []
  n = max(y) + 1
  print('Total col: ', n)
  for i in y:
      arr = np.zeros(n)
      arr[i] = 1
      Y.append(arr)

  return np.asarray(Y)

def class4(X,y):
  choosed = [0,1,4,6]
  np.random.seed(0)
  Xs = []
  ys = []
  for i in choosed:
    index = [y == i]
    Xs.append(X[index])
    ys.append(y[index])
    #print(X[index].shape)


  sampled_Xs = np.zeros((0,28))
  ys = np.zeros(0)
  for cnt,i in enumerate(Xs):
    sampled_X = np.random.choice(len(i), size = 20000)
    ys = np.concatenate((ys, np.ones(20000)* cnt))
    sampled_Xs = np.concatenate((sampled_Xs,i[sampled_X]))

  sampled_ys = ys.astype(int)

  return sampled_Xs, sampled_ys


def Multi_logistic(X, Y, learning_rate = 1e-9, max_iteration = 100):

  #records
  losses = []
  risks = []
  risk_2_arr = []

  #size of the sample
  size = X.shape[0]
  
  #init
  n = Y.shape[1]
  ws = []
  for i in range(n):
    ws.append(np.random.randn(X.shape[1]) * 1e-10)
    #ws.append(np.zeros(X.shape[1]))
  ws = np.array(ws)
  #print(ws.shape)

  
  #convert 1-hot back to regular
  y = np.argmax(Y, axis = 1)
  for it in range(max_iteration):
    print('Iteration ', it)
    #pred
    Y_pred = np.exp(X @ ws.T)

    #training
    new_ws = []
    for c in range(n):
      softmax = np.array([Y_pred[i, c] / sum(Y_pred[i, :]) for i in range(len(Y_pred))])
      diff = Y[:, c] - softmax
      d_w = sum([ X[i] * diff[i] for i in range(len(X))])
      
      #step
      ws[c] += learning_rate * d_w

    #convert 1-hot back to TOP-1
    label_pred = np.argmax(Y_pred, axis = 1)
    print(label_pred.shape)

    #risk for TOP-1
    risks.append(sum(label_pred != y) / size)
    print('risk = ', risks[-1])

    #loss with 1-hot
    losses.append(np.linalg.norm(Y_pred - Y))
    print('Loss = ', losses[-1])

    #TOP-2 prediction
    label_n_pred = np.argsort(Y_pred, axis = 1)
    correct_2 = [y[i] in label_n_pred[i, -2:] for i in range(len(label_n_pred))]
    risk_2 = 1 - sum(correct_2) / size
    risk_2_arr.append(risk_2)
    print('risk_2 = ', risk_2_arr[-1])


    if it % 10 == 0:
      print(it)


  
  plt.plot(risks, label = 'Top-1 prediction')
  plt.plot(risk_2_arr, label = 'Top-2 prediction ')
  plt.xlabel('Iteration')
  plt.ylabel('Risk')
  plt.legend()
  plt.title(str(len(np.unique(y))) + '-class Training Risk')
  plt.show()
  return ws,losses, risks, risk_2_arr


def evaluation(X, Y_tv, theta):
  y = np.argmax(Y_tv, axis = 1)
  Y_pred = np.exp(X @ theta.T)
  label_pred = np.argmax(Y_pred, axis = 1)
  risks = sum(label_pred != y) / X.shape[0]

  #confusion matrix plot
  import seaborn as sns
  train_cfm = sklearn.metrics.confusion_matrix(y, label_pred)
  #categories = [i[3:] for i in onehot_encoder.get_feature_names()]
  plt.figure(figsize = (7,4))
  sns.heatmap(train_cfm/np.sum(train_cfm), annot=True, 
              fmt='.1%', cmap='Blues')
  plt.title("MLR " + str(len(np.unique(y))) + '-class Testing Result')
  plt.show()

  label_n_pred = np.argsort(Y_pred, axis = 1)
  correct_2 = [y[i] in label_n_pred[i, -2:] for i in range(len(label_n_pred))]
  risk_2 = 1 - sum(correct_2) / X.shape[0]

  return 1-risks, 1-risk_2

def plot_ws(ws, class_name):
  plt.bar(range(28), np.linalg.norm(ws, axis = 0, ord = 1))
  plt.xlabel('Feature Index')
  plt.ylabel('sum(|theta_k|)')
  plt.title(class_name + ' Feature importance')
  plt.show()
  return

#### analysis
X,y = load_data()
Y = create_one_hot(y)

X_sampled, y_sampled = class4(X,y)
Y_sampled = create_one_hot(y_sampled)

X_tv, X_test, Y_tv, Y_test = split_data(normalize_X(X), Y, method = 'random')
X_tv4, X_test4, Y_tv4, Y_test4 = split_data(normalize_X(X_sampled), Y_sampled, method = 'random')

#11class
ws,losses, risks, risk_2_arr = Multi_logistic(X_tv, Y_tv, max_iteration=10)
plot_ws(ws, '11-class')

print('Training Result')
acc1,acc2 = evaluation(X_tv, Y_tv,ws)
print('TOP-1 Accuracy: ', acc1)
print('TOP-2 Accuracy: ', acc2)

print()
print('Test Result')
acc1,acc2 = evaluation(X_test, Y_test,ws)
print('TOP-1 Accuracy: ', acc1)
print('TOP-2 Accuracy: ', acc2)

#4class
ws,losses, risks, risk_2_arr = Multi_logistic(X_tv4, Y_tv4, max_iteration=10)
plot_ws(ws, '4-class')

print('Training Result')
acc1,acc2 = evaluation(X_tv4, Y_tv4,ws)
print('TOP-1 Accuracy: ', acc1)
print('TOP-2 Accuracy: ', acc2)

print()
print('Test Result')
acc1,acc2 = evaluation(X_test4, Y_test4,ws)
print('TOP-1 Accuracy: ', acc1)
print('TOP-2 Accuracy: ', acc2)