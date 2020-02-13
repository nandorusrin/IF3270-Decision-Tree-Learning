import math
import pandas as pd
import numpy as np
import Tree

class Id3:
  def __init__(self):
    self._accuracy = 0.0
    super().__init__()

  def get_accuracy(self):
    return self._accuracy

  '''
  I.S	: List of label (target_values)
  F.S	: floating point ENTROPY dari data yang dihitung
  '''
  def _compute_entropy(self, target_values):
    unique, counts = np.unique(target_values, return_counts=True)
    y_len = len(target_values)
    
    result = 0.0
    e = lambda a : (a/y_len*math.log2(a/y_len))
    for c in counts:
      result += e(c)

    return result*(-1)
  
  def _compute_attr_entropy(self, training_samples, target_values, attr):
    Ea = 0.0
    ts_len = len(training_samples)
    
    for val in training_samples[attr].value_counts().reset_index().values:
      val_name = val[0]; val_count = val[1]
      filtered_idx = training_samples.loc[training_samples[attr] == val_name].index
      Ea += (val_count/ts_len)*self._compute_entropy(target_values[filtered_idx])

    return Ea
  
  def _construct_tree(self, training_samples, target_values):
    root = self._build_tree_rec(training_samples, target_values)
    return root
  
  def _build_tree_rec(self, training_samples, target_values):
    # check if it's should become leaf node
    if target_values.nunique() == 1:
      label = target_values.iloc[0]
      target_count = [(label, len(target_values))]
      
      created_node = Tree.Tree(label, target_count)
      created_node.setEntropy = self._compute_entropy(target_values)
      return created_node
    else:
      best_gain, best_attr = 0.0, ''
      Es = self._compute_entropy(target_values)

      for attr in training_samples.columns:
        Ea = self._compute_attr_entropy(training_samples, target_values, attr)
        gain = Es - Ea
        if gain > best_gain:
          best_attr = attr
          best_gain = gain
      
      target_count = []
      val_names = []
      for val in training_samples[best_attr].value_counts().reset_index().values:
        val_name = val[0]; val_count = val[1]
        val_names.append(val_name)
        target_count.append((val_name, val_count))

      created_node = Tree.Tree(best_attr, target_count)
      created_node.setEntropy(Ea)

      for val_name in val_names:
        filtered_training_samples = training_samples.loc[training_samples[best_attr] == val_name]
        filtered_target_values = target_values[filtered_training_samples.index]

        created_node.addChildren(val_name, self._build_tree_rec(filtered_training_samples, filtered_target_values))
      
      return created_node

  def _train_test_split(self, X, y, test_size=0.2):
    sample_len = len(y)
    cut_idx = int(sample_len*test_size)
    X_test = pd.DataFrame(X.columns)
    X_test = X.iloc[:cut_idx, :]
    y_test = y.iloc[:cut_idx]
    X = X.iloc[cut_idx:, :]
    y = y.iloc[cut_idx:]
    
    return X, X_test, y, y_test

  def fit(self, X, y):
    X, X_test, y, y_test = self._train_test_split(X, y)

    self._tree = self._construct_tree(X, y)
    self._tree.printTree()
    self._accuracy = self._compute_accuracy(X_test, y_test)

    return self

  def _predict(self, X):
    ret = []
    err = False
    for i, row in X.iterrows():
      node = self._tree
      depth = 0
      while (len(node.children) > 0): # node still inner node
        if node.checkChildrenValueExist(row[node.value]):
          node = node.gotoSpesificChildren(row[node.value])
        else:
          node = node.gotoMaxChildrenCount()

        if node == None:
          err = True
          break
        depth += 1
      if err:
        break
      
      ret.append(node.value)
    return ret

  def predict(self, X):
    return self._predict(X)
  
  def _compute_accuracy(self, validation_sample, validation_target):
    predictions = self._predict(validation_sample)

    count_true = 0
    for i in range(len(predictions)):
      if predictions[i] == validation_target.iloc[i]:
        count_true += 1
    
    return count_true / len(validation_target)

dataTrain = pd.read_csv('play-tennis.csv')
headers = list(dataTrain.columns[1:])
feature_names, target_names = headers[:-1], headers[-1]

clf = Id3()
clf = clf.fit(dataTrain[feature_names], dataTrain[target_names])
print('akurasi:', clf.get_accuracy())

newData = pd.read_csv('play-tennis-predict.csv')
headers_new = list(newData.columns[1:])
feature_names_new, target_names_new = headers_new[:-1], headers_new[-1]

print(clf.predict(newData[feature_names_new]))
