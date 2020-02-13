import pandas as pd
import Tree
import math
import numpy as np
pd.options.mode.chained_assignment = None

class C45:
  def __init__(self, gain_ratio=False):
    self.gain_ratio = gain_ratio
    super().__init__()
  
  def _compute_entropy(self, target_values):
    unique, counts = np.unique(target_values, return_counts=True)
    y_len = len(target_values)
    
    ETemp = 0.0
    e = lambda a : (a/y_len*math.log2(a/y_len))
    for c in counts:
      ETemp += e(c)

    return ETemp*(-1)

  def _compute_attr_entropy(self, training_samples, target_values, attr):
    Ea = 0.0
    ts_len = len(training_samples)
    
    for val in training_samples[attr].value_counts().reset_index().values:
      val_name = val[0]; val_count = val[1]
      filtered_idx = training_samples.loc[training_samples[attr] == val_name].index
      Ea += (val_count/ts_len)*self._compute_entropy(target_values[filtered_idx])

    return Ea
  
  def _compute_split_information(self, training_samples, target_values, attr):
    result = 0.0
    ts_len = len(training_samples)

    for val in training_samples[attr].value_counts().reset_index().values:
      p = val[1]/ts_len
      result += p * math.log2(p)
    
    return result * (-1)

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
      best_gain, best_attr_by_gain = 0.0, ''
      best_gain_ratio, best_attr_by_gain_ratio = 0.0, ''
      Es = self._compute_entropy(target_values)

      for attr in training_samples.columns:
        Ea = self._compute_attr_entropy(training_samples, target_values, attr)
        gain = Es - Ea
        split_information = self._compute_split_information(training_samples, target_values, attr)
        gainratio = (gain / split_information) if (split_information > 0) else 0
        if gain > best_gain:
          best_attr_by_gain = attr
          best_gain = gain
        if gainratio > best_gain_ratio:
          best_gain_ratio = gainratio
          best_attr_by_gain_ratio = attr
      
      if (self.gain_ratio):
        best_attr = best_attr_by_gain_ratio
      else:
        best_attr = best_attr_by_gain

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

  def _get_potential_breakpoints(self, sorted_col, sorted_target):
    sorted_target = list(sorted_target)
    idx_list = []
    breakpoints_list = []
    last_label = sorted_target[0]
    for i in range(1, len(sorted_target)):
      if sorted_target[i] != last_label:
        idx_list.append(i-1)
        breakpoints_list.append((sorted_col[i]+sorted_col[i-1])*0.5)
        last_label = sorted_target[i]
    
    return idx_list, breakpoints_list
  
  def _compute_entropy_continuous(self, feature, target, idx_cut):
    sample_len = len(target)
    left_proportion = (idx_cut + 1) / sample_len
    right_proportion = (sample_len - (idx_cut + 1)) / sample_len
    
    eleft = left_proportion*self._compute_entropy(target[:idx_cut+1]) 
    eright = right_proportion*self._compute_entropy(target[idx_cut+1:])
    return eleft + eright

  def _unique(self, ar):
    unique_list = list(set(ar))
    counts = []
    for u in unique_list:
      counts.append(ar.count(u))
    
    return unique_list, counts
  
  def _get_common_target_values(self, training_samples, target_values):
    retval = {}
    klass, counts = np.unique(target_values, return_counts=True)
    for k in klass:
      filtered_klass_idx = target_values.loc[target_values == k].index
      
      for col in training_samples.columns:
        filtered_samples = (training_samples[col])[filtered_klass_idx]
        attr_val, count_val = self._unique(list(filtered_samples))

        if col not in retval:
          retval[col] = {}
        retval[col][k] = attr_val[np.argmax(count_val)]

    return retval

  def fit(self, X, y):
    # handle continuous values
    continu_features = X.select_dtypes(include='number')
    for col in continu_features.columns:
      sorted_feature = continu_features[col].sort_values()
      sorted_target = y[sorted_feature.index]
      
      potential_idx_list, breakpoints_list = self._get_potential_breakpoints(sorted_feature, sorted_target)
      Es_con = self._compute_entropy(sorted_target)

      best_breakpoint = 0
      best_gain = 0.0
      for i in range(len(potential_idx_list)):
        idx_cut = potential_idx_list[i]
        gain = Es_con - self._compute_entropy_continuous(sorted_feature, sorted_target, idx_cut)
        if gain > best_gain:
          best_gain = gain
          best_breakpoint = breakpoints_list[i]

      smaller, greater = '<= ' + str(best_breakpoint), '> ' + str(best_breakpoint)
      X.loc[:, col] = X[col].apply(lambda x: smaller if x <= best_breakpoint else greater)
    
    # handle missing values
    common_target_values = self._get_common_target_values(X, y)
    cols_list = [col for col in X.columns]
    for idx, row in X.iterrows():
      for c in cols_list:
        if pd.isna(row[c]):
          X.loc[idx, c] = common_target_values[c][y[idx]]
    
    self._tree = self._construct_tree(X, y)
    self._tree.printTree()

  def _predict(self, X):
    ret = []
    err = False
    for i, row in X.iterrows():
      node = self._tree
      depth = 0
      while (len(node.children) > 0): # node still inner node
        if node.checkChildrenValueExist(row[node.value]) or node.checkChildrenValueSatisfy(row[node.value]):
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
    

# df = pd.read_csv('play-tennis.csv')
df = pd.read_csv('sabtusore-missing.csv')
headers = list(df.columns[1:])
feature_names, target_names = headers[:-1], headers[-1]

clf = C45()
clf.fit(df[feature_names], df[target_names])
