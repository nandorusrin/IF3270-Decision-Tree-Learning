from itertools import combinations, accumulate
import numpy as np
import functools
import math

def entropy_y(y):
  unique, counts = np.unique(y, return_counts=True)
  y_len = len(y)
  
  ETemp = 0.0
  e = lambda a : (a/y_len*math.log2(a/y_len))
  for c in counts:
    ETemp += e(c)

  return ETemp*(-1)

def entropy_cont(X, y, idx_cut):
  example_count = len(y)  # nanti ga pake len(y)
  left_ct, right_ct = idx_cut+1, example_count-idx_cut
  E_sum = (left_ct/example_count)*entropy_y(y[:idx_cut+1]) + (right_ct/example_count)*entropy_y(y[idx_cut+1:])
  
  return E_sum

X = [40, 48, 60, 72, 80, 90]
y = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No']

# get potential breakpoint idxs
potential_idx_list = []
breakpoints = []
last_y = y[0]
for i in range(1, len(y)):
  if y[i] != last_y:
    potential_idx_list.append(i-1)
    breakpoints.append((X[i]+X[i-1])*0.5)
  last_y = y[i]

Es = entropy_y(y)

for idx in potential_idx_list:
  print('breakpoint idx =', idx)
  print('gain =', Es-entropy_cont(X, y, idx))
