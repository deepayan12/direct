from functools import partial
from multiprocessing import Pool
import numpy as np
import numpy.linalg
from pandas import Series, DataFrame
import pandas as pd
from scipy.sparse import csr_matrix
import time
from utils import all_metrics_SVM, all_metrics
import direct


np.set_printoptions(precision=3, suppress=True)
pd.set_option('precision', 3, 'max_columns', 200, 'display.width', 400)


def do_all_worker_oneIdx(idx, train_list_item, test_indices, Xfull, yfull, seed, use_K):
  if seed is not None:
    np.random.seed(seed + idx)

  Xtrain, ytrain = train_list_item

  Xtest = Xfull[test_indices]
  ytest = yfull[test_indices]
  

  start_time = time.time()
  direct_c, direct_beta = direct.direct(Xtrain, ytrain, use_K=use_K)
  train_time = time.time() - start_time

  metrics, test_time = all_metrics_SVM(Xtest, ytest, direct_c, direct_beta)
  return metrics, train_time, test_time

def do_all_worker_manyIdx_forPool(this_data_setting, do_all_worker_manyIdx_filledParams):
  start_idx, train_list, test_indices_list = this_data_setting
  return do_all_worker_manyIdx_filledParams(start_idx, train_list, test_indices_list)

def do_all_worker_manyIdx(start_idx, train_list, test_indices_list, seed, use_K, verbose=True):
  global Xfull, yfull

  start_time_manyIdx = time.time()
  results = []
  for sub_idx in range(len(train_list)):
    idx = start_idx + sub_idx
    print(idx, end=' ')
    metrics, train_time, test_time = do_all_worker_oneIdx(idx, train_list[sub_idx], test_indices_list[sub_idx], Xfull, yfull, seed, use_K)
    results.append((idx, metrics, train_time, test_time),)
  return results


def create_train_test(dataset, n_lo, n_hi, num_repeats, seed, min_test_samples=1, max_test_samples=-1):
  global Xfull, yfull
  if seed is not None:
    np.random.seed(seed)
  p = Xfull.shape[1]
  lo_class = -1 if (yfull==1).sum() > len(yfull) / 2 else 1
  lo_class_indices = np.where(yfull == lo_class)[0]
  hi_class_indices = np.where(yfull == -lo_class)[0]

  print('{}: n_lo={} n_hi={} p={} lo_class={}, num points in lo={}, num points in hi={}'.format(dataset, n_lo, n_hi, p, lo_class, len(lo_class_indices), len(hi_class_indices)))
  if len(lo_class_indices) < (n_lo+min_test_samples) or len(hi_class_indices) < (n_hi+min_test_samples):
    print('Not enough data for a minimum of {} test samples.'.format(min_test_samples))


  all_train_indices = []
  all_test_indices = []
  for i in range(num_repeats):
    train_indices = np.hstack((np.random.choice(lo_class_indices, n_lo, replace=False), np.random.choice(hi_class_indices, n_hi, replace=False)))
    test_indices = [x for x in np.arange(Xfull.shape[0]) if x not in train_indices]
    if max_test_samples > 0:
      test_indices = np.random.choice(test_indices, size=max_test_samples, replace=False)
    all_train_indices.append(train_indices)
    all_test_indices.append(test_indices)

  return all_train_indices, all_test_indices, lo_class

    
def do_all_multiclass(dataset, n_lo, n_hi, max_classes=1, num_repeats=30, seed=None, use_saved_file=False, max_n_lo=-1, max_n_hi=-1, **kwargs):
  global Xfull, yfull
  
  df = pd.read_csv(dataset, header=None)
  category = df.iloc[:,0].values
  M = df.iloc[:, 1:].values

  uniq_cats = Series(category).unique()

  if max_classes == -1:
    max_classes = len(uniq_cats)

  all_train = []
  all_test = []
  for i in range(max_classes):
    lo_indices = np.where(category==uniq_cats[i])[0]
    hi_indices = np.array([i for i in range(M.shape[0]) if i not in lo_indices])
    if max_n_lo > 0:
      lo_indices = np.random.choice(lo_indices, max_n_lo)

    if max_n_hi > 0:
      hi_indices = np.random.choice(hi_indices, max_n_hi)

    Xfull = np.empty((len(lo_indices) + len(hi_indices), M.shape[1]))
    Xfull[:len(lo_indices)] = M[lo_indices]
    Xfull[len(lo_indices):] = M[hi_indices]
    yfull = -1 * np.ones(Xfull.shape[0])
    yfull[:len(lo_indices)] = 1

    min_test_samples = 1

    l_train_indices, l_test_indices, lo_class = create_train_test(dataset='{}-{}'.format(dataset, i), n_lo=n_lo, n_hi=n_hi, num_repeats=num_repeats, seed=seed, min_test_samples=min_test_samples)

    do_all(np.array(l_train_indices), np.array(l_test_indices), seed=seed, **kwargs)


def do_all(train_indices_list, test_indices_list, seed=None, use_K=False, num_procs=10, set_nls_eigs=False, std_dev_multiplier=0):
  global Xfull, yfull  # These are global to avoid copying them in multiprocessing

  start_time_do_all = time.time()
  train_Xylist = []
  for i in range(len(train_indices_list)):
    train_indices = train_indices_list[i]
    Xtrain = Xfull[train_indices]
    ytrain = yfull[train_indices]
    train_Xylist.append((Xtrain, ytrain))

  verbose=(num_procs==1)
  this_worker_manyIdx = \
      partial(do_all_worker_manyIdx,
              seed=seed,
              use_K=use_K,
              verbose=verbose)

  if num_procs == 1:
    out_results = this_worker_manyIdx(0, train_Xylist, test_indices_list)
  else:
    data_settings = [(i, [train_Xylist[i]], [test_indices_list[i]]) for i in range(len(train_Xylist))]
    this_worker_forPool = partial(do_all_worker_manyIdx_forPool, do_all_worker_manyIdx_filledParams=this_worker_manyIdx)
    with Pool(num_procs) as p:
      out_results = [x[0] for x in p.map(this_worker_forPool, data_settings)]

  all_metrics = [None] * len(train_indices_list)
  all_method_train_times = [None] * len(train_indices_list)
  all_method_test_times = [None] * len(train_indices_list)
  for i in range(len(train_indices_list)):
    this_idx, this_metrics, this_method_train_times, this_method_test_times = out_results[i]
    all_metrics[this_idx] = this_metrics
    all_method_train_times[this_idx] = this_method_train_times
    all_method_test_times[this_idx] = this_method_test_times

  Z = DataFrame(all_metrics)
  Z['train_time'] = all_method_train_times
  Z['test_time'] = all_method_test_times
  print()
  print(Z.describe().loc[['mean', 'std', '50%', 'max']])
