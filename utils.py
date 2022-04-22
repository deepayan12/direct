import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import time

def all_metrics(X, y, scores, scores01):
  y01 = (y>0).astype(int)
  if scores is None:
    scores = np.random.uniform(size=len(y)) 
  if scores01 is None:
    scores01 = np.ones_like(y) * (1 if (y01.sum() > 0.5 * len(y01)) else 0)

  precision, recall, _ = precision_recall_curve(y01, scores)

  result = {
            'auc':roc_auc_score(y01, scores),
            'aucpr':auc(recall, precision),
           } 
  return result

def all_metrics_SVM(X, y, c, beta):
  start_time = time.time()
  scores = c + np.sum(X * beta[np.newaxis, :], axis=1)
  scores01 = (scores>0).astype(int)
  test_time = time.time() - start_time
  return all_metrics(X, y, scores, scores01), test_time

