import numpy as np
from functools import partial
import scipy.optimize


###
### Robust eigs for covariance
###
def do_one_ls(X, min_std_to_count_in_p=-1):
  # X is n by p
  n, p = X.shape
  meanX = np.mean(X, axis=0)

  if min_std_to_count_in_p >= 0:
    stdX = np.std(X, axis=0)
    effp = (stdX > min_std_to_count_in_p).sum()
  else:
    effp = p

  Xdm = X - meanX[np.newaxis,:]
  E = Xdm.dot(Xdm.T)  # n by n
  m = np.sum(np.diag(E))/((n-1) * effp)
  tr_St_S = np.sum(np.square(E)) / np.square(n-1)
  d2 = tr_St_S / effp - np.square(m)
  b_bar2 = (np.sum(np.square(np.diag(E))) - (n-2) * tr_St_S) / ((n-1) * (n-1) * effp)
  b2 = min(d2, b_bar2)
  a2 = d2 - b2
  
  return b2/d2 * m, a2/d2 / (n-1)

###
### Change of basis
###
def reduce_basis(X):
  q, r = np.linalg.qr(X.T, mode='reduced')
  mod_X, overall_Vt = r.T, q.T
  return mod_X, overall_Vt


def svd_lo(mod_X, y, lo_class):
  X_lo = mod_X[y==lo_class]
  mu_lo = np.mean(X_lo, axis=0)
  X_lo2 = X_lo - mu_lo[np.newaxis, :]
  _, _, Vt = np.linalg.svd(X_lo2, full_matrices=False)
  return mu_lo, Vt


###
### Optimization of Eq 3
###
def myPhi(x):
  return 0.5 * (1 + scipy.special.erf(x/np.sqrt(2)))
def myphi(x):
  return 1.0 / np.sqrt(2*np.pi) * np.exp(-x*x/2)

def obj_cbeta(x, posns_lo, posns_hi, q, sprime_mult, lo_class, ret_jac=False, ret_both_classes=False):
  c, beta = x[0], x[1:]
  weights_lo = 1 / posns_lo.shape[0]
  weights_hi = 1 / posns_hi.shape[0]

  objs = np.zeros(2)
  jacs = np.zeros((2, len(x)))

  # hi class
  s = 1 + (lo_class) * (c + posns_hi.dot(beta))
  mP = myPhi(s/1e-7)
  tmp = np.maximum(s, 0)
  
  if not ret_jac:
    objs[0] = np.sum(weights_hi * tmp)
  else:
    jacs[0,0] = lo_class * np.sum(weights_hi * mP)
    jacs[0,1:] = lo_class * (weights_hi*mP).T.dot(posns_hi)

  # lo class
  s = 1 - (lo_class) * (c + posns_lo.dot(beta))

  
  Sigma_beta = sprime_mult * (posns_lo.T).dot(posns_lo.dot(beta)) + q * beta

  t = np.sqrt(beta.dot(Sigma_beta))
  mp = myphi(s/(t+1e-7))
  mP = myPhi(s/(t+1e-7))

  if not ret_jac:
    objs[1] = np.sum(weights_lo * np.maximum(s * mP + t * mp, 0))
  else:
    jacs[1,0] = (-lo_class) * np.sum(weights_lo * mP)
    jacs[1,1:] = (-lo_class) * (weights_lo*mP).T.dot(posns_lo) + 1/(t+1e-3) * np.sum(weights_lo * mp) * Sigma_beta

  if not ret_both_classes:
    return np.mean(objs) if not ret_jac else np.mean(jacs, axis=0)
  else:
    return objs if not ret_jac else jacs

def solve_Eq3(posns_lo, posns_hi, q, sprime_mult, lo_class, do_nelder_mead=True):
  func_cbeta = partial(obj_cbeta,
                       posns_lo=posns_lo,
                       posns_hi=posns_hi,
                       q=q,
                       sprime_mult=sprime_mult,
                       lo_class=lo_class)
  use_obj = partial(func_cbeta, ret_jac=False)
  use_jac = partial(func_cbeta, ret_jac=True)

  init_beta = (np.mean(posns_lo, axis=0) - np.mean(posns_hi, axis=0)) * lo_class
  init_beta /= np.linalg.norm(init_beta)
  init_c = (np.mean(posns_lo, axis=0) - np.mean(posns_hi, axis=0)).dot(init_beta) / 2
  x0 = np.hstack((init_c, init_beta))

  res = scipy.optimize.minimize(use_obj, x0=x0, jac=use_jac, method='L-BFGS-B')

  if not res.success and do_nelder_mead:
    res = scipy.optimize.minimize(use_obj, x0=res.x, method='Nelder-Mead', options={'maxiter':5000})

  c_Eq3 = res.x[0]
  beta_Eq3 = res.x[1:]
  return c_Eq3, beta_Eq3


###
### Optimization of Eq 4
###

def choose_c4(posns_lo_1d, posns_hi_1d, Sigma_lo_1d, lo_class, c_guess):
  weights_lo = 1 / len(posns_lo_1d)
  weights_hi = 1 / len(posns_hi_1d)

  grid = np.linspace(posns_hi_1d.mean(), -c_guess, 10)
  hi_err = np.mean(np.maximum(lo_class * (posns_hi_1d[np.newaxis,:] - grid[:,np.newaxis]), 0), axis=1)
  lo_err = np.mean(myPhi(-lo_class * (posns_lo_1d[np.newaxis,:] - grid[:,np.newaxis]) / np.sqrt(Sigma_lo_1d)), axis=1)

  score = np.abs(hi_err - lo_err)
  best_grid_idx = np.argmin(score)
  return -grid[best_grid_idx]

def solve_Eq4(posns_lo, posns_hi, q, sprime_mult, lo_class, c_Eq3, beta_Eq3):
  posns_lo_1d = posns_lo.dot(beta_Eq3)
  posns_hi_1d = posns_hi.dot(beta_Eq3)
  Sigma_lo_1d = sprime_mult * np.sum(np.square(posns_lo.dot(beta_Eq3))) + q * beta_Eq3.dot(beta_Eq3)
  c_Eq4 = choose_c4(posns_lo_1d, posns_hi_1d, Sigma_lo_1d, lo_class, c_Eq3)
  return c_Eq4

###
### Overall
###

  
def direct_param_fitting(X, y, lo_class, q, sprime_mult, overall_Vt, use_K):
  # We assume that QR decomposition has been done if necessary.
  # overall_Vt gives us the mapping back into the original feature space.
  # q and sprime needed to be computed on the original X, before any change of basis.

  mu_lo = np.mean(X[y==lo_class], axis=0)
  X2 = X - mu_lo[np.newaxis, :] # shift the origin to mu_lo (in the basis of overall_Vt)

  if not use_K:
    posns_lo = X2[y==lo_class]
    posns_hi = X2[y!=lo_class]
  else:
    K = X2.dot(X2.T)
    posns_lo = K[y==lo_class]
    posns_hi = K[y!=lo_class]

  c_Eq3, beta_Eq3 = solve_Eq3(posns_lo=posns_lo, posns_hi=posns_hi, q=q, sprime_mult=sprime_mult, lo_class=lo_class)
  c_Eq4 = solve_Eq4(posns_lo=posns_lo, posns_hi=posns_hi, q=q, sprime_mult=sprime_mult, lo_class=lo_class, c_Eq3=c_Eq3, beta_Eq3=beta_Eq3)
  
  if not use_K:
    c = c_Eq4 - beta_Eq3.dot(mu_lo)
    if overall_Vt is not None:
      beta = np.sum(overall_Vt * beta_Eq3[:, np.newaxis], axis=0)
    else:
      beta = beta_Eq3
  else:
    beta = X2.T.dot(beta_Eq3)
    c = c_Eq4 - beta.dot(mu_lo)

  return c, beta

def direct(X, y, use_K=False):
  lo_class = -1 if (y==1).sum() > len(y) / 2 else 1
  n, p = X.shape

  q, sprime_mult = do_one_ls(X[y==lo_class])
  mod_X, overall_Vt = reduce_basis(X) if n<p and not use_K else (X, None)
  c, beta = direct_param_fitting(X=mod_X, y=y, lo_class=lo_class, q=q, sprime_mult=sprime_mult, overall_Vt=overall_Vt, use_K=use_K)

  return c, beta

