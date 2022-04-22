import argparse
import numpy as np
import time
import imbalance

def parse_args():
  parser = argparse.ArgumentParser(description='Robust Imbalanced Classification')
  parser.add_argument('--dataset', type=str, default='Tumors_scaled.csv.gz')
  parser.add_argument('--nlo', nargs='+', type=int, default=[5], help='Number of datapoints from minority class')
  parser.add_argument('--nhi', nargs='+', type=int, default=[100], help='Number of datapoints from majority class')
  parser.add_argument('--seed', type=int, default=0, help='Random number seed for repeatability')
  parser.add_argument('--num_procs', type=int, default=30, help='Number of processors')
  parser.add_argument('--num_repeats', type=int, default=30, help='Number of repetitions of experiment')
  parser.add_argument('--use_K', dest='use_K', default=False, action='store_true', help='If set, computation is typically faster (but results may occasionally differ)')
  parser.add_argument('--max_classes', type=int, default=-1, help='Number of classes to try in dataset (-1 for all classes)')

  args = parser.parse_args()
  print(args)
  return args


def main():
  args = parse_args()
  assert len(args.nlo) == len(args.nhi)
  for i in range(len(args.nlo)):
    n_lo = args.nlo[i]
    n_hi = args.nhi[i]

    start_time = time.time()
    imbalance.do_all_multiclass(dataset=args.dataset,
                                n_lo=n_lo, 
                                n_hi=n_hi,
                                max_classes=args.max_classes, 
                                num_repeats=args.num_repeats,
                                seed=args.seed, 
                                use_K=args.use_K,
                                num_procs=args.num_procs,
                                )
    print('time taken (in secs) = {:2.0f}'.format(time.time() - start_time))
    print()



if __name__ == '__main__':
  main()
