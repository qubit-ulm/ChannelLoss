# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# This is a sample command line application that can be used for optimization with fixed r.

import numpy as np
import scipy.linalg as spla
import mosek
import DickeOptimizer
import OptimizerDickeOuterState
import OptimizerDickeOuterMap
# depending on your choice in DickeBasis, either import the SVD variant or the normal one
import OptimizerDickeConvexIteration as OptimizerDickeConvexIteration
import OptimizerErasure
from scipy.special import comb as binomial
from itertools import product

# We need to index to probability and do arithmetics, so don't use floats. Every probability is in
# multiples of this.
pmultiple = 1/100
env = mosek.Env()

class OptimizeNested:
   def __init__(self, outerMap, *args):
      self.convexIteration = OptimizerDickeConvexIteration.Optimizer(env, *args)
      if outerMap:
         self.outerMap = OptimizerDickeOuterMap.Optimizer(env, *args)
      else:
         self.outerMap = None
      self.outerState = OptimizerDickeOuterState.Optimizer(env, *args)

   def optimize(self, pdist):
      succ, choi, rho = self.convexIteration.optimize(pdist, .5, False)
      if not succ:
         return 0, None, None
      # scan the fidelity in small intervals. While this is a lot less efficient than binary search
      # in principle, finding a low-fidelity solution is easy, and getting a solution with a
      # slightly higher fidelity is not too hard in most cases, while finding a high-fidelity
      # solution with no good start point may prove impossible.
      for f in np.arange(.51, 1, .01):
         try:
            succ, choi, rho = self.convexIteration.optimize(pdist, f, True)
            if not succ:
               break
         except:
            break
      # do not trust the rank optimization too much; the true fidelity should come from the
      # nonlinear optimization, which is always accurate.
      if self.outerMap is None:
         return self.outerState.optimize(pdist, rho)
      else:
         # for this optimization, we might not get a result at all in some extreme cases
         # (pdist = 1), most likely. So in this case, we switch back to the other procedure.
         ret = self.outerMap.optimize(pdist, choi)
         if ret[0] == 0:
            return self.outerState.optimize(pdist, rho)
         else:
            # but the optimization over rho is the more accurate one (but potentially scales worse,
            # so it may pay off to first find a good neighborhood).
            return self.outerState.optimize(pdist, ret[2])

   def reoptimize(self, pdist, initialRhoVec):
      return self.outerState.optimize(pdist, initialRhoVec)

class OptimizeErasure:
   def __init__(self, *args):
      self.erasure = OptimizerErasure.Optimizer(env, *args)

   def optimize(self, pdist, initialDickeRho=None):
      if initialDickeRho is None:
         init = np.random.rand(2 * (self.erasure.ditness ** self.erasure.send))
      else:
         # Note this codepath is never taken, since we don't use the Dicke results as initial
         # states. Currently, it appears that doing so will lead to not improving the results, while
         # the fully random initial state can give much better results - but is extremely slow.
         # We start with some vector in the Dicke basis, but we actually need the full basis
         initRho = np.empty((2 * binomial(self.erasure.send + self.erasure.ditness -1,
                                          self.erasure.ditness -1, exact=True),) *2)
         initRho[np.triu_indices(initRho.shape[0])] = initialDickeRho
         initialPsi = spla.eigh(initRho, lower=False, check_finite=False,
                                subset_by_index=(initRho.shape[0] -1,) *2)[1][:, -1]
         # now upgrade this to the large state
         init = np.zeros((2, self.erasure.ditness ** self.erasure.send))
         sendDict = DickeOptimizer.mulnomSend(self.erasure.ditness, self.erasure.send,
                                              self.erasure.receive)
         sendKeys = {k: v for k, v in zip(sendDict.keys(), range(len(sendDict)))}
         for i in (0, 1):
            for j, basis in enumerate(product(range(self.erasure.ditness),
                                              repeat=self.erasure.send)):
               key = (i,) + tuple(basis.count(n) for n in range(self.erasure.ditness))
               init[i, j] = initialPsi[sendKeys[key]] / np.sqrt(sendDict[key])
      return self.erasure.optimize(pdist, init.reshape((-1,)))

   def reoptimize(self, pdist, initialPsiVec):
      return self.erasure.optimize(pdist, initialPsiVec)

def initialize(which, *args):
   global opti
   if which == 0:
      opti = OptimizeNested(*args)
   elif which == 1:
      opti = OptimizeErasure(*args)

def runa(p):
   global opti
   return opti.optimize(p)

def runb(p, initialVec):
   global opti
   return opti.reoptimize(p, initialVec)

# command line
if __name__ == '__main__':
   import time
   starttime = time.time()

   import argparse
   import os
   import platform
   import psutil
   # multiprocessing.Pool seems to be broken and sometimes doesn't catch up with adding new tasks
   # in callbacks, so we use futures.
   import concurrent.futures as fut

   parser = argparse.ArgumentParser("Finds the optimal distillation algorithm for the best possible"
                                    " multiplexed state.")
   parser.add_argument("d", help="Ditness", type=int, default=2)
   parser.add_argument("s", help="Send", type=int, default=3)
   parser.add_argument("r", help="Receive", type=int, default=2)
   parser.add_argument("--outermap", help="Performs nonlinear map optimization (may be helpful for "
                       "large send values)", action="store_true")
   parser.add_argument("--erasure", help="Takes into account the full erasure knowledge after findi"
                       "ng the best Dicke solution. Will leave Dicke space and lead to exponential "
                       "data growth. Use only for small parameters.", action="store_true")
   parser.add_argument("--pmin", help="Minimum probability to scan (percent)", type=int, default=1)
   parser.add_argument("--pmax", help="Maximum probability to scan (percent)", type=int,
                       default=100)
   parser.add_argument("--workers", help="Number of worker threads", type=int,
                       default=int(os.environ.get('SLURM_NTASK',
                                                  psutil.cpu_count(logical=False) or 1)))
   args = parser.parse_args()
   assert args.d >= 2
   assert args.s > args.r
   if args.outermap and args.erasure:
      raise Exception("--outermap and --erasure are mutually exclusive")
   if args.erasure:
      initargs = (1, args.d, args.s, args.r)
   else:
      initargs = (0, args.outermap, args.d, args.s, args.r)

   print("Parameters: d = {:d}, s = {:d}, r = {:d}".format(args.d, args.s, args.r))

   try:
      directory = "{:d} {:d} {:d}".format(args.d, args.s, args.r)
      if args.erasure:
         directory += " erasure"
      if not os.path.exists(directory):
         os.mkdir(directory)
      os.chdir(directory)
   except OSError:
      print("Current directory is still " + os.getcwd())

   # callbacks from multiprocessing (all run in main process)
   # First, every scan starts with the same initial data and triggers the normal callback done().
   # Here, we try to use the output data to re-scan the neighboring probabilities (to avoid missing
   # bifurcations). If the result of the re-scan is able to improve the fidelity, we continue
   # re-scanning.
   results = {p: False for p in range(args.pmin, args.pmax +1)}

   def done(p, direction, f, choi, rho):
      global results, active, pmultiple, pool, args
      if f - results[p] > 0:
         pfrac = p * pmultiple
         print("Success for p = {:.2f}: F = {:.8f}".format(pfrac, f))
         newfn = "{:.2f} {:.8f}".format(pfrac, f)
         np.savetxt(newfn + (" psi.dat" if args.erasure else " rho.dat"), rho, delimiter=',')
         np.savetxt(newfn + " choi.dat", choi, delimiter=',')
         if results[p] is not False:
            oldfn = "{:.2f} {:.8f}".format(pfrac, results[p])
            if newfn != oldfn:
               os.remove(oldfn + (" psi.dat" if args.erasure else " rho.dat"))
               os.remove(oldfn + " choi.dat")
         results[p] = f
         if direction <= 0:
            newp = p - 1
            if newp >= args.pmin:
               active += 1
               pool.submit(runb, newp * pmultiple, rho).add_done_callback(
                  lambda x, p=newp: done(p, -1, *x.result())
               )
         if direction >= 0:
            newp = p + 1
            if newp <= args.pmax:
               active += 1
               pool.submit(runb, newp * pmultiple, rho).add_done_callback(
                  lambda x, p=newp: done(p, 1, *x.result())
               )
      active -= 1

   if platform.system() in {'Windows', 'Darwin'}:
      # Windows: no fork, all processes have to initialize by themselves
      # Mac: broken fork, so Python spawns by default
      pool = fut.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                     initargs=initargs)
   else:
      # *nix:
      initialize(*initargs)
      pool = fut.ProcessPoolExecutor(max_workers=args.workers)
   active = 0
   for p in range(args.pmin, args.pmax +1):
      active += 1
      pool.submit(runa, p * pmultiple).add_done_callback(lambda x, p=p: done(p, 0, *x.result()))
   while active > 0:
      time.sleep(.1)
   pool.shutdown()

   print("Total time: {:.2f} s".format(time.time() - starttime))