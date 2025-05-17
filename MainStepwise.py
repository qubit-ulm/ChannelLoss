# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# This is a sample command line application that can be used for optimization with fixed s and r.

import numpy as np
import scipy.linalg as spla
import mosek
import OptimizerStepwiseConvexIteration
import OptimizerStepwiseOuterState
import OptimizerStepwiseKnownMap
import CanonicalSubbasis
from math import inf

# We need to index to probability and do arithmetics, so don't use floats. Every probability is in
# multiples of this.
pmultiple = 1/100

class SharedVariable:
   def __init__(self, x):
      self.x = x

def opti(optMap, optState, fidBound, initMaps):
   #if fidBound < SHARED.x - 1e-8:
   #   # we already found something better than our bound in the meantime
   #   return (optMap.prepPattern, fidBound, -inf, False)
   best = -1
   for initMap in initMaps:
      _, _, _, lastPsi = optMap.optimize(PDIST, initMap)
      lastF = 0
      lastPerturbation = -10
      for iteration in range(100): # no more than a certain amount of iterations
         lastChoi = optState.parseChoiVec(optState._minimizer(lastPsi, PDIST, retState=True,
                                                              isCartesian=True))
         newF, _, _, lastPsi = optMap.optimize(PDIST, lastChoi)
         if newF > best:
            best = newF
            bestVec = lastPsi
         if newF - lastF < 1e-6:
            if iteration - lastPerturbation < 10:
               break
            else:
               lastPsi = lastPsi + np.random.rand(len(lastPsi))*.01
               lastPsi /= np.linalg.norm(lastPsi)
               lastPerturbation = iteration
         lastF = newF
   best, _, _, bestVec = optState.optimize(PDIST, initialPsi=bestVec)
   return (optMap.prepPattern, fidBound, best, min(abs(bestVec)) <= 1e-3)

def setInit(env, d, s, r, pdist, shared):
   global ENV, D, S, R, PDIST, SHARED
   ENV, D, S, R, PDIST, SHARED = env, d, s, r, pdist, shared

class OptimizeStepwise:
   def __init__(self, env, d, s, r, pdist):
      assert d > 1 and s > r and r >= 1 and 0 < pdist and pdist <= 1
      self.env, self.d, self.s, self.r, self.pdist = env, d, s, r, pdist
      self.__shared = SharedVariable(.5)
      import concurrent.futures as fut
      import psutil
      self.pool = fut.ThreadPoolExecutor(max_workers=int(os.environ.get('SLURM_NTASK',
                                                            psutil.cpu_count(logical=False) or 1)),
                                         initializer=setInit,
                                         initargs=(env, d, s, r, pdist, self.__shared))

   def optimize(self, minKets):
      def done(fut):
         nonlocal best, improved, running
         prepPattern, fidBound, opt, invalid = fut.result()
         print(prepPattern, ":", opt, "<=", fidBound, invalid, flush=True)
         if opt > best[0] + 1e-5:
            self.__shared.x = opt
            best = (opt, prepPattern)
            improved = True
         running -= 1

      import time
      best = (.5,)
      self.__shared.x = .5
      notImprovedSince = 0
      for sbl in range(minKets, self.d**self.s +1):
         improved = False
         running = 0
         uniqueIdentifiers = {}
         for prepPattern in CanonicalSubbasis._canonicalPrepPatterns(self.d, self.s, sbl):
            optMap = OptimizerStepwiseKnownMap.Optimizer(ENV, D, S, R, prepPattern)
            uniqueID = optMap.uniqueIdentifierHeuristic()
            if uniqueID is False:
               print(prepPattern, ": redundant dofs")
               continue
            elif uniqueID in uniqueIdentifiers:
               print(prepPattern, ": same as ", uniqueIdentifiers[uniqueID])
               continue
            else:
               uniqueIdentifiers[uniqueID] = prepPattern
            optState = OptimizerStepwiseOuterState.Optimizer(ENV, D, S, R, prepPattern)
            initMaps = tuple(optMap.guessOptimalMap())
            # While there are different maps possible, if there is no successful distillation for a
            # slot in one map, none will be successful there. Use this to bound the fidelity.
            fidBound = sum(1 if m else .5 for m in initMaps[0].values()) / len(initMaps[0])
            if fidBound > best[0] + 1e-8:
               running += 1
               self.pool.submit(opti, optMap, optState, fidBound, initMaps).add_done_callback(done)
            else:
               print(prepPattern,
                     ": bound {:.8f} below current best estimate {:.8f}".format(fidBound, best[0]))
         while running > 0:
            time.sleep(.1)
         print(best)
         print(flush=True)
         if not improved:
            notImprovedSince += 1
            if notImprovedSince == 3:
               break
      return best

   def _optimalBasis(self, subbasisLength):
      assert subbasisLength > 1
      def zeroFillKet(alice, sendEnd=tuple()):
         if len(sendEnd) > self.s:
            raise Exception("Maximum reached")
         return (alice,) + ((0,) * (self.s - len(sendEnd))) + sendEnd

      if subbasisLength == 2:
         yield [zeroFillKet(0, (1,)), zeroFillKet(1)]
      elif subbasisLength == 3:
         yield [zeroFillKet(0, (1,)), zeroFillKet(0, (1, 0)), zeroFillKet(1)]
      elif self.s == 3 and self.r == 2:
         if subbasisLength == 4:
            yield [(0, 0, 0, 1), (0, 1, 1, 0), (1, 0, 0, 0), (1, 1, 1, 1)]
         elif subbasisLength == 5:
            yield [(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 1), (1, 0, 1, 0), (1, 1, 0, 0)]
         elif subbasisLength == 6:
            yield [(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 0), (1, 1, 0, 1)]
            yield [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 0), (1, 1, 1, 1)]
         else:
            raise Exception("Maximum reached")
      else:
         yield [zeroFillKet(0, (1,) + ((0,) * i)) for i in range(subbasisLength -1)] + [zeroFillKet(1)]
         if self.s > 3:
            if subbasisLength == 4:
               yield [zeroFillKet(0, (1, 1)), zeroFillKet(0, (1, 1, 0, 0)), zeroFillKet(1), zeroFillKet(1, (1, 1, 1, 1))]
            elif subbasisLength == 5:
               yield [zeroFillKet(0, (1, 1)), zeroFillKet(0, (1, 1, 0, 0)), zeroFillKet(0, (1, 1, 0, 0, 0, 0)), zeroFillKet(1), zeroFillKet(1, (1, 1, 1, 1))]
            elif subbasisLength == 6:
               yield [zeroFillKet(0, (1,)), zeroFillKet(0, (1, 0)), zeroFillKet(0, (1, 1, 1, 1)), zeroFillKet(1), zeroFillKet(1, (1, 1, 0, 1)), zeroFillKet(1, (1, 1, 1, 0))]
            else:
               raise Exception("Not implemented")

if __name__ == '__main__':
   import argparse
   import os
   env = mosek.Env()
   parser = argparse.ArgumentParser("")
   parser.add_argument("d", help="Ditness", type=int, default=2)
   parser.add_argument("s", help="Send", type=int, default=3)
   parser.add_argument("r", help="Receive", type=int, default=2)
   parser.add_argument("pdist", help="Distillation succes probability", type=float, default=1)
   parser.add_argument("--min", help="Minimum number of kets", type=int, default=2)
   args = parser.parse_args()
   print("d = {}, s = {}, r = {}, pdist = {:.2f}".format(args.d, args.s, args.r, args.pdist))
   setInit(env, args.d, args.s, args.r, args.pdist, None)

   opt = OptimizeStepwise(ENV, D, S, R, PDIST)
   opt.optimize(max(args.min, 2))
