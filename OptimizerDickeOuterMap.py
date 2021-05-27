# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that perform the optimization using a general solver for the variables
# pertaining to the map and a convex solver for the input state. This corresponds to program 15a.

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
import mosek
import DickeOptimizer

# We need an itemgetter that, unlike the one from operator, always returns a tuple.
def itemgetter(*items):
   def g(obj):
      return tuple(obj[item] for item in items)
   return g

class Optimizer(DickeOptimizer.Optimizer):
   def __setupTask(self):
      self._d = len(self._multinomialsS)
      self._indicesR = list(self._multinomialsR.keys())
      self._indicesS = list(self._multinomialsS.keys())

      trKeys = tuple(frozenset(map(lambda x: x[1:], self._indicesR)))
      dtrOut = len(trKeys)
      indPos = {k: v for k, v in zip(self._indicesR, range(len(self._indicesR)))}

      self._task.appendcons(2)
      self._task.appendbarvars([self._d])
      self._task.putobjsense(mosek.objsense.maximize)

      self._extractions = {
         (i, j): self._task.appendsparsesymmat(self._d, [i], [j], [1.0 if i == j else 0.5])
         for i in range(self._d) for j in range(i +1)
      }

      # linear constraint: total trace
      self._task.putbaraij(1, 0, [self._extractions[(i, i)] for i in range(self._d)],
                           [1.0] * self._d)
      self._task.putconbound(1, mosek.boundkey.fx, 1, 1)

      # PSD constraint: trace non-increasing map
      self._trOut = [None] * (dtrOut * (dtrOut +1) //2)
      self._trOutM = np.zeros((dtrOut, dtrOut))
      idx = 0
      for i in range(dtrOut):
         for j in range(i +1):
            p1, p2 = indPos.get((0, *trKeys[i])), indPos.get((0, *trKeys[j]))
            p3, p4 = indPos.get((1, *trKeys[i])), indPos.get((1, *trKeys[j]))
            if ((p1 is None or p2 is None) and (p3 is None or p4 is None)):
               self._trOut[idx] = tuple()
            elif p1 is None or p2 is None:
               self._trOut[idx] = ((min(p3, p4), max(p3, p4)),)
            elif p3 is None or p4 is None:
               self._trOut[idx] = ((min(p1, p2), max(p1, p2)),)
            else:
               self._trOut[idx] = ((min(p1, p2), max(p1, p2)), (min(p3, p4), max(p3, p4)))
            idx += 1

      self.__setupMatrices()

   def __setupMatrices(self):
      self._leftIndices = np.triu_indices(len(self._multinomialsR))
      self._rightIndices = np.triu_indices(self._d)
      self._fidMat = np.zeros((len(self._leftIndices[0]), len(self._rightIndices[0])))
      self._trMat = np.zeros_like(self._fidMat)
      for xLeft in range(self._fidMat.shape[0]):
         b1, k = self._indicesR[self._leftIndices[1][xLeft]][0], \
                    self._indicesR[self._leftIndices[1][xLeft]][1:]
         b2, l = self._indicesR[self._leftIndices[0][xLeft]][0], \
                    self._indicesR[self._leftIndices[0][xLeft]][1:]
         for xRight in range(self._fidMat.shape[1]):
            a1, kp = self._indicesS[self._rightIndices[1][xRight]][0], \
                        self._indicesS[self._rightIndices[1][xRight]][1:]
            a2, lp = self._indicesS[self._rightIndices[0][xRight]][0], \
                        self._indicesS[self._rightIndices[0][xRight]][1:]
            factor = np.sqrt(
               self._multinomialsR[(b1, *k)] / self._multinomialsS[(a1, *kp)] *
               self._multinomialsR[(b2, *l)] / self._multinomialsS[(a2, *lp)]
            )
            kpminusk = tuple(x - y for x, y in zip(kp, k))
            lpminusl = tuple(x - y for x, y in zip(lp, l))
            if k != l or kp != lp:
               kpminusl = tuple(x - y for x, y in zip(kp, l))
               lpminusk = tuple(x - y for x, y in zip(lp, k))
            if a1 == b1 and a2 == b2:
               # fidelity: 00 00; 00 11; 11 00; 11 11
               factor2 = 0
               if a1 == a2:
                  # 00 00 or 11 11
                  if k == l:
                     if kpminusk == lpminusl and kpminusk in self._multinomialsSR:
                        factor2 += .5 * self._multinomialsSR[kpminusk]
                     if kp != lp and kpminusl == lpminusk and kpminusl in self._multinomialsSR:
                        factor2 += .5 * self._multinomialsSR[kpminusl]
                  else:
                     if kpminusk == lpminusl and kpminusk in self._multinomialsSR:
                        if kp == lp:
                           factor2 += .5 * self._multinomialsSR[kpminusk]
                        else:
                           factor2 += self._multinomialsSR[kpminusk]
                     if kpminusl == lpminusk and kpminusl in self._multinomialsSR:
                        if kp == lp:
                           factor2 += .5 * self._multinomialsSR[kpminusl]
                        else:
                           factor2 += self._multinomialsSR[kpminusl]
               else:
                  # 11 00; since we only get the lower triangle, we cannot obtain 00 11, so no
                  # factor .5. Note that 11 00 is always strictly off-diagonal and therefore the
                  # complete block is available.
                  if kpminusk == lpminusl and kpminusk in self._multinomialsSR:
                     factor2 += self._multinomialsSR[kpminusk]
               self._fidMat[xLeft, xRight] += factor * factor2
            if a1 == a2 and b1 == b2:
               factor2 = 0
               if k == l:
                  if kpminusk == lpminusl and kpminusk in self._multinomialsSR:
                     factor2 += self._multinomialsSR[kpminusk]
                  if kp != lp and kpminusl == lpminusk and kpminusl in self._multinomialsSR:
                     factor2 += self._multinomialsSR[kpminusl]
               else:
                  if kpminusk == lpminusl and kpminusk in self._multinomialsSR:
                     if kp == lp:
                        factor2 += self._multinomialsSR[kpminusk]
                     else:
                        factor2 += 2 * self._multinomialsSR[kpminusk]
                  if kpminusl == lpminusk and kpminusl in self._multinomialsSR:
                     if kp == lp:
                        factor2 += self._multinomialsSR[kpminusl]
                     else:
                        factor2 += 2 * self._multinomialsSR[kpminusl]
               self._trMat[xLeft, xRight] += factor * factor2

   def _setupDistillation(self, choiVec):
      distFid = choiVec @ self._fidMat
      distTr = choiVec @ self._trMat
      # fidelity objective
      self._task.putbarcj(0,
                          [self._extractions[(self._rightIndices[1][i], self._rightIndices[0][i])]
                           for i in range(len(self._rightIndices[0])) if distFid[i] != 0],
                          [distFid[i] for i in range(len(self._rightIndices[0]))
                                      if distFid[i] != 0])
      # trace constraint
      self._task.putbaraij(0, 0,
                           [self._extractions[(self._rightIndices[1][i], self._rightIndices[0][i])]
                            for i in range(len(self._rightIndices[0])) if distTr[i] != 0],
                           [distTr[i] for i in range(len(self._rightIndices[0])) if distTr[i] != 0])

   def _optimizeDistillation(self, pdist, retState=False):
      # trace constraint: trace must be fixed to the given value
      self._task.putconbound(0, mosek.boundkey.fx, pdist, pdist)
      self._task.optimize()
      solsta = self._task.getsolsta(mosek.soltype.itr)
      if solsta == mosek.solsta.optimal:
         fid = self._task.getprimalobj(mosek.soltype.itr) / pdist
         if retState:
            rho = np.empty(self._d * (self._d +1)//2)
            self._task.getbarxj(mosek.soltype.itr, 0, rho)
            return (fid, rho)
         else:
            return fid
      else:
         if retState:
            return (0, np.empty(self._d * (self._d +1)//2))
         else:
            return 0

   def _minimizer(self, x, pdist, returnMap=False):
      choi = np.empty((len(self._multinomialsR),) *2)
      choi[self._leftIndices] = x
      # make sure the initial map is trace non-increasing
      idx = 0
      n = len(self._trOutM)
      for i in range(n):
         for j in range(i +1):
            self._trOutM[i, j] = sum(itemgetter(*self._trOut[idx])(choi))
            idx += 1
      normalization = spla.eigh(self._trOutM, lower=True,
                                eigvals_only=True,
                                overwrite_a=True, check_finite=False)[-1]
      if normalization > 1:
         choi /= normalization

      self._setupDistillation(choi[self._leftIndices])
      if returnMap:
         _, rho = self._optimizeDistillation(pdist, True)
         return choi[self._leftIndices], rho
      else:
         f = self._optimizeDistillation(pdist)
         return -f

   def optimize(self, pdist, initialChoiVec):
      """
         pdist:          distillation success probability
         initialChoiVec: vectorized upper triangle of the Choi matrix of the distillation map, used
                         as initial point

         return: tuple(fidelity, vec of optimal Choi matrix, vec of optimal density matrix)
      """
      assert len(initialChoiVec) == len(self._leftIndices[0])
      mini = spopt.minimize(self._minimizer, initialChoiVec, args=(pdist,), method='SLSQP',
                            bounds=[(-1, 1)] * len(initialChoiVec))
      return (-mini.fun, *self._minimizer(mini.x, pdist, True))