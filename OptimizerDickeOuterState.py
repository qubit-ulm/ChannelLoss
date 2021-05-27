# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that performs the optimization using a general solver for the variables
# pertaining to the input state and a convex solver for the map. This corresponds to program 15b.

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
import mosek
import DickeOptimizer

class Optimizer(DickeOptimizer.Optimizer):
   def __setupTask(self):
      self._d = len(self._multinomialsR)
      self._indicesR = list(self._multinomialsR.keys())
      self._indicesS = list(self._multinomialsS.keys())

      trKeys = tuple(frozenset(map(lambda x: x[1:], self._indicesR)))
      dtrOut = len(trKeys)
      indPos = {k: v for k, v in zip(self._indicesR, range(len(self._indicesR)))}

      self._task.appendcons(1 + dtrOut * (dtrOut +1) //2)
      self._task.appendbarvars([self._d, dtrOut])
      self._task.putobjsense(mosek.objsense.maximize)

      self._extractions = {
         (i, j): self._task.appendsparsesymmat(self._d, [i], [j], [1.0 if i == j else 0.5])
         for i in range(self._d) for j in range(i +1)
      }

      # linear constraint: traced out map
      curcon = 1
      for i in range(dtrOut):
         for j in range(i +1):
            p1, p2 = indPos.get((0, *trKeys[i])), indPos.get((0, *trKeys[j]))
            p3, p4 = indPos.get((1, *trKeys[i])), indPos.get((1, *trKeys[j]))
            if ((p1 is None or p2 is None) and (p3 is None or p4 is None)):
               self._task.putbaraij(curcon, 1,
                                    [self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                   [1.0 if i == j else 0.5])],
                                    [1.0])
            elif p1 is None or p2 is None:
               self._task.putbaraijlist([curcon] *2, [0, 1], [0, 1], [1, 2],
                                        [self._extractions[(max(p3, p4), min(p3, p4))],
                                         self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                       [1.0 if i == j else 0.5])],
                                        [1.0] *2)
            elif p3 is None or p4 is None:
               self._task.putbaraijlist([curcon] *2, [0, 1], [0, 1], [1, 2],
                                        [self._extractions[(max(p1, p2), min(p1, p2))],
                                         self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                       [1.0 if i == j else 0.5])],
                                        [1.0] *2)
            else:
               self._task.putbaraijlist([curcon] *2, [0, 1], [0, 2], [2, 3],
                                        [self._extractions[(max(p1, p2), min(p1, p2))],
                                         self._extractions[(max(p3, p4), min(p3, p4))],
                                         self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                       [1.0 if i == j else 0.5])],
                                        [1.0] *3)
            self._task.putconbound(curcon, mosek.boundkey.fx,
                                   0.0 if i != j else 1.0, 0.0 if i != j else 1.0)
            curcon += 1

      self.__setupMatrices()

   def __setupMatrices(self):
      self._leftIndices = np.triu_indices(self._d)
      self._rightIndices = np.triu_indices(len(self._multinomialsS))
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
      self._rhoMatrix = np.empty((len(self._multinomialsS),) *2)

   def _setupDistillation(self, rhoVec):
      distFid = self._fidMat @ rhoVec
      distTr = self._trMat @ rhoVec
      # fidelity objective
      self._task.putbarcj(0, [self._extractions[(self._leftIndices[1][i], self._leftIndices[0][i])]
                              for i in range(len(self._leftIndices[0])) if distFid[i] != 0],
                          [distFid[i] for i in range(len(self._leftIndices[0])) if distFid[i] != 0])
      # trace constraint
      self._task.putbaraij(0, 0,
                           [self._extractions[(self._leftIndices[1][i], self._leftIndices[0][i])]
                            for i in range(len(self._leftIndices[0])) if distTr[i] != 0],
                           [distTr[i] for i in range(len(self._leftIndices[0])) if distTr[i] != 0])

   def _optimizeDistillation(self, pdist, retState=False):
      # trace constraint: trace must be fixed to the given value
      self._task.putconbound(0, mosek.boundkey.fx, pdist, pdist)
      self._task.optimize()
      solsta = self._task.getsolsta(mosek.soltype.itr)
      if solsta == mosek.solsta.optimal:
         fid = self._task.getprimalobj(mosek.soltype.itr) / pdist
         if retState:
            choi = np.empty(self._d * (self._d +1)//2)
            self._task.getbarxj(mosek.soltype.itr, 0, choi)
            return (fid, choi)
         else:
            return fid
      else:
         if retState:
            return (0, np.empty(self._d * (self._d +1)//2))
         else:
            return 0

   def _minimizer(self, x, pdist, retState=False):
      # make sure the initial state is normalized
      nrm = np.linalg.norm(x)
      if nrm != 1:
         x /= nrm
      np.outer(x, x, out=self._rhoMatrix)

      self._setupDistillation(self._rhoMatrix[self._rightIndices])
      if retState:
         _, choi = self._optimizeDistillation(pdist, True)
         return choi, self._rhoMatrix[self._rightIndices]
      else:
         f = self._optimizeDistillation(pdist)
         return -f

   def optimize(self, pdist, initialRhoVec):
      """
         pdist: distillation success probability
         initialRhoVec: vectorized upper triangle of the initial density matrix, of which the
                        dominant eigenvector is taken as actual pure state initializer

         return: tuple(fidelity, vec of optimal Choi matrix, vec of optimal density matrix)
      """
      assert len(initialRhoVec) == len(self._rightIndices[0])
      initialRho = np.empty((len(self._multinomialsS),) *2)
      initialRho[self._rightIndices] = initialRhoVec
      initialPsi = spla.eigh(initialRho, lower=False, check_finite=False,
                             subset_by_index=(initialRho.shape[0] -1,) *2)[1][:, -1]
      #import cma
      #es = cma.CMAEvolutionStrategy(initialPsi, .5)
      #mini = es.optimize(self._minimizer, args=(pdist,), verb_disp=0)
      #return (-mini.result[1], *self._minimizer(mini.result[0], pdist, True))
      with np.errstate(divide='ignore'):
         mini = spopt.minimize(self._minimizer, initialPsi, args=(pdist,), method='BFGS')
      return (-mini.fun, *self._minimizer(mini.x, pdist, True))