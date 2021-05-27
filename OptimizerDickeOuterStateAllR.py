# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that performs the optimization using a general solver for the variables
# pertaining to the input state and a convex solver for the maps, where we consider all (relevant)
# values of the number of received particles, r, instead of a single one; the states are still in
# the Dicke basis. This corresponds to the more sophisticated processing discussed in section III.H.

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
import mosek
import GenericOptimizer
import DickeOptimizer
from scipy.special import comb as binomial

def tmap(*args):
   return tuple(map(*args))

class Optimizer(GenericOptimizer.Optimizer):
   def __init__(self, env, ditness, send, ptrans, threshold=.001):
      """
         env:       MOSEK environment
         ditness:   dimensionality of the carriers
         send:      number of particles to be sent (in Dicke form)
         ptrans:    probability of successful transmission
         threshold: let p(r) be the probability that exactly r particles arrive, then we will take
                    into account all r values where p(r) >= threshold * max_u p(u)
      """
      assert ditness >= 2
      assert send > 1
      assert ptrans > 0 and ptrans <= 1
      assert threshold >= 0 and threshold <= 1
      self.ditness = ditness
      self.send = send
      self.ptrans = ptrans
      self.Ptrans = {r: binomial(send, r) * (ptrans**r) * ((1 - ptrans)**(send - r))
                     for r in range(1, send +1)}
      pthreshold = threshold * max(self.Ptrans.values())
      self.Ptrans = {r: p for r, p in self.Ptrans.items() if p >= pthreshold}
      self.rvalues = tuple(self.Ptrans.keys())
      self._multinomialsR = tuple(DickeOptimizer.mulnomReceive(ditness, send, receive)
                                  for receive in self.rvalues)
      self._multinomialsS = DickeOptimizer.mulnomSend(ditness, send, -1)
      self._multinomialsSR = tuple(DickeOptimizer.mulnom(ditness, send - receive)
                                   for receive in self.rvalues)

      super().__init__(env)

   def __setupTask(self):
      self._d = tmap(len, self._multinomialsR)
      self._indicesR = tmap(lambda multinomialsR: list(multinomialsR.keys()),
                            self._multinomialsR)
      self._indicesS = list(self._multinomialsS.keys())

      trKeys = tmap(lambda indicesR: tuple(frozenset(map(lambda x: x[1:], indicesR))),
                    self._indicesR)
      dtrOut = tmap(len, trKeys)
      indPos = tmap(lambda indicesR: {k: v for k, v in zip(indicesR, range(len(indicesR)))},
                    self._indicesR)

      self._task.appendcons(1 + sum(map(lambda dtrOut: dtrOut * (dtrOut +1) //2, dtrOut)))
      self._task.appendbarvars(self._d + dtrOut)
      self._task.putobjsense(mosek.objsense.maximize)

      self._extractions = {
         (d, i, j): self._task.appendsparsesymmat(d, [i], [j], [1.0 if i == j else 0.5])
         for d in self._d for i in range(d) for j in range(i +1)
      }

      # linear constraint: traced out map
      curcon = 1
      for idx, (d, dtrOut, indPos, trKeys) in enumerate(zip(self._d, dtrOut, indPos, trKeys)):
         for i in range(dtrOut):
            for j in range(i +1):
               p1, p2 = indPos.get((0, *trKeys[i])), indPos.get((0, *trKeys[j]))
               p3, p4 = indPos.get((1, *trKeys[i])), indPos.get((1, *trKeys[j]))
               if ((p1 is None or p2 is None) and (p3 is None or p4 is None)):
                  self._task.putbaraij(curcon, len(self.rvalues) + idx,
                                       [self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                      [1.0 if i == j else 0.5])],
                                       [1.0])
               elif p1 is None or p2 is None:
                  self._task.putbaraijlist([curcon] *2, [idx, len(self.rvalues) + idx],
                                           [0, 1], [1, 2],
                                           [self._extractions[(d, max(p3, p4), min(p3, p4))],
                                            self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                          [1.0 if i == j else 0.5])],
                                           [1.0] *2)
               elif p3 is None or p4 is None:
                  self._task.putbaraijlist([curcon] *2, [idx, len(self.rvalues) + idx],
                                           [0, 1], [1, 2],
                                           [self._extractions[(d, max(p1, p2), min(p1, p2))],
                                            self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                          [1.0 if i == j else 0.5])],
                                           [1.0] *2)
               else:
                  self._task.putbaraijlist([curcon] *2, [idx, len(self.rvalues) + idx],
                                           [0, 2], [2, 3],
                                           [self._extractions[(d, max(p1, p2), min(p1, p2))],
                                            self._extractions[(d, max(p3, p4), min(p3, p4))],
                                            self._task.appendsparsesymmat(dtrOut, [i], [j],
                                                                          [1.0 if i == j else 0.5])],
                                           [1.0] *3)
               self._task.putconbound(curcon, mosek.boundkey.fx,
                                      0.0 if i != j else 1.0, 0.0 if i != j else 1.0)
               curcon += 1

      self.__setupMatrices()

   def __setupMatrices(self):
      self._leftIndices = tmap(np.triu_indices, self._d)
      self._rightIndices = np.triu_indices(len(self._multinomialsS))
      self._fidMat = tmap(lambda li: np.zeros((len(li[0]), len(self._rightIndices[0]))),
                          self._leftIndices)
      self._trMat = tmap(np.zeros_like, self._fidMat)
      for fidMat, trMat, indicesR, leftIndices, multinomialsR, multinomialsSR in \
         zip(self._fidMat, self._trMat, self._indicesR, self._leftIndices, self._multinomialsR,
             self._multinomialsSR):
         for xLeft in range(fidMat.shape[0]):
            b1, k = indicesR[leftIndices[1][xLeft]][0], indicesR[leftIndices[1][xLeft]][1:]
            b2, l = indicesR[leftIndices[0][xLeft]][0], indicesR[leftIndices[0][xLeft]][1:]
            for xRight in range(fidMat.shape[1]):
               a1, kp = self._indicesS[self._rightIndices[1][xRight]][0], \
                           self._indicesS[self._rightIndices[1][xRight]][1:]
               a2, lp = self._indicesS[self._rightIndices[0][xRight]][0], \
                           self._indicesS[self._rightIndices[0][xRight]][1:]
               factor = np.sqrt(
                  multinomialsR[(b1, *k)] / self._multinomialsS[(a1, *kp)] *
                  multinomialsR[(b2, *l)] / self._multinomialsS[(a2, *lp)]
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
                        if kpminusk == lpminusl and kpminusk in multinomialsSR:
                           factor2 += .5 * multinomialsSR[kpminusk]
                        if kp != lp and kpminusl == lpminusk and kpminusl in multinomialsSR:
                           factor2 += .5 * multinomialsSR[kpminusl]
                     else:
                        if kpminusk == lpminusl and kpminusk in multinomialsSR:
                           if kp == lp:
                              factor2 += .5 * multinomialsSR[kpminusk]
                           else:
                              factor2 += multinomialsSR[kpminusk]
                        if kpminusl == lpminusk and kpminusl in multinomialsSR:
                           if kp == lp:
                              factor2 += .5 * multinomialsSR[kpminusl]
                           else:
                              factor2 += multinomialsSR[kpminusl]
                  else:
                     # 11 00; since we only get the lower triangle, we cannot obtain 00 11, so no
                     # factor .5. Note that 11 00 is always strictly off-diagonal and therefore the
                     # complete block is available.
                     if kpminusk == lpminusl and kpminusk in multinomialsSR:
                        factor2 += multinomialsSR[kpminusk]
                  fidMat[xLeft, xRight] += factor * factor2
               if a1 == a2 and b1 == b2:
                  factor2 = 0
                  if k == l:
                     if kpminusk == lpminusl and kpminusk in multinomialsSR:
                        factor2 += multinomialsSR[kpminusk]
                     if kp != lp and kpminusl == lpminusk and kpminusl in multinomialsSR:
                        factor2 += multinomialsSR[kpminusl]
                  else:
                     if kpminusk == lpminusl and kpminusk in multinomialsSR:
                        if kp == lp:
                           factor2 += multinomialsSR[kpminusk]
                        else:
                           factor2 += 2 * multinomialsSR[kpminusk]
                     if kpminusl == lpminusk and kpminusl in multinomialsSR:
                        if kp == lp:
                           factor2 += multinomialsSR[kpminusl]
                        else:
                           factor2 += 2 * multinomialsSR[kpminusl]
                  trMat[xLeft, xRight] += factor * factor2
      self._rhoMatrix = np.empty((len(self._multinomialsS),) *2)

   def _setupDistillation(self, rhoVec):
      for idx, (fidMat, trMat, leftIndices) in \
         enumerate(zip(self._fidMat, self._trMat, self._leftIndices)):
         distFid = fidMat @ rhoVec
         distTr = trMat @ rhoVec
         prob = self.Ptrans[self.rvalues[idx]]
         # fidelity optimization
         self._task.putbarcj(idx,
                             [self._extractions[(self._d[idx], leftIndices[1][i],
                                                 leftIndices[0][i])]
                              for i in range(len(leftIndices[0])) if distFid[i] != 0],
                             [prob*distFid[i] for i in range(len(leftIndices[0]))
                                              if distFid[i] != 0])
         # trace constraint
         self._task.putbaraij(0, idx,
                              [self._extractions[(self._d[idx], leftIndices[1][i],
                                                  leftIndices[0][i])]
                               for i in range(len(leftIndices[0])) if distTr[i] != 0],
                              [prob*distTr[i] for i in range(len(leftIndices[0]))
                                              if distTr[i] != 0])

   def _optimizeDistillation(self, ptot, retState=False):
      # trace constraint: trace must be fixed to the given value
      self._task.putconbound(0, mosek.boundkey.fx, ptot, ptot)
      self._task.optimize()
      solsta = self._task.getsolsta(mosek.soltype.itr)
      if solsta == mosek.solsta.optimal:
         fid = self._task.getprimalobj(mosek.soltype.itr) / ptot
         if retState:
            chois = np.empty(sum(map(lambda d: d * (d +1)//2, self._d)))
            self._task.getbarxslice(mosek.soltype.itr, 0, len(self.rvalues), len(chois), chois)
            return (fid, chois)
         else:
            return fid
      else:
         if retState:
            return (0, np.empty(sum(map(lambda d: d * (d +1)//2, self._d))))
         else:
            return 0

   def _minimizer(self, x, ptot, retState=False):
      nrm = np.linalg.norm(x)
      if nrm != 1:
         x /= nrm
      np.outer(x, x, out=self._rhoMatrix)

      self._setupDistillation(self._rhoMatrix[self._rightIndices])
      if retState:
         _, choi = self._optimizeDistillation(ptot, True)
         return choi, self._rhoMatrix[self._rightIndices]
      else:
         f = self._optimizeDistillation(ptot)
         return -f

   def optimize(self, ptot, initialRhoVec):
      """
         ptot:          total success probability (transmission + distillation)
         initialRhoVec: vectorized upper triangle of the initial density matrix, of which the
                        dominant eigenvector is taken as actual pure state initializer

         return tuple(fidelity, ndarray[r situation][vec of optimal Choi matrix],
                      vec of optimal density matrix)
      """
      assert len(initialRhoVec) == len(self._rightIndices[0])
      initialRho = np.empty((len(self._multinomialsS),) *2)
      initialRho[self._rightIndices] = initialRhoVec
      initialPsi = spla.eigh(initialRho, lower=False, check_finite=False,
                             subset_by_index=(initialRho.shape[0] -1,) *2)[1][:, -1]
      #import cma
      #es = cma.CMAEvolutionStrategy(initialPsi, .5)
      #mini = es.optimize(self._minimizer, args=(ptot,), verb_disp=0)
      #return (-mini.result[1], *self._minimizer(mini.result[0], ptot, True))
      with np.errstate(divide='ignore'):
         mini = spopt.minimize(self._minimizer, initialPsi, args=(ptot,), method='BFGS')
      return (-mini.fun, *self._minimizer(mini.x, ptot, True))