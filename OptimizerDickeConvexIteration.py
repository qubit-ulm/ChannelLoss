# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that performs the optimization using the convex iteration procedure in
# the computational basis. This will work for all choices made in DickeBasis.

import numpy as np
import scipy.linalg as spla
import mosek
import DickeOptimizer
from math import inf

# note that this optimization is done using the dualized problem
class Optimizer(DickeOptimizer.Optimizer):
   def __setupTask(self):
      self._dChoi = len(self._multinomialsR)
      self._dRho = len(self._multinomialsS)
      indicesR = list(self._multinomialsR.keys())
      indicesS = list(self._multinomialsS.keys())

      trKeys = tuple(sorted(frozenset(map(lambda x: x[1:], indicesR))))
      indPos = {k: v for k, v in zip(indicesR, range(self._dRho))}
      dtrOut = len(trKeys)

      def getIndex(row, col):
         return max(row, col), min(row, col)

      def listToDict(l):
         d = dict()
         for fac, m1, m2 in l:
            if (m1, m2) in d:
               d[(m1, m2)] += fac
            else:
               d[(m1, m2)] = fac
         return d

      # rhoFinal will be a list containing the six relevant element elements of the final density
      # matrix, in the following form: each matrix element is a list of 3-tuples, where the first
      # entry is a multiplicative factor, the second and third are row and column indices of the
      # Choi and rho matrix (lower triangle), all are to be multiplied and the whole list to be
      # summed over.
      rhoFinal = [[(self._multinomialsSR[kpminusk] *
                    np.sqrt(
                       self._multinomialsR[indicesR[b1k]] / self._multinomialsS[indicesS[a1kp]] *
                       self._multinomialsR[indicesR[b2l]] / self._multinomialsS[indicesS[a2lp]]
                    ),
                    getIndex(b1k, b2l), getIndex(a1kp, a2lp)
                   ) for b1k in range(self._dChoi) if indicesR[b1k][0] == b1
                     for b2l in range(self._dChoi) if indicesR[b2l][0] == b2
                     for a1kp in range(self._dRho) if indicesS[a1kp][0] == a1
                     for a2lp in range(self._dRho) if indicesS[a2lp][0] == a2
                     for kpminusk in (tuple(x - y for x, y in zip(indicesS[a1kp][1:],
                                                                  indicesR[b1k][1:])),)
                     if kpminusk in self._multinomialsSR and \
                        kpminusk == tuple(x - y for x, y in zip(indicesS[a2lp][1:],
                                                                indicesR[b2l][1:]))]
                  for a1, b1, a2, b2 in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1),
                                         (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1)]]
      psucc = listToDict(rhoFinal[0] + rhoFinal[2] + rhoFinal[3] + rhoFinal[5])
      fid = listToDict(rhoFinal[0] + rhoFinal[1] + rhoFinal[4] + rhoFinal[5]) # still divide by 2
      allQuadratics = set(psucc.keys()) | set(fid.keys())
      allQuadratics = {k: 3 + v for k, v in zip(allQuadratics, range(len(allQuadratics)))}

      self._task.appendbarvars([self._dChoi, self._dRho, dtrOut] +
                               ([3] * len(allQuadratics)))
      self._task.putobjsense(mosek.objsense.maximize)

      self._rankMatrices = len(allQuadratics)

      # for the dual part, we always will need to multiply the off-diagonals by 2, so just do it
      # here
      extractions = {
         (d, i, j): self._task.appendsparsesymmat(d, [i], [j], [1.0])
         for d in {self._dChoi, self._dRho, dtrOut, 3}
         for j, i in zip(*np.triu_indices(d))
      }
      # dual form. oneDual free, pdual free, fdual >= 0, trrhoDual >= 0,
      self._task.appendvars(4)
      self._task.putvarboundlist([0, 1, 2, 3],
                                 [mosek.boundkey.fr, mosek.boundkey.fr, mosek.boundkey.lo,
                                  mosek.boundkey.lo],
                                 [-inf, -inf, 0, 0], [inf] *4)
      # In principle, we just need an indexable and ordered set, which we implement by a dictionary
      # whose values are the indices.
      qvars = {1: 0}
      index = 0
      for choiIdx, rhoIdx in allQuadratics:
         for x in (("c", choiIdx), ("r", rhoIdx), (choiIdx, rhoIdx), ("c2", choiIdx),
                   ("r2", rhoIdx)):
            if x not in qvars:
               index += 1
               qvars[x] = index
      for j, i in zip(*np.triu_indices(self._dChoi)):
         if ("c", (i, j)) not in qvars:
            index += 1
            qvars[("c", (i, j))] = index
      for j, i in zip(*np.triu_indices(self._dRho)):
         if ("r", (i, j)) not in qvars:
            index += 1
            qvars[("r", (i, j))] = index
      # we now need to append the constraints in the order of the quadratic matrix elements
      self._task.appendcons(max(qvars.values()) +1)

      # dual to 1 = 1
      self._task.putaij(0, qvars[1], 1)
      self._task.putcj(0, 1)
      # dual to extractProb[qvars] = pdist
      self._task.putacol(1, [qvars[q] for q in psucc], psucc.values())
      # putcj appropriately
      # dual to extractFid[qvars] = pdist*f
      self._task.putacol(2, [qvars[q] for q in fid], [.5 * x for x in fid.values()])
      # putcj appropriately
      # dual to tr[extractRho[qvars]] <= 1
      self._task.putacol(3, [qvars[("r", (i, i))] for i in range(self._dRho)],
                         [-1.0] * self._dRho)
      self._task.putcj(3, -1)
      # dual to C >> 0
      abarqueue = dict()
      def putbaraij(i, j, sub, weight):
         if (i, j) in abarqueue:
            if sub in abarqueue[(i, j)]:
               abarqueue[(i, j)][sub] += weight
            else:
               abarqueue[(i, j)][sub] = weight
         else:
            abarqueue[(i, j)] = {sub: weight}
      self._choiIndices = [None] * (self._dChoi * (self._dChoi +1) //2)
      idx = 0
      for j, i in zip(*np.triu_indices(self._dChoi)):
         putbaraij(qvars[("c", (i, j))], 0, extractions[(self._dChoi, i, j)], 1.0)
         self._choiIndices[idx] = qvars[("c", (i, j))]
         idx += 1
      # dual to rho >> 0
      self._rhoIndices = [None] * (self._dRho * (self._dRho +1) //2)
      idx = 0
      for j, i in zip(*np.triu_indices(self._dRho)):
         putbaraij(qvars[("r", (i, j))], 1, extractions[(self._dRho, i, j)], 1.0)
         self._rhoIndices[idx] = qvars[("r", (i, j))]
         idx += 1
      # dual to trout C << id
      for j, i in zip(*np.triu_indices(dtrOut)):
         p1, p2 = indPos.get((0, *trKeys[i])), indPos.get((0, *trKeys[j]))
         p3, p4 = indPos.get((1, *trKeys[i])), indPos.get((1, *trKeys[j]))
         if p1 is not None and p2 is not None:
            putbaraij(qvars[("c", (max(p1, p2), min(p1, p2)))], 2,
                      extractions[(dtrOut, i, j)], -1.0)
         if p3 is not None and p4 is not None:
            putbaraij(qvars[("c", (max(p3, p4), min(p3, p4)))], 2,
                      extractions[(dtrOut, i, j)], -1.0)
      self._task.putbarcj(2, [extractions[(dtrOut, i, i)] for i in range(dtrOut)],
                          [-1.0] * dtrOut)
      # dual to mul_i >> 0
      self._mulIndices = [None] * len(allQuadratics)
      for idx, i in allQuadratics.items():
         putbaraij(qvars[("c2", idx[0])], i, extractions[(3, 0, 0)], 1.0)
         putbaraij(qvars[idx], i, extractions[(3, 1, 0)], 1.0)
         putbaraij(qvars[("c", idx[0])], i, extractions[(3, 2, 0)], 1.0)
         putbaraij(qvars[("r2", idx[1])], i, extractions[(3, 1, 1)], 1.0)
         putbaraij(qvars[("r", idx[1])], i, extractions[(3, 2, 1)], 1.0)
         putbaraij(qvars[1], i, extractions[(3, 2, 2)], 1.0)
         self._mulIndices[i -3] = (qvars[("c2", idx[0])], qvars[idx], qvars[("c", idx[0])],
                                   qvars[("r2", idx[1])], qvars[("r", idx[1])], qvars[1])
      for k, v in abarqueue.items():
         self._task.putbaraij(*k, v.keys(), v.values())

      self._task.putconboundlistconst(qvars.values(), mosek.boundkey.fx, 0.0, 0.0)

   def optimize(self, pdist, f, reuse):
      """
         Performs convex iterations until the rank criterion is violated by less than 10^-8, no
         progress was made for 50 iterations, or an error occurs.

         pdist: distillation success probability
         f:     required fidelity
         reuse: set to True to use the rank matrix directions from the previous call of optimize as
                starting points; else, we start with the identity matrix.

         return: tuple(boolean that indicates success,
                       vec of best Choi matrix, vec of best density matrix)
      """
      self._task.putclist([1, 2], [pdist, pdist * f])

      resultA = np.empty((3, 3))
      resultAIndices = np.triu_indices(3)
      resultChoi = np.empty((self._dChoi,) *2)
      resultChoiIndices = np.triu_indices(self._dChoi)

      lastRank = [inf] * self._rankMatrices
      bestRank = inf
      bestRankIteration = 0

      def putconboundlist(cons, vals):
         for k, v in zip(cons, vals):
            if k in bounds:
               bounds[k] += v
            else:
               bounds[k] = v

      if not reuse:
         bounds = dict()
         for i in range(self._rankMatrices):
            putconboundlist(self._mulIndices[i],
                            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0] # identity matrix - faster, but more
                                                           # likely not to recover from stalls. We
                                                           # use it anyway, as the convex iteration
                                                           # only provides a seed for the following
                                                           # algorithms
                            # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # matrix of ones - probably better use
                                                             # this if you want to use the iteration
                                                             # scheme without further processing
                           )
         self._task.putconboundlist(bounds.keys(), [mosek.boundkey.fx] * len(bounds),
                                    bounds.values(), bounds.values())
      bestChoi = np.empty(len(self._choiIndices))
      bestRho = np.empty(len(self._rhoIndices))

      duals = np.zeros(self._task.getnumcon())
      iterations = 0
      while True:
         iterations += 1
         self._task.optimize()
         if self._task.getsolsta(mosek.soltype.itr) != mosek.solsta.optimal:
            #print("No optimal solution found in iteration {:d}".format(iterations))
            return False, bestChoi, bestRho
         self._task.gety(mosek.soltype.itr, duals)
         bounds = dict()
         # multiplication matrices as rank 1
         for i in range(self._rankMatrices):
            resultA[resultAIndices] = duals[(self._mulIndices[i],)]
            eVal, eVec = spla.eigh(resultA, lower=False, check_finite=False)
            rankViolation = max(eVal[:2])
            rankViolation = max(rankViolation, rankViolation / eVal[-1])
            lowEVSys = eVec[:, :2]
            if rankViolation >= 1e-8 and rankViolation > .95 * lastRank[i]:
               # fix stall
               np.dot(lowEVSys, np.outer(.01 * np.random.rand(2), eVec[:, 2]) + lowEVSys.T,
                      out=resultA)
            else:
               np.dot(lowEVSys, lowEVSys.T, out=resultA)
            putconboundlist(self._mulIndices[i],
                            [resultA[0, 0], 2*resultA[0, 1], 2*resultA[0, 2],
                             resultA[1, 1], 2*resultA[1, 2], resultA[2, 2]])
            lastRank[i] = rankViolation

         self._task.putconboundlist(bounds.keys(), [mosek.boundkey.fx] * len(bounds),
                                    bounds.values(), bounds.values())

         thisRank = max(lastRank)
         progress = thisRank < .95 * bestRank
         if thisRank < bestRank:
            bestRank = thisRank
            np.copyto(bestChoi, duals[(self._choiIndices,)])
            np.copyto(bestRho, duals[(self._rhoIndices,)])
            if bestRank < 1e-8:
               #print("Finished in {:d} iterations with rank {:e}".format(iterations, bestRank))
               return True, bestChoi, bestRho
         if not progress and (iterations - bestRankIteration) % 50 == 0:
            #print("Canceled after {:d} iterations with rank {:e}".format(iterations, bestRank))
            return False, bestChoi, bestRho
         if thisRank < bestRank:
            bestRankIteration = iterations