# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that performs the optimization using the convex iteration procedure in
# the computational basis.

import numpy as np
import scipy.linalg as spla
import mosek
import PatternOptimizer
from math import inf
from itertools import accumulate

# note that this optimization is done using the dualized problem
class Optimizer(PatternOptimizer.Optimizer):
   def __setupTask(self):
      self._dRho = len(self.prepPattern)
      # We must make all Chois trace preserving, so that we know what we get upon partial tracing,
      # so we introduce an additional level that flags failure. Although self._leftIndices does not
      # know about the additional level, this actually does not change anything for the index
      # mapping, due to the ordering of the indices.
      self._dChois = tuple((self.dStore +1) * len(b['choiBasis']) for b in self._slotData.values())
      slots = len(self._dChois)
      dtrOuts = tuple(len(b['choiBasis']) for b in self._slotData.values())
      self._task.appendcons(3 + (slots * self._dRho * (self._dRho +1) +
                                 sum(dtrOut * (dtrOut +1) for dtrOut in dtrOuts)
                                )//2)
      # 0: rho
      # 1..slots: full product matrices (dimensions _dRho * _dChois)
      bardims = (self._dRho,) + tuple(self._dRho * x for x in self._dChois)
      self._task.appendbarvars(bardims)
      self._task.putobjsense(mosek.objsense.minimize)

      self._extractions = {}
      def extr(dim, i, j):
         assert i <= j
         if (dim, i, j) not in self._extractions:
            self._extractions[(dim, i, j)] = self._task.appendsparsesymmat(dim, [j], [i],
                                                                           [1.0 if i == j else 0.5])
         return self._extractions[(dim, i, j)]
      # make sure we have the full rho matrix available
      for i, j in zip(*self._rightIndices):
         extr(self._dRho, i, j)

      curcon = 0
      # linear constraint: output success probability & Bell overlap
      # self._trMat[i] is a CSR matrix that is to be left-multiplied with the vectorized upper
      # triangle of the choi matrix and right-multiplied with the vectorized upper triangle of the
      # rho matrix. This product is contained in the full matrices. Full matrix index
      # (i*dChoi + k, j*dChoi + l) contains rho[i, j] * choi[k, l].
      # We need the data in canonicalized coo format (which find would create anyway), and then we
      # can most efficiently just extract the data in the correct order)
      ux = [ui['uniqueIdx'] for ui in self._slotData.values()]
      for mat_str in ('trMat', 'fidMat'):
         mats = [self._uniqueData[ui][mat_str] for ui in ux]
         for i in range(slots):
            mats[i] = mats[i].tocoo()
            mats[i].eliminate_zeros()
            mats[i].sum_duplicates()
         self._task.putbaraijlist([curcon] * slots, list(range(1, slots +1)),
                                  [0] + list(accumulate(mat.getnnz() for mat in mats[:-1])),
                                  list(accumulate(mat.getnnz() for mat in mats)),
                                  [extr(bardims[iMat +1],
                                        self._rightIndices[0][rhoIdx] * choiDim +
                                        self._uniqueData[ux[iMat]]['leftIndices'][0][choiIdx],
                                        self._rightIndices[1][rhoIdx] * choiDim +
                                        self._uniqueData[ux[iMat]]['leftIndices'][1][choiIdx])
                                   for iMat, (choiDim, mat) in
                                       enumerate(zip(self._dChois, mats))
                                   for choiIdx, rhoIdx in zip(mat.row, mat.col)],
                                  np.concatenate(tuple(mat.data for mat in mats)))
         curcon += 1

      # linear constraint: unit trace state
      self._task.putbaraijlist([curcon], [0], [0], [self._dRho],
                               [extr(self._dRho, i, i) for i in range(self._dRho)],
                               [1.0] * self._dRho)
      self._task.putconbound(curcon, mosek.boundkey.fx, 1.0, 1.0)
      curcon += 1

      # linear constraint: tracing out Choi should give dim(choi)*rho
      for iMat, dChoi in enumerate(self._dChois, start=1):
         for i in range(self._dRho):
            for j in range(i, self._dRho):
               self._task.putbaraijlist([curcon] *2, [0, iMat], [0, 1], [1, 1 + dChoi],
                                        [extr(self._dRho, i, j)] +
                                        [extr(bardims[iMat], i*dChoi + k, j*dChoi + k)
                                         for k in range(dChoi)],
                                        [-dChoi/(self.dStore +1)] + ([1.0] * dChoi))
               self._task.putconbound(curcon, mosek.boundkey.fx, 0.0, 0.0)
               curcon += 1

      # linear constraint: traced out map
      num = (self.dStore +1) * self._dRho
      for iMat, (dChoi, dtrOut) in enumerate(zip(self._dChois, dtrOuts), start=1):
         for i in range(dtrOut):
            for j in range(i, dtrOut):
               self._task.putbaraij(curcon, iMat,
                                    [extr(bardims[iMat], k*dChoi + l*dtrOut + i,
                                          k*dChoi + l*dtrOut + j)
                                     for k in range(self._dRho) for l in range(self.dStore +1)],
                                    [1.0] * num)
               self._task.putconbound(curcon, mosek.boundkey.fx,
                                      0.0 if i != j else 1.0, 0.0 if i != j else 1.0)
               curcon += 1

      self._rankMatrix = np.empty((self._dRho * (self._dRho +1) //2,))

   def optimize(self, pdist, f, reuse):
      """
         Performs convex iterations until the rank criterion is violated by less than 10^-8, no
         progress was made for 50 iterations, or an error occurs.

         pdist: distillation success probability
         f:     required fidelity
         reuse: set to True to use the rank matrix directions from the previous call of optimize as
                starting points; else, we start with the identity matrix.

         return: tuple(boolean that indicates success,
                       None, optimal state vector)
      """
      self._task.putconbound(0, mosek.boundkey.fx, pdist, pdist)
      self._task.putconbound(1, mosek.boundkey.lo, pdist * f, inf)
      if not reuse:
         self._rankMatrix.fill(1.0)

      lastRank = inf
      bestRank = inf
      bestRankIteration = 0
      rankMatrix = np.zeros((self._dRho, self._dRho))
      bestRho = np.empty(self._dRho * (self._dRho +1) //2)
      extr = [self._extractions[(self._dRho, i, j)] for i, j in zip(*self._rightIndices)]
      extrMult = [1.0 if i == j else 2.0 for i, j in zip(*self._rightIndices)]
      iterations = 0

      while True:
         iterations += 1
         self._task.putbarcj(0, extr, self._rankMatrix * extrMult)
         self._task.optimize()
         if self._task.getsolsta(mosek.soltype.itr) != mosek.solsta.optimal:
            #print("No optimal solution found in iteration {:d}".format(iterations))
            return False, None, bestRho
         self._task.getbarxj(mosek.soltype.itr, 0, self._rankMatrix)
         rankMatrix[self._rightIndices] = self._rankMatrix
         eVal, eVec = spla.eigh(rankMatrix, lower=False, check_finite=False)
         rankViolation = max(eVal[:-1])
         rankViolation = max(rankViolation, rankViolation / eVal[-1])

         progress = lastRank < .95 * bestRank
         lastRank = rankViolation
         if rankViolation < bestRank:
            bestRank = lastRank
            np.copyto(bestRho, self._rankMatrix)
            if rankViolation < 1e-6:
               #print("Finished in {:d} iterations with rank {:e}".format(iterations, bestRank))
               return True, None, bestRho
            bestRankIteration = iterations
         elif not progress and (iterations - bestRankIteration) % 50 == 0:
            np.copyto(self._rankMatrix, bestRho)
            #print("Canceled after {:d} iterations with rank {:e}".format(iterations, bestRank))
            return False, None, bestRho

         lowEVSys = eVec[:, :-1]
         if rankViolation >= 1e-8 and rankViolation > .95 * lastRank:
            # fix stall
            np.dot(-lowEVSys, np.outer(np.random.rand(self._dRho -1),
                                       eVec[:, -1]) + lowEVSys.T,
                   out=rankMatrix)
         else:
            np.dot(lowEVSys, lowEVSys.T, out=rankMatrix)
         np.copyto(self._rankMatrix, rankMatrix[self._rightIndices])