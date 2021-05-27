# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that performs the optimization using the convex iteration procedure in
# the singular value basis. Note that this will most likely fails if the functions defined in
# DickeBasis do not return the full set of coefficients, since the fidelity and trace matrices will
# then not share a common singular basis. In this case, use the convex iteration routine that does
# not use SVD. (In principle, we could still use an SVD approach in the other case, using different
# bases.)

# We try to choose a most simple representation of the SVD basis in terms of the computational
# basis. Set this value to None (no simplification), 0 (re-SVD approach), or 1 (optimization
# approach, slow and not terribly accurate).
simplificationProcedure = 0

import numpy as np
import scipy.linalg as spla
if simplificationProcedure == 1:
   import scipy.optimize as spopt
import mosek
import DickeOptimizer
from math import inf

def relevantSVD(m, *args, **kwargs):
   """
      Returns the singular value decomposition of m, but removes all zero singular values.
      Returns three lists that contain the left singular vectors, singular values, and right
      singular vectors, respectively.
   """
   u, s, vh = spla.svd(m, *args, full_matrices=False, **kwargs)
   u = u.T
   for i, v in enumerate(s):
      if v < 1e-10:
         u = u[:i]
         s = s[:i]
         vh = vh[:i]
         break
   return u, s, vh

# note for the SVD-based convex iteration, we use the primal formulation (out of no particular
# reason, the dual one would probably be more efficient...)
class Optimizer(DickeOptimizer.Optimizer):
   def __setupTask(self):
      trFactors, fidFactors, choiBasis, rhoBasis = self.__svdDtensor()
      assert len(trFactors) == len(fidFactors) and len(fidFactors) == len(choiBasis) and \
         len(choiBasis) == len(rhoBasis)
      self._bigsvd = np.count_nonzero(trFactors)
      self._smallsvd = len(fidFactors) - self._bigsvd
      self._choiBasis = choiBasis
      self._rhoBasis = rhoBasis

      self._dChoi = len(self._multinomialsR)
      self._dRho = len(self._multinomialsS)
      indicesR = list(self._multinomialsR.keys())

      trKeys = tuple(frozenset(map(lambda x: x[1:], indicesR)))
      dtrOut = len(trKeys)
      indPos = {k: v for k, v in zip(indicesR, range(self._dChoi))}

      self._task.appendcons(3 + dtrOut * (dtrOut +1) //2 + 3 * self._bigsvd + 2 * self._smallsvd +
                            self._smallsvd)
      self._task.appendbarvars([self._dChoi, self._dRho, dtrOut])
      self._task.appendbarvars(([3] * self._bigsvd) + ([2] * self._smallsvd))
      self._task.appendvars(3 * self._smallsvd)
      self._task.putobjsense(mosek.objsense.minimize)

      self._extractions3 = {
         (i, j): self._task.appendsparsesymmat(3, [i], [j], [1.0 if i == j else 0.5])
         for i in range(3) for j in range(i +1)
      }
      self._extractions2 = {
         (i, j): self._task.appendsparsesymmat(2, [i], [j], [1.0 if i == j else 0.5])
         for i in range(2) for j in range(i +1)
      }
      triuMap = np.triu_indices(self._dChoi)
      extractionsChoiSB = tuple(
         self._task.appendsparsesymmat(self._dChoi, triuMap[1], triuMap[0],
                                       [x if triuMap[1][i] == triuMap[0][i] else .5*x
                                        for i, x in enumerate(cb)])
         for cb in choiBasis
      )
      triuMap = np.triu_indices(self._dRho)
      extractionsRhoSB = tuple(
         self._task.appendsparsesymmat(self._dRho, triuMap[1], triuMap[0],
                                       [x if triuMap[1][i] == triuMap[0][i] else .5*x
                                        for i, x in enumerate(rb)])
         for rb in rhoBasis
      )

      # linear constraint: success probability
      curcon = 0
      self._task.putbaraijlist([curcon] * self._bigsvd, range(3, 3 + self._bigsvd),
                               range(self._bigsvd), range(1, self._bigsvd +1),
                               [self._extractions3[(1, 0)]] * self._bigsvd, trFactors)
      curcon += 1
      # linear constraint: Bell overlap
      self._task.putbaraijlist([curcon] * len(fidFactors), range(3, 3 + len(fidFactors)),
                               range(len(fidFactors)), range(1, len(fidFactors) +1),
                               ([self._extractions3[(1, 0)]] * self._bigsvd) +
                               ([self._extractions2[(0, 0)]] * self._smallsvd), fidFactors)
      self._task.putarow(curcon, range(0, 3*self._smallsvd, 3), -fidFactors[self._bigsvd:])
      curcon += 1
      # linear constraint: input state trace
      self._task.putbaraij(curcon, 1,
                           [self._task.appendsparsesymmat(self._dRho, range(self._dRho),
                                                          range(self._dRho), [1.0]*self._dRho)],
                           [1.0])
      self._task.putconbound(curcon, mosek.boundkey.fx, 1.0, 1.0)
      curcon += 1
      # linear constraint: traced out map
      for i in range(dtrOut):
         for j in range(i +1):
            cons = []
            for k in range(2):
               p1, p2 = indPos.get((k, *trKeys[i])), indPos.get((k, *trKeys[j]))
               if p1 is not None and p2 is not None:
                  cons.append((max(p1, p2), min(p1, p2)))
            newidx = self._task.appendsparsesymmat(dtrOut, [i], [j], [1.0 if i == j else 0.5])
            if len(cons) == 0:
               self._task.putbaraij(curcon, 2, [newidx], [1.0])
            else:
               self._task.putbaraij(curcon, 0,
                                    [self._task.appendsparsesymmat(self._dChoi, [x for x, y in cons],
                                                                   [y for x, y in cons],
                                                                   [1.0 if x == y else 0.5
                                                                    for x, y in cons])],
                                    [1.0])
               self._task.putbaraij(curcon, 2, [newidx], [1.0])
            self._task.putconbound(curcon, mosek.boundkey.fx,
                                   0.0 if i != j else 1.0, 0.0 if i != j else 1.0)
            curcon += 1
      # linear constraint: proper content for the big rank matrices
      for i in range(self._bigsvd):
         self._task.putbaraijlist([curcon, curcon, curcon +1, curcon +1, curcon +2],
                                  [0, 3 + i, 1, 3 + i, 3 + i],
                                  range(5), range(1, 6),
                                  [extractionsChoiSB[i], self._extractions3[(2, 0)],
                                   extractionsRhoSB[i], self._extractions3[(2, 1)],
                                   self._extractions3[(2, 2)]],
                                  [1.0, -1.0, 1.0, -1.0, 1.0])
         self._task.putconboundlist(range(curcon, curcon +3), [mosek.boundkey.fx] *3,
                                    [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
         curcon += 3
      # linear constraint: proper content for the small rank matrices
      for i in range(self._bigsvd, self._bigsvd + self._smallsvd):
         self._task.putbaraijlist([curcon, curcon, curcon, curcon +1], [0, 1, 3 + i, 3 + i],
                                  range(4), range(1, 5),
                                  [extractionsChoiSB[i], extractionsRhoSB[i],
                                   self._extractions2[(1, 0)], self._extractions2[(1, 1)]],
                                  [1.0, 1.0, -2.0, 1.0]) # we need a factor of 1/4 in the square
         self._task.putconboundlist(range(curcon, curcon +2), [mosek.boundkey.fx] *2,
                                    [0.0, 1.0], [0.0, 1.0])
         curcon += 2
      # quadratic constraint: lower bounding difference
      self._task.putvarboundsliceconst(0, 3 * self._smallsvd, mosek.boundkey.fr, -inf, inf)
      self._task.putvarboundlistconst(range(1, 3 * self._smallsvd, 3), mosek.boundkey.fx, 0.5, 0.5)
      for idx, i in enumerate(range(0, 3 * self._smallsvd, 3), start=self._bigsvd):
         self._task.putbaraijlist([curcon, curcon], [0, 1], [0, 1], [1, 2],
                                  [extractionsChoiSB[idx], extractionsRhoSB[idx]], [1.0, -1.0])
         self._task.putaij(curcon, i +2, -2.0)
         self._task.putconbound(curcon, mosek.boundkey.fx, 0.0, 0.0)
         self._task.appendconeseq(mosek.conetype.rquad, 0.0, 3, i)
         curcon += 1

      self._rankMatricesBig = np.empty((self._bigsvd, 6))
      self._rankMatricesSmall = np.empty((self._smallsvd, 3))

   def __svdDtensor(self):
      assert simplificationProcedure in {None, 0, 1}
      indicesR = list(self._multinomialsR.keys())
      indicesS = list(self._multinomialsS.keys())
      indicesSR = list(self._multinomialsSR.keys())
      indSpos = {k: v for k, v in zip(indicesS, range(len(indicesS)))}
      rd, sd = len(self._multinomialsR), len(self._multinomialsS)

      def dtens(k, l, kp, lp):
         assert k in self._multinomialsR and l in self._multinomialsR and \
            kp in self._multinomialsS and lp in self._multinomialsS
         kpminusk = tuple(x - y for x, y in zip(kp[1:], k[1:]))
         if kpminusk in self._multinomialsSR:
            if lp[1:] == tuple(x + y for x, y in zip(kpminusk, l[1:])):
               return self._multinomialsSR[kpminusk] * \
                     np.sqrt((self._multinomialsR[k] * self._multinomialsR[l]) /
                             (self._multinomialsS[kp] * self._multinomialsS[lp]))
         return 0

      if simplificationProcedure == 1:
         def gramSchmidt(vecs):
            if len(vecs) == 0:
               return []
            elif len(vecs) == 1:
               return np.array(vecs / np.linalg.norm(vecs[0]))
            output = np.array(vecs, copy=True)
            i = 0
            while i < len(output):
               for j in range(i):
                  output[i] -= (output[j] @ output[i]) * output[j]
               norm = np.linalg.norm(output[i])
               if norm < 1e-8:
                  output = np.delete(output, i, 0)
               else:
                  output[i] /= norm
                  i += 1
            return output

      # first, prepare the two matrices that upon left- and right-multiplication give the trace and
      # overlap with Phi^+, respectively.
      rmap, smap = np.triu_indices(rd), np.triu_indices(sd)
      mTr = np.zeros((len(rmap[0]), len(smap[0])))
      mBell = np.zeros_like(mTr)
      for ji34, i34 in enumerate(zip(*rmap)):
         for ji56, i56 in enumerate(zip(*smap)):
            if i34[0] == i34[1]:
               factor = .5
            else:
               factor = 1.
            if i56[0] == i56[1]:
               factor *= .5
            i3, i4 = indicesR[i34[0]], indicesR[i34[1]]
            i5, i6 = indicesS[i56[0]], indicesS[i56[1]]
            if i5[0] == i6[0] and i3[0] == i4[0]:
               mTr[ji34, ji56] = factor * (dtens(i3, i4, i5, i6) + dtens(i4, i3, i5, i6) +
                                           dtens(i3, i4, i6, i5) + dtens(i4, i3, i6, i5))
            if i5[0] == i3[0] and i6[0] == i4[0]:
               if i5[0] == i4[0] and i6[0] == i3[0]:
                  mBell[ji34, ji56] = factor * .5 * (dtens(i3, i4, i5, i6) + dtens(i4, i3, i6, i5) +
                                                     dtens(i4, i3, i5, i6) + dtens(i3, i4, i6, i5))
               else:
                  mBell[ji34, ji56] = factor * .5 * (dtens(i3, i4, i5, i6) + dtens(i4, i3, i6, i5))
            elif i5[0] == i4[0] and i6[0] == i3[0]:
               mBell[ji34, ji56] = factor * .5 * (dtens(i4, i3, i5, i6) + dtens(i3, i4, i6, i5))
      # perform SVD on sum of the two (since we assume they share an SVD basis)
      leftBasis, d, rightBasis = relevantSVD(mTr + np.pi * mBell, overwrite_a=True, check_finite=False)
      # normalize the SVD basis: check for degenerate subspaces and choose as simple bases as possible
      if simplificationProcedure is not None:
         d = d.round(decimals=8)
         lastSV = d[0]
         subspaceStart = 0
         for i in range(1, len(d)):
            if abs(d[i] - lastSV) > 1e-8 or i == len(d -1):
               if i > subspaceStart +1:
                  if simplificationProcedure == 0:
                     # re-SVD approach: re-compose the subspaces, then chop zero values and perform
                     # another SVD within - it will by its algorithm already be with as few
                     # coefficients as possible in the computational basis...
                     newSVD = sum(np.outer(x, y) for x, y in zip(leftBasis[subspaceStart:i],
                                                                 rightBasis[subspaceStart:i]))
                     newSVD[abs(newSVD) < 1e-10] = 0
                     newSVD = relevantSVD(newSVD, overwrite_a=True, check_finite=False,
                                          lapack_driver='gesvd') # only gesvd will do the job
                     leftBasis[subspaceStart:i] = newSVD[0]
                     rightBasis[subspaceStart:i] = newSVD[2]
                  else:
                     # optimization approach (slow and relatively low accuracy)
                     subspaceL = leftBasis[subspaceStart:i]
                     subspaceR = rightBasis[subspaceStart:i]
                     restBasisL = subspaceL
                     restBasisR = subspaceR
                     for j in range(len(subspaceL)):
                        assert len(restBasisL) == len(subspaceL) - j
                        # sequentially find new simple singular vectors
                        opt = spopt.minimize(
                           lambda x: np.linalg.norm(x @ restBasisL, 1),
                           np.full((len(restBasisL),), 1/np.sqrt(len(restBasisL))),
                           constraints={'type': 'eq', 'fun': lambda x: np.linalg.norm(x) -1,
                                        'jac': lambda x: x/np.linalg.norm(x)},
                           method='SLSQP',
                        )
                        x = opt.x / np.linalg.norm(opt.x)
                        subspaceL[j] = x @ restBasisL
                        subspaceR[j] = x @ restBasisR
                        restBasisL = gramSchmidt(np.vstack((subspaceL[j], restBasisL)))[1:]
                        restBasisR = gramSchmidt(np.vstack((subspaceR[j], restBasisR)))[1:]
               subspaceStart = i
               lastSV = d[i]
         # check orthonormality (we don't raise an exception here, but if the violation is too large,
         # the two matrices do not share a common eigenbasis. The results will then be meaningless.
         if not np.allclose(leftBasis @ leftBasis.T, np.eye(len(leftBasis))):
            print("orthonormality of left basis violated by {:.8f}".format(
               np.linalg.norm(leftBasis @ leftBasis.T - np.eye(len(leftBasis)), inf)
            ))
         if not np.allclose(rightBasis @ rightBasis.T, np.eye(len(rightBasis))):
            print("orthonormality of right basis violated by {:.8f}".format(
               np.linalg.norm(rightBasis @ rightBasis.T - np.eye(len(rightBasis)), inf)
            ))
      # extract singular values on original matrices
      sv1 = np.array([l @ mTr @ r for l, r in zip(leftBasis, rightBasis)])
      sv2 = np.array([l @ mBell @ r for l, r in zip(leftBasis, rightBasis)])
      # check coincidence - this will fail if we did not have joint singular bases for both mTr and
      # mBell (which will happen if mulnomSend or mulnomReceive are truncated)
      if not np.allclose(sum(v * np.outer(l, r) for v, l, r in zip(sv1, leftBasis, rightBasis)), mTr):
         print("coincidence of tr violated by {:.8f}".format(
            np.linalg.norm(sum(v * np.outer(l, r)
                               for v, l, r in zip(sv1, leftBasis, rightBasis)) - mTr, inf)
         ))
      if not np.allclose(sum(v * np.outer(l, r)
                             for v, l, r in zip(sv2, leftBasis, rightBasis)), mBell):
         print("coincidence of bell violated by {:.8f}".format(
            np.linalg.norm(sum(v * np.outer(l, r)
                               for v, l, r in zip(sv2, leftBasis, rightBasis)) - mBell, inf)
         ))
      sv1 = sv1.round(decimals=8)
      sv2 = sv2.round(decimals=8)
      order = np.argsort(np.array([(x, y) for x, y in zip(sv1, sv2)],
                                  dtype=[('x', 'float64'), ('y', 'float64')]),
                         order=('x', 'y'))[::-1]
      return sv1[order], sv2[order], leftBasis.round(decimals=8)[order], \
         rightBasis.round(decimals=8)[order]

   def optimize(self, pdist, f, reuse):
      """
         Performs convex iterations until the rank criterion is violated by less than 10^-8, no
         progress was made for 50 iterations, or an error occurs.

         pdist: distillation success probability
         f:     required fidelity
         reuse: set to True to use the rank matrix directions from the previous call of optimize as
                starting points; else, we start with the matrix of all ones.

         return: tuple(boolean that indicates success,
                       vec of best Choi matrix, vec of best density matrix)
      """
      self._task.putconbound(0, mosek.boundkey.fx, pdist, pdist)
      self._task.putconbound(1, mosek.boundkey.lo, pdist * f, inf)
      if not reuse:
         self._rankMatricesBig.fill(1.0)
         self._rankMatricesSmall.fill(1.0)

      lastRank = [inf] * (self._bigsvd + self._smallsvd)
      bestRank = inf
      bestRankIteration = 0
      rankMatrixBig = np.zeros((3, 3))
      rankMatrixSmall = np.zeros((2, 2))
      bestChoi = np.empty(self._dChoi * (self._dChoi +1) //2)
      bestRho = np.empty(self._dRho * (self._dRho +1) //2)
      triusBig = np.triu_indices(3)
      triusSmall = np.triu_indices(2)
      extr3 = [self._extractions3[(0, 0)], self._extractions3[(1, 0)], self._extractions3[(2, 0)],
               self._extractions3[(1, 1)], self._extractions3[(2, 1)], self._extractions3[(2, 2)]]
      extr2 = [self._extractions2[(0, 0)], self._extractions2[(1, 0)], self._extractions2[(1, 1)]]
      iterations = 0
      while True:
         iterations += 1
         for i, j in enumerate(range(self._bigsvd), start=3):
            self._task.putbarcj(i, extr3, self._rankMatricesBig[j] * [1.0, 2.0, 2.0, 1.0, 2.0, 1.0])
         for i, j in enumerate(range(self._smallsvd), start=3 + self._bigsvd):
            self._task.putbarcj(i, extr2, self._rankMatricesSmall[j] * [1.0, 2.0, 1.0])
         self._task.optimize()
         if self._task.getsolsta(mosek.soltype.itr) != mosek.solsta.optimal:
            #print("No optimal solution found in iteration {:d}".format(iterations))
            return False, bestChoi, bestRho
         self._task.getbarxj(mosek.soltype.itr, 0, bestChoi)
         self._task.getbarxj(mosek.soltype.itr, 1, bestRho)
         self._task.getbarxslice(mosek.soltype.itr, 3, 3 + self._bigsvd, 6 * self._bigsvd,
                                 np.ravel(self._rankMatricesBig))
         self._task.getbarxslice(mosek.soltype.itr, 3 + self._bigsvd,
                                 3 + self._bigsvd + self._smallsvd, 3 * self._smallsvd,
                                 np.ravel(self._rankMatricesSmall))
         rankViolation = 0
         for i, m in enumerate(self._rankMatricesBig):
            rankMatrixBig[triusBig] = m
            eVal, eVec = spla.eigh(rankMatrixBig, lower=False, check_finite=False)
            rankViolation = max(eVal[:2])
            rankViolation = max(rankViolation, rankViolation / eVal[-1])
            lowEVSys = eVec[:, :2]
            if rankViolation >= 1e-8 and rankViolation > .95 * lastRank[i]:
               # fix stall
               np.dot(lowEVSys, np.outer(.01 * np.random.rand(2), eVec[:, 2]) + lowEVSys.T,
                      out=rankMatrixBig)
            else:
               np.dot(lowEVSys, lowEVSys.T, out=rankMatrixBig)
            np.copyto(m, rankMatrixBig[triusBig])
            lastRank[i] = rankViolation
         for i, m in enumerate(self._rankMatricesSmall, start=self._bigsvd):
            rankMatrixSmall[triusSmall] = m
            eVal, eVec = spla.eigh(rankMatrixSmall, lower=False, check_finite=False)
            rankViolation = max(eVal[0], eVal[0] / eVal[-1])
            lowEV = eVec[:, 0]
            if rankViolation >= 1e-8 and rankViolation > .95 * lastRank[i]:
               # fix stall
               np.outer(lowEV, .01 * eVec[:, 1] + lowEV, out=rankMatrixSmall)
            else:
               np.outer(lowEV, lowEV, out=rankMatrixSmall)
            np.copyto(m, rankMatrixSmall[triusSmall])
            lastRank[i] = rankViolation

         thisRank = max(lastRank)
         progress = thisRank < .95 * bestRank
         if thisRank < bestRank:
            bestRank = thisRank
            if bestRank < 1e-8:
               #print("Finished in {:d} iterations with rank {:e}".format(iterations, bestRank))
               return True, bestChoi, bestRho
         if not progress and (iterations - bestRankIteration) % 50 == 0:
            #print("Canceled after {:d} iterations with rank {:e}".format(iterations, bestRank))
            return False, bestChoi, bestRho
         if thisRank < bestRank:
            bestRankIteration = iterations
      l = len(self._rankMatricesBig)
      rankMatrixBig = np.zeros((3, 3))
      rankMatrixSmall = np.zeros((2, 2))
      triusBig = np.triu_indices(3)
      triusSmall = np.triu_indices(2)
      for i, (cb, rb) in enumerate(zip(self._choiBasis, self._rhoBasis)):
         overlapC = cb @ choiVec
         overlapRho = rb @ rhoVec
         if i < l:
            rankMatrixBig[triusBig] = (overlapC * overlapC, overlapC * overlapRho, overlapC,
                                       overlapRho * overlapRho, overlapRho,
                                       1.)
            eVal, eVec = spla.eigh(rankMatrixBig, lower=False, check_finite=False)
            lowEVSys = eVec[:, :2]
            np.dot(lowEVSys, lowEVSys.T, out=rankMatrixBig)
            np.copyto(self._rankMatricesBig[i], rankMatrixBig[triusBig])
         else:
            rankMatrixSmall[triusSmall] = (.25 * (overlapC + overlapRho)**2,
                                           .5 * (overlapC + overlapRho), 1.)
            eVal, eVec = spla.eigh(rankMatrixSmall, lower=False, check_finite=False)
            lowEV = eVec[:, 0]
            np.outer(lowEV, lowEV, out=rankMatrixSmall)
            np.copyto(self._rankMatricesSmall[i - l], rankMatrixSmall[triusSmall])
