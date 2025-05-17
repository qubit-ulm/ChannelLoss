# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

import numpy as np
import scipy.optimize as spopt
import scipy.sparse as spsp
import scipy.linalg as spla
import mosek
import PatternOptimizer
from itertools import combinations, accumulate
from math import inf

class Optimizer(PatternOptimizer.Optimizer):
   def __init__(self, env, ditness, send, receive, prepPattern):
      """
         env:     MOSEK environment
         ditness: dimensionality of the carriers
         send:    number of particles to be sent
         receive: number of particles to be received
         prepPattern: list of basis states that are present in the preparation. Each entry is a
                      tuple (Alice, *send vector)
      """
      super().__init__(env, ditness, send, receive, prepPattern)

      self._rhoMatrix = np.empty((len(self.prepPattern),) *2)

      self._cosines = np.full(len(self.prepPattern), 1.)
      self._sines = np.full(len(self.prepPattern), 1.)

   def getParameterCount(self):
      return len(self.prepPattern) -1

   def __setupTask(self):
      dChois = tuple(ud['choiBasisLen'] for ud in self._uniqueData)
      dtrOuts = tuple(x // self.dStore for x in dChois)
      self._task.appendcons(1 + sum(d * (d +1) //2 for d in dtrOuts))
      self._task.appendbarvars(dChois + dtrOuts)
      self._task.putobjsense(mosek.objsense.maximize)

      self._extractions = {
         (d, i, j): self._task.appendsparsesymmat(d, [i], [j], [1.0 if i == j else 0.5])
         for d in set(dChois + dtrOuts)
         for i in range(d) for j in range(i +1)
      }

      curcon = 1
      # linear constraint: traced out map
      itrOut = len(dChois)
      for iChoi, (dChoi, dtrOut) in enumerate(zip(dChois, dtrOuts)):
         for i in range(dtrOut):
            for j in range(i +1):
               self._task.putbaraijlist([curcon] *2, [iChoi, itrOut], [0, self.dStore],
                                        [self.dStore, self.dStore +1],
                                        [self._extractions[(dChoi, i + k*dtrOut, j + k*dtrOut)]
                                         for k in range(self.dStore)] +
                                        [self._extractions[(dtrOut, i, j)]],
                                        [1.0] * (self.dStore +1))
               self._task.putconbound(curcon, mosek.boundkey.fx,
                                      0.0 if i != j else 1.0, 0.0 if i != j else 1.0)
               curcon += 1
         itrOut += 1

   def _setupDistillation(self, rhoVec):
      for i, ud in enumerate(self._uniqueData):
         d = ud['choiBasisLen']
         # fidelity objective
         distFid = ud['fidMat'].dot(rhoVec)
         self._task.putbarcj(i, [self._extractions[(d, row, col)]
                                 for col, row, f in zip(*ud['leftIndices'], distFid) if f != 0],
                             distFid[distFid != 0])
         # trace constraint
         distTr = ud['trMat'].dot(rhoVec)
         self._task.putbaraij(0, i, [self._extractions[(d, row, col)]
                                     for col, row, t in zip(*ud['leftIndices'], distTr) if t != 0],
                              distTr[distTr != 0])

   def _optimizeDistillation(self, pdist, retState=False):
      # trace constraint: trace must be fixed to the given value
      self._task.putconbound(0, mosek.boundkey.fx, pdist, pdist)
      self._task.optimize()
      solsta = self._task.getsolsta(mosek.soltype.itr)
      if solsta == mosek.solsta.optimal:
         fid = self._task.getprimalobj(mosek.soltype.itr) / pdist
         if retState:
            chois = np.empty(sum(ud['choiBasisLen'] * (ud['choiBasisLen'] +1) //2
                                 for ud in self._uniqueData))
            self._task.getbarxslice(mosek.soltype.itr, 0, len(self._uniqueData), len(chois), chois)
            return (fid, chois)
         else:
            return fid
      else:
         if retState:
            return (0, np.empty(sum(ud['choiBasisLen'] * (ud['choiBasisLen'] +1) //2
                                    for ud in self._uniqueData)))
         else:
            return 0

   def _sphericalToCartesian(self, x):
      # convert input spherical coordinates to cartesian ones
      assert len(x) == len(self.prepPattern) -1
      self._cosines[:-1] = np.cos(x)
      self._sines[1:] = np.sin(x)
      for i in range(2, len(self._sines)):
         self._sines[i] *= self._sines[i -1]
      tmp = self._sines * self._cosines
      return self._sines * self._cosines

   def _cartesianToSpherical(self, x):
      # convert input cartesian coordinates to spherical ones
      assert len(x) == len(self.prepPattern)
      x = np.array(x, dtype='float')
      x /= np.linalg.norm(x)
      norms = list(accumulate(x[::-1] **2))
      norms = np.sqrt(norms[:0:-1])
      zeroIndices = norms == 0
      nonzeroIndices = norms != 0
      last = x[-1]
      x = x[:-1]

      result = np.zeros(len(x))
      result[nonzeroIndices] = np.arccos(x[nonzeroIndices] / norms[nonzeroIndices])
      result[zeroIndices * (result < 0)] = np.pi
      if last < 0:
         result[-1] = 2*np.pi - result[-1]
      return result

   def _minimizer(self, x, pdist, retState=False, isCartesian=False):
      if not isCartesian:
         x = self._sphericalToCartesian(x)
      np.outer(x, x, out=self._rhoMatrix)

      self._setupDistillation(self._rhoMatrix[self._rightIndices])
      if retState:
         _, chois = self._optimizeDistillation(pdist, True)
         return chois
      else:
         f = self._optimizeDistillation(pdist)
         return -f

   def optimize(self, pdist, *, initialPsi=None, initialAngles=None, returnAngles=False):
      """
         pdist: distillation success probability
         initialPsi: initial state vector
         returnAngles: if True, optimal state vector will instead be in the parametric angular basis

         return: tuple(fidelity, entanglement of formation, vec of optimal Choi matrices,
                       optimal state vector)
      """
      if initialPsi is None and initialAngles is None:
         initialPsi = np.zeros(len(self.prepPattern) -1)
      elif initialPsi is None:
         initialPsi = np.array(initialAngles)
      elif initialAngles is None:
         initialPsi = self._cartesianToSpherical(initialPsi)
      else:
         assert False
      bd = ([(0, np.pi)] * (len(initialPsi) -1)) + [(0, 2*np.pi)]
      with np.errstate(divide='ignore'):
         initialPsi = spopt.minimize(self._minimizer, initialPsi, args=(pdist,),
                                     method='Powell', bounds=bd).x
         mini = spopt.minimize(self._minimizer, initialPsi, args=(pdist,),
                               method='Nelder-Mead')
      chois = self._minimizer(mini.x, pdist, True)
      psi = self._sphericalToCartesian(mini.x)
      finalState = self.getFinalState(chois, psi, mat=True)
      return (-mini.fun, self.entanglementOfFormation(finalState), chois,
              mini.x if returnAngles else psi)

   def bestSuccess(self):
      _, eof, choiVecs, initialPsi = self.optimize(1, returnAngles=True)
      stateVec = self._sphericalToCartesian(initialPsi)
      rhoVec = np.outer(stateVec, stateVec)[self._rightIndices]
      # we investigate the individual Choi states, extract the clear failure parts, calculate their
      # individual probabilities and remove them
      overlaps = [[] for _ in self._uniqueData]
      minP = 0
      idx = 0
      for i, ud in enumerate(self._uniqueData):
         d, leftIndices = ud['choiBasisLen'], ud['leftIndices']
         choi = np.empty((d, d))
         choiSucc = np.zeros_like(choi)
         choi[leftIndices] = choiVecs[idx:idx + len(leftIndices[0])]
         idx += len(leftIndices[0])
         evals, evecs = spla.eigh(choi, lower=False, overwrite_a=True, check_finite=False)
         for eval, evec in zip(evals, evecs.T):
            if eval < 1e-7:
               continue
            evec[abs(evec) < 1e-8] = 0.0
            emat = np.outer(evec, evec)
            # is this eigenvector entangled? -> success, else failure
            if -sum(x * np.log2(x) if x > 1e-10 else 0
                    for x in spla.eigh(np.einsum("abcb",
                                       emat.reshape((self.dStore, len(evec) // self.dStore) *2)),
                                       eigvals_only=True, overwrite_a=True)) < 1e-5:
               overlaps[i].append([emat[i, i] if i == j else 2*emat[i, j]
                                   for i, j in zip(*leftIndices)])
            else:
               minP += eval * (emat[leftIndices] @ ud['trMat'] @ rhoVec)
      # now make the Chois always orthogonal
      curcon = self._task.getnumcon()
      newcons = sum(map(len, overlaps))
      self._task.appendcons(newcons)
      self._task.putconboundsliceconst(curcon, curcon + newcons, mosek.boundkey.fx, 0.0, 0.0)
      for mat, (o, ud) in enumerate(zip(overlaps, self._uniqueData)):
         for overlap in o:
            self._task.putbaraij(curcon, mat, [self._extractions[(ud['choiBasisLen'], row, col)]
                                               for col, row in zip(*ud['leftIndices'])],
                                 overlap)
            curcon += 1
      # iteratively try to find the new probability - we know what it must be at least!
      pLow, pHigh = minP, 1
      _, eof, choiVecs, initialPsi = self.optimize(pLow, initialPsi=initialPsi, returnAngles=True)
      while pHigh - pLow > 1e-5:
         curP = (pHigh + pLow) / 2
         _, newEof, newChoiVecs, newPsi = self.optimize(curP, initialPsi=initialPsi,
                                                        returnAngles=True)
         if newEof < eof:
            pHigh = curP
         else:
            pLow = curP
            eof, choiVec, initialPsi = newEof, newChoiVecs, newPsi
      return (eof, pLow, choiVecs, self._sphericalToCartesian(initialPsi))

   def toMathematica(self):
      import io
      output = io.StringIO()
      print("Block[{choiMats, choiVecs, choi, pdist, conds=True, \\[Psi]Vec, \\[Psi], \\[Rho]Vec, dist"
            "Fid, distTr},", file=output)
      print("  choiMats = With[{{lens = {{{}}}}}, Table[choi[i, Min[j, k], Max[j, k]]"
            ", {{i, {}}}, {{j, lens[[i]]}}, {{k, lens[[i]]}}]];".format(
               str([ud['choiBasisLen'] for ud in self._uniqueData])[1:-1], len(self._uniqueData)
            ), file=output)
      print("  choiVecs = getStorage/@choiMats;", file=output)
      for i, ud in enumerate(self._uniqueData, start=1):
         print('  conds = conds && VectorGreaterEqual[{{choiMats[[{}]], 0}}, "SemidefiniteCone"] &&'
               ' VectorLessEqual[{{TensorContract[Reshape[choiMats[[{}]], {{2, -1, 2, -1}}], {{1, 3'
               '}}], IdentityMatrix[{}]}}, "SemidefiniteCone"];'.format(i, i,
                                                                  ud['choiBasisLen']//self.dStore),
               file=output)
      print("  \\[Psi]Vec = Array[\\[Psi], {}];".format(len(self.prepPattern)), file=output)
      print("  \\[Rho]Vec = getStorage[Outer[Times, \\[Psi]Vec, \\[Psi]Vec]];", file=output)
      print("  distFid = {", file=output)
      for ud in self._uniqueData:
         fidMat = ud['fidMat']
         print("    SparseArray[Automatic, {{{}, {}}}, 0, {{1, {{{{{}}}, Transpose[{{{{{}}}}}] +1}}"
               ", {{{}}}}}].\\[Rho]Vec, ".format(*fidMat.shape,
                                                str(fidMat.indptr.tolist())[1:-1],
                                                str(fidMat.indices.tolist())[1:-1],
                                                str(fidMat.data.tolist())[1:-1]), file=output)
      print("    Nothing};", file=output)
      print("  distTr = {", file=output)
      for ud in self._uniqueData:
         trMat = ud['trMat']
         print("    SparseArray[Automatic, {{{}, {}}}, 0, {{1, {{{{{}}}, Transpose[{{{{{}}}}}] +1}}"
               ", {{{}}}}}].\\[Rho]Vec, ".format(*trMat.shape,
                                                str(trMat.indptr.tolist())[1:-1],
                                                str(trMat.indices.tolist())[1:-1],
                                                str(trMat.data.tolist())[1:-1]), file=output)
      print("    Nothing};", file=output)
      print("  ParametricConvexOptimization[", file=output)
      print("    -Flatten[distFid].Flatten[choiVecs],", file=output)
      print("    conds && Flatten[distTr].Flatten[choiVecs] == pdist,", file=output)
      print("    Reduce`FreeVariables[choiMats], Prepend[\\[Psi]Vec, pdist],", file=output)
      print('    "PrimalMinimumValue"', file=output)
      print("  ]", file=output)
      print("]", file=output, end='')
      content = output.getvalue()
      output.close()
      return content