# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

import numpy as np
import scipy.sparse as spsp
import scipy.linalg as spla
import mosek
import collections
import PatternOptimizer
from itertools import product, chain
from math import inf

class Optimizer(PatternOptimizer.Optimizer):
   def __setupTask(self):
      d = len(self.prepPattern)
      self._task.appendcons(2)
      self._task.appendbarvars([d])
      self._task.putobjsense(mosek.objsense.maximize)

      self._extractions = {
         (i, j): self._task.appendsparsesymmat(d, [i], [j], [1.0 if i == j else 0.5])
         for i in range(d) for j in range(i +1)
      }

      # linear constraint: trace
      self._task.putbaraij(0, 0, [self._extractions[(i, i)] for i in range(d)], [1.0] * d)
      self._task.putconbound(0, mosek.boundkey.up, -inf, 1.0)

   def _setupDistillation(self, maps):
      assert set(maps.keys()) == set(self._slots)
      distFid = np.zeros(len(self._rightIndices[0]))
      distTr = np.zeros_like(distFid)
      self._mapVecs = {}
      for slot, choi in maps.items():
         sd = self._slotData[slot]
         ud = self._uniqueData[sd['uniqueIdx']]
         inBasis = sd['inBasis']
         leftIndices = ud['leftIndices']
         # the map entries themselves are given as dictionaries that map certain basis elements to
         # their respective values; we must match this with our basis structure
         if __debug__ or isinstance(choi, dict):
            fullBasis = tuple((output, *input) for output in range(self.dStore) for input in inBasis)
         if isinstance(choi, dict):
            mapVec = np.fromiter((choi.get((fullBasis[row], fullBasis[col]), 0)
                                  for row, col in zip(*leftIndices)), 'float64',
                                 count=len(leftIndices[0]))
         else:
            # for output that directly comes from parseChoiVec (with mat=False, full=False)
            mapVec = choi.copy()
         # check whether we are good...
         if __debug__:
            testchoi = np.empty((len(fullBasis),)*2)
            testchoi[leftIndices] = mapVec
            assert spla.eigh(testchoi, lower=False, check_finite=False, eigvals_only=True)[0] > -1e-10
            testchoi.T[leftIndices] = mapVec
            maxEig = spla.eigh(np.einsum("abac",
                                       testchoi.reshape((self.dStore, testchoi.shape[0] //2) *2)),
                             eigvals_only=True)[-1]
            if maxEig > 1 + 1e-7:
               print("maximal eigenvalue:", maxEig, self.prepPattern)
               assert False
         # note that fidMat and trMat already include the multiplicities if they occur multiple
         # times, but here, we iterate through each single slot - so we need to undo the
         # pre-multiplication!
         # fidelity objective, trace output
         distFid += mapVec @ ud['fidMat'] / ud['multiplicity']
         distTr += mapVec @ ud['trMat'] / ud['multiplicity']
         self._mapVecs[slot] = mapVec
      self._task.putbarcj(0, [self._extractions[(row, col)]
                              for col, row, f in zip(*self._rightIndices, distFid) if f != 0],
                          distFid[distFid != 0])
      self._distTr = distTr
      self._task.putbaraij(1, 0, [self._extractions[(i, j)] for j, i in zip(*self._rightIndices)],
                           self._distTr)

   def _optimizeDistillation(self, pdist):
      if pdist is None:
         self._task.putconbound(1, mosek.boundkey.fr, -inf, inf)
      else:
         self._task.putconbound(1, mosek.boundkey.fx, pdist, pdist)
      self._task.optimize()
      solsta = self._task.getsolsta(mosek.soltype.itr)
      if solsta == mosek.solsta.optimal:
         rhoVec = np.empty(len(self._rightIndices[0]))
         self._task.getbarxj(mosek.soltype.itr, 0, rhoVec)
         pdist = self._distTr @ rhoVec
         if pdist > 1e-8:
            fid = self._task.getprimalobj(mosek.soltype.itr) / pdist
         else:
            fid = 0
         return (fid, pdist, rhoVec)
      else:
         return (0, 0, np.empty(len(self._rightIndices[0])))

   def optimize(self, pdist, maps):
      """
         pdist: distillation success probability
         maps: {(tuple of arriving slots): {(basis element [triu sufficient]): value}}

         return: tuple(fidelity, entanglement of formation, success probability,
                       optimal state vector)
      """
      self._setupDistillation(maps)
      fid, pdist, rhoVec = self._optimizeDistillation(pdist)
      rho = np.empty((len(self.prepPattern),) *2)
      rho[self._rightIndices] = rhoVec
      eVal, eVec = spla.eigh(rho, lower=False, check_finite=False)
      finalState = self.getFinalState(self._mapVecs, eVec[:, -1], mat=True)
      return (fid, self.entanglementOfFormation(finalState), pdist, eVec[:, -1])

   def guessOptimalMap(self):
      """
         Based on empiric observations, we provide a means that constructs a good initial guess for
         distillation maps (trace non-increasing, will only contain parts that successfully
         distill) given a certain ket basis.
         This can be in turn fed to optimize().

         return: list of {(tuple of arriving slots): {(basis element [triu sufficient]): value}}

         Note that since sometimes, there are multiple good initial candidates, we must return a
         list of all those.
      """
      assert self.dStore == 2
      def conditionsContradictory(conditions):
         """
            conditions is a list of 2-tuples of elements, where the first element is supposed to be
            strictly greater than the second; we check whether this is consistent with an arbitrary
            partial order.
         """
         greaterThan = {}
         for i, j in conditions:
            if i in greaterThan:
               greaterThan[i].add(j)
            else:
               greaterThan[i] = {j}

         while True:
            changes = False
            for i, s in greaterThan.items():
               additions = set()
               for j in s:
                  if j == i:
                     return True
                  else:
                     additions |= greaterThan.get(j, set())
               additions -= s
               if additions:
                  s |= additions
                  changes = True
            if not changes:
               return False

      def mapsToEigenvectors(mapList):
         conditions = set()
         for i in range(len(mapList) -1):
            # How does a ket pattern have to look like so that it is in contradiction with the
            # current one? It must have the same receival part, but a different store part
            contradictions = frozenset((1 - item[0], *item[1:]) for item in mapList[i].keys())
            # Now check if these contradictions are present.
            for j in range(i +1, len(mapList)):
               for issue in contradictions & mapList[j].keys():
                  # We can resolve a contradiction by putting a condition that makes one coefficient
                  # larger than another, which will eliminate the smaller one.
                  resolution = (mapList[i][(1 - issue[0], *issue[1:])], mapList[j][issue])
                  if resolution[1] < resolution[0]:
                     resolution = (resolution[1], resolution[0])
                  conditions.add(resolution)
         # We now have all possible conditions and must put them together
         newMapLists = {}
         for conditionTuple in product(*((cond, (cond[1], cond[0])) for cond in conditions)):
            if conditionsContradictory(conditionTuple):
               continue
            drops = {c[1] for c in conditionTuple}
            # Now we merge entries in the map list. This will potentially create duplicate keys that
            # we need to retain - so convert the dictionaries into lists.
            maps = [list(dictionary.items()) for dictionary in mapList
                                             if not (drops & frozenset(dictionary.values()))]
            if not maps:
               continue
            i = 0
            while i < len(maps):
               j = i +1
               while j < len(maps):
                  if {item[0] for item in maps[i]} & {item[0] for item in maps[j]}:
                     maps[i] += maps[j]
                     del maps[j]
                     j = i +1
                  else:
                     j += 1
               i += 1
            maps = tuple(sorted(tuple(sorted([item[0] for item in items])) for items in maps))
            if maps in newMapLists:
               newMapLists[maps].append(set(conditionTuple))
            else:
               newMapLists[maps] = [set(conditionTuple)]
         # We now know all possible resolutions, but there may be duplicates, in particular if one
         # condition makes another redundant. We try to find these redundancies and remove them; if
         # there are further duplicates, we yield all of them since they may make a difference when
         # combined with other slots.
         for maps, conditionSets in newMapLists.items():
            while len(conditionSets) > 1:
               for conditionSet in conditionSets:
                  for cond in conditionSet:
                     newCond = conditionSet.copy()
                     newCond.remove(cond)
                     newCond.add((cond[1], cond[0]))
                     try:
                        j = conditionSets.index(newCond)
                     except ValueError:
                        continue
                     # Since we get the same result regardless of whether the condition is in the one
                     # direction or the other, just drop it.
                     conditionSet.remove(cond)
                     del conditionSets[j]
                     break
                  else:
                     continue
                  break
               else:
                  # If we were not able to drop any of the items by redundancy-removal, we must keep
                  # them.
                  break
            # Now we merge entries in the map list. This will potentially create duplicate keys that
            # we need to retain - so convert the dictionaries into lists.
            for conditionSet in conditionSets:
               yield (list(maps), frozenset(conditionSet))

      def buildChoi(eigenvectors):
         # turn the list of vectors into a state, deriving the proper coefficients
         choi = {}
         for eigenvector in eigenvectors:
            weights = [(x, np.sqrt(count / len(eigenvector)))
                       for x, count in collections.Counter(eigenvector).items()]
            for t1, t2 in product(weights, repeat=2):
               key = (t1[0], t2[0])
               val = 2 * t1[1] * t2[1]
               if key in choi:
                  choi[key] += val
               else:
                  choi[key] = val
         return {k: v for k, v in sorted(choi.items())}

      def patchTogether(current, previous={}, conditions=set()):
         previous = previous.copy()
         if not current:
            yield previous
         else:
            key, items = current.popitem()
            for choi, choiConditions in items:
               newCon = conditions | choiConditions
               if not conditionsContradictory(newCon):
                  previous[key] = choi
                  yield from patchTogether(current.copy(), previous, newCon)

      optimals = {}
      for slot, lossAssoc in self._getLossMaps():
         # Next, we build a dictionary based on the diagonal entries that for every origin tells us
         # which arrival pattern it refers to
         individualMaps = {x: p1 for (p1, p2), origin in lossAssoc.items() if p1 == p2
                                 for x, y in origin if x == y}
         # Build 2-tuples of all terms where an Alice-0 is combined with an Alice-1
         maps = [{(0, *individualMaps[x]): x, (1, *individualMaps[y]): y}
                  for origin in lossAssoc.values()
                  for x, y in origin if self.prepPattern[x][0] != self.prepPattern[y][0]]
         if not maps:
            optimals[slot] = [({}, frozenset())]
         else:
            optimals[slot] = [(buildChoi(eigenvectors), condition)
                              for eigenvectors, condition in mapsToEigenvectors(maps)]
         # maybe an assert on the validity of the Choi state would be in order...

      yield from patchTogether(optimals)

o = Optimizer(mosek.Env(), 2, 3, 2, [(0,0,0,0),(0,0,0,1),(0,1,1,0),(0,1,1,1),(1,0,1,0),(1,1,0,1)])
list(o.guessOptimalMap())