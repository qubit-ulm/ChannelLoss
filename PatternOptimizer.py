# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

import numpy as np
import scipy.sparse as spsp
import GenericOptimizer
import operator
import itertools

# We need an itemgetter that, unlike the one from operator, always returns a tuple.
def itemgetter(*items):
   if len(items) > 1:
      return operator.itemgetter(*items) # far more efficient
   else:
      def g(obj):
         return tuple(obj[item] for item in items)
      return g

class Optimizer(GenericOptimizer.Optimizer):
   def __init__(self, env, ditness, send, receive, prepPattern):
      """
         env:     MOSEK environment
         ditness: dimensionality of the carriers
         send:    number of particles to be sent
         receive: number of particles to be received
         prepPattern: list of basis states that are present in the preparation. Each entry is a
                      tuple (Alice, *send vector)
      """
      assert ditness >= 2
      assert send > receive
      self.prepPattern = list(sorted(prepPattern))
      self.ditness = ditness
      self.dStore = max(x[0] for x in prepPattern) +1
      self.send = send
      self.receive = receive
      self._slots = list(itertools.combinations(range(1, send +1), receive))

      self.__setupMatrices()
      super().__init__(env)

   def __setupMatrices(self):
      # construct the loss map: we input the vectorized upper triangle of rho, expressed in the
      # prepPattern basis; we output the vectorized upper triangle of the traced out rho, expressed
      # in the particular loss basis that will be constructed on-the-fly
      
      # For every slot combination, we need quite some information
      self._slotData = {s: None for s in self._slots}
      # However, for the SDPs this would be wasteful since lots of the relevant maps actually act in
      # the same way in matrix form.
      self._uniqueData = [None] * len(self._slots)
      
      self._rightIndices = np.triu_indices(len(self.prepPattern))
      triuLarge = {k: v for v, k in enumerate(zip(*self._rightIndices))}
      uniqueIdx = 0
      for slot in self._slots:
         lossMap = {}
         lose = itemgetter(0, *slot)
         lost = itemgetter(*sorted(set(range(1, self.send +1)) - set(slot)))
         # first build the complete basis
         for j, largeKet in enumerate(self.prepPattern):
            smallKet = lose(largeKet)
            if smallKet in lossMap:
               lossMap[smallKet].append(j)
            else:
               lossMap[smallKet] = [j]
         lossMap = dict(sorted(lossMap.items()))
         triuSmall = np.triu_indices(len(lossMap))
         lossBasis = tuple(lossMap.keys())
         inBasis = tuple(sorted(set(map(lambda b: b[1:], lossBasis))))
         lossOrigin = tuple(lossMap.values())

         # construct the loss map
         # TODO: this is inefficient, as this representation first creates a COO matrix. Can we
         # directly get CSR?
         csrData = np.asarray(tuple(
            (j, triuLarge[(min(largeIndexX, largeIndexY), max(largeIndexX, largeIndexY))])
            for j, (x, y) in enumerate(zip(*triuSmall))
            for largeIndexX in lossOrigin[x]
            for largeIndexY in lossOrigin[y]
            if lost(self.prepPattern[largeIndexX]) == lost(self.prepPattern[largeIndexY])
         ), dtype='int')
         lossMat = spsp.csr_matrix(([1.0] * len(csrData), (csrData[:, 0], csrData[:, 1])),
                                   shape=(len(triuSmall[0]), len(triuLarge)))
         lossMat.eliminate_zeros()
         lossMat.sum_duplicates()

         leftIndices = np.triu_indices(self.dStore * len(inBasis))
         choiBasis = [(j, k) for j in range(self.dStore) for k in inBasis]
         # we also want to get the maps for the final state
         liMap = {x: j for j, x in enumerate(zip(*leftIndices))}
         riMap = {x: j for j, x in enumerate(zip(*triuSmall))}
         triuOut = np.triu_indices(self.dStore **2)
         csrData = np.asarray(tuple(
            (type, liMap[(min(leftRow, leftCol), max(leftRow, leftCol))],
             riMap[(min(rightRow, rightCol), max(rightRow, rightCol))], 1)
            for leftRow in range(self.dStore * len(inBasis))
            for leftCol in range(self.dStore * len(inBasis))
            for rightRow in range(len(lossBasis))
            for rightCol in range(len(lossBasis))
            for type, (outI, outJ) in enumerate(zip(*triuOut))
            if lossBasis[rightRow][1:] == choiBasis[leftRow][1] and
               lossBasis[rightCol][1:] == choiBasis[leftCol][1] and
               (lossBasis[rightRow][0], choiBasis[leftRow][0]) == divmod(outI, self.dStore) and
               (lossBasis[rightCol][0], choiBasis[leftCol][0]) == divmod(outJ, self.dStore)
         ), dtype='int')
         resMats = [None] * len(triuOut[0])
         for j in range(len(resMats)):
            csrMat = csrData[csrData[:, 0] == j]
            resMats[j] = spsp.csr_matrix((csrMat[:, 3], (csrMat[:, 1], csrMat[:, 2])),
                                         shape=(len(leftIndices[0]), len(triuSmall[0]))).\
                         dot(lossMat)
            resMats[j].eliminate_zeros()
            resMats[j].sum_duplicates()
         trMat = resMats[0] + sum(resMats[i]
                                  for i in itertools.accumulate(range(self.dStore **2, 1, -1)))
         trMat.eliminate_zeros()
         trMat.sum_duplicates()
         # The maximally entangled state vector in d dimensions has entries at the positions
         # i(d +1), i = 0, ..., d -1. Its projector has entries at {i(d +1), j(d +1)}, so the
         # vectorized upper triangle at (1+d) (2 d^2 i+2 j-i-(1+d) i^2)/2
         # We don't normalize the state here, so that we keep an exact integer dtype
         fidMat = sum(resMats[x] if i == j else 2*resMats[x] for i in range(self.dStore)
                      for j in range(i, self.dStore)
                      for x in ((1 + self.dStore) * (2 * self.dStore**2 * i + j + j -
                                                     i * (1 + (1 + self.dStore) * i)) //2,))
         fidMat.eliminate_zeros()
         fidMat.sum_duplicates()

         # check for duplicates (allows reduction of number of semidefinite variables, so gives a
         # speed up that is worth checking)
         dup = False
         for dupIdx, dupData in enumerate(self._uniqueData[:uniqueIdx]):
            # in principle, the nnz check is unnecessary, as this is already implied in the
            # other checks, but we do it before we perform the array checks
            # note the equality checks are unproblematic, since all the arrays (including data) have
            # dtype int.
            if all(m1.shape == m2.shape and m1.nnz == m2.nnz and (m1.data == m2.data).all() and
                   (m1.indices == m2.indices).all() and (m1.indptr == m2.indptr).all()
                   for m1, m2 in ((trMat, dupData['trMat']), (fidMat, dupData['fidMat']))):
               dupData['multiplicity'] += 1
               dup = True
               break
         self._slotData[slot] = {
            # the full incoming basis, including Alice
            'lossBasis': lossBasis,
            # the CSR matrix mapping the full state (vectorized) into what arrives (vectorized)
            'lossMat': lossMat,
            # the basis for the incoming vector space of the map
            'inBasis': inBasis,
            # the complete basis for the choi map (outgoing + incoming)
            'choiBasis': choiBasis,
            # a tuple, where each item is a sparse matrix M such that choiVec @ M @ rhoFullVec gives
            # an entry in the vectorized (dStore**2)x(dStore**2) output matrix
            'resMats': resMats,
            # the index that contains the relevant fidelity and trace matrix in their properties
            'uniqueIdx': dupIdx if dup else uniqueIdx
         }
         if not dup:
            self._uniqueData[uniqueIdx] = {
               'trMat': trMat,
               'fidMat': fidMat,
               'leftIndices': leftIndices,
               'multiplicity': 1,
               'choiBasisLen': len(choiBasis)
            }
            uniqueIdx += 1
      # remove all the unnecessary items from the arrays
      if uniqueIdx < len(self._slots):
         self._uniqueData = self._uniqueData[:uniqueIdx]
      # re-scale the matrices appropriately
      frac = 1 / len(self._slots)
      fidNorm = 1 / self.dStore
      for ud in self._uniqueData:
         ud['trMat'] *= frac * ud['multiplicity']
         ud['fidMat'] *= fidNorm * frac * ud['multiplicity']

   def parseChoiVec(self, vec, *, mat=False, full=False):
      """
         vec: Choi vector as returned by the second return argument of optimize
         mat: if False, return vectorized upper triangles; if True, return full matrices
         full: if False, return result in the basis that can be queried by getChoiBasis; if True,
               return result in the full basis, (dStore * ditness**receive)-dimensional

         returns: dictionary that gives for every combination of arrived slots the Choi matrices or
                  vectors, corresponding to the given options
      """
      startIndices = [0] + list(itertools.accumulate(ud['choiBasisLen'] * (ud['choiBasisLen'] +1) //2
                                                     for ud in self._uniqueData))
      if mat:
         if full:
            fullBasis = [(i, *j) for i in range(self.dStore)
                         for j in itertools.product(range(self.ditness), repeat=self.receive)]
            result = {slot: np.zeros((len(fullBasis),) *2) for slot in self._slotData}
            for res, sd in zip(result.values(), self._slotData.values()):
               matIdx = sd['uniqueIdx']
               redBasis = sd['inBasis']
               smallIndices = self._uniqueData[matIdx]['leftIndices']
               startIdx = startIndices[matIdx]
               for idx, (iSmall, jSmall) in enumerate(zip(*smallIndices)):
                  q, r = divmod(iSmall, len(redBasis))
                  iLarge = fullBasis.index((q, *redBasis[r]))
                  q, r = divmod(jSmall, len(redBasis))
                  jLarge = fullBasis.index((q, *redBasis[r]))
                  res[iLarge, jLarge] = vec[startIdx + idx]
                  res[jLarge, iLarge] = vec[startIdx + idx]
            return result
         else:
            result = {}
            for slot, sd in self._slotData.items():
               matIdx = sd['uniqueIdx']
               smallIndices = self._uniqueData[matIdx]['leftIndices']
               startIdx = startIndices[matIdx]
               result[slot] = np.empty((len(sd['choiBasis']),) *2)
               result[slot][smallIndices] = vec[startIdx:startIdx + len(smallIndices[0])]
               result[slot].T[smallIndices] = vec[startIdx:startIdx + len(smallIndices[0])]
            return result
      else:
         if full:
            fullBasis = [(i, *j) for i in range(self.dStore)
                         for j in itertools.product(range(self.ditness), repeat=self.receive)]
            dim = len(fullBasis)
            result = {slot: np.zeros(dim * (dim +1) //2) for slot in self._slotData}
            for res, sd in zip(result.values(), self._slotData.values()):
               matIdx = sd['uniqueIdx']
               redBasis = sd['inBasis']
               smallIndices = self._uniqueData[matIdx]['leftIndices']
               startIdx = startIndices[matIdx]
               for idx, (iSmall, jSmall) in enumerate(zip(*smallIndices), start=startIdx):
                  q, r = divmod(iSmall, len(redBasis))
                  iLarge = fullBasis.index((q, *redBasis[r]))
                  q, r = divmod(jSmall, len(redBasis))
                  jLarge = fullBasis.index((q, *redBasis[r]))
                  res[jLarge - iLarge + (1 + 2*dim - iLarge) * iLarge //2] = vec[idx]
            return result
         else:
            result = {}
            for slot, sd in self._slotData.items():
               matIdx = sd['uniqueIdx']
               startIdx = startIndices[matIdx]
               result[slot] = vec[startIdx:startIdx +
                                           len(self._uniqueData[matIdx]['leftIndices'][0])]
            return result

   def getChoiBasis(self, slot):
      """
         slot: index of the corresponding slot arrival

         returns: list that corresponds to all kets in the basis for this slot's Choi matrix
      """
      return self._slotData[slot]['choiBasis']

   def getFinalState(self, choiVec, psi, mat=False):
      """
         returns the final state based on the results of an optimization routine
         choiVec: a continuous array containing the vectorized upper triangles of the Choi matrices,
                  where duplicates (in terms of p and F) are already dropped
                  _or_ a dictionary that maps to every slot the appropriate vectorized upper
                  triangle of the Choi matrix, including all duplicates (in terms of p and F)
         mat: if False, returns vectorized upper triangle, else returns full matrix
      """
      result = np.zeros(self.dStore**2 * (self.dStore**2 +1) //2)
      rhoVec = np.outer(psi, psi)[self._rightIndices]
      complete = isinstance(choiVec, dict)
      if not complete:
         startIndices = [0] + list(itertools.accumulate(ud['choiBasisLen'] *
                                                        (ud['choiBasisLen'] +1) //2
                                                        for ud in self._uniqueData))
      for i in range(len(result)):
         for slot, sd in self._slotData.items():
            if complete:
               result[i] += choiVec[slot] @ sd['resMats'][i] @ rhoVec
            else:
               result[i] += choiVec[startIndices[sd['uniqueIdx']]:
                                    startIndices[sd['uniqueIdx'] +1]] @ sd['resMats'][i] @ rhoVec
      result /= len(self._slots) # normalize
      if mat:
         res = np.empty((self.dStore**2,) *2)
         ind = np.triu_indices(res.shape[0])
         res[ind] = result
         res.T[ind] = result
         return res
      else:
         return result
   
   def _getLossMaps(self):
      for slot, sd in self._slotData.items():
         # The loss maps are in normalized CSR form - directly use them to obtain the contributions 
         # to the final states
         lossAssoc = {}
         leftIndices = np.triu_indices(len(sd['lossBasis']))
         for row, (colStart, colEnd) in enumerate(zip(sd['lossMat'].indptr,
                                                      sd['lossMat'].indptr[1:])):
            if colStart == colEnd:
               continue
            inI, inJ = leftIndices[0][row], leftIndices[1][row]
            key = (sd['lossBasis'][inI][1:], sd['lossBasis'][inJ][1:])
            if key[1] < key[0]:
               key = (key[1], key[0])
            if key not in lossAssoc:
               lossAssoc[key] = []
            lossAssoc[key] += [(self._rightIndices[0][col], self._rightIndices[1][col])
                               for col, val in zip(sd['lossMat'].indices[colStart:colEnd],
                                                   sd['lossMat'].data[colStart:colEnd])]
         yield (slot, lossAssoc)
   
   def redundancyHeuristic(self):
      """
         Checks whether the particular choice of subspace contains unhelpful degrees of freedom,
         i.e., coefficients that will be zero in the best configuration.
         If this function return False, the pattern may still be redundant. If it returns True, it is
         guaranteed to be so.
         This function is very cheap to run.
      """
      coefficients = set(range(len(self.prepPattern)))
      for _, lossAssoc in self._getLossMaps():
         for origin in lossAssoc.values():
            for x, y in origin:
               if self.prepPattern[x][0] != self.prepPattern[y][0]:
                  coefficients -= {x, y}
      return not not coefficients
   
   def uniqueIdentifierHeuristic(self):
      """
         Transforms the current state into a canonical form by employing permutations of coefficients
         and slots. Returns a unique identifier of the canonical form (hashable), or False.
         Two optimizers with the same canonical form identifier are guaranteed to give the same
         output fidelity and are unitarily related.
         If two optimizers have different canonical form identifiers, their optimizations may still
         give the same fidelities.
         Note that this function will return False if redundancyHeuristic would yield True.
         This function is moderately cheap to run.
      """
      alphas, betas = set(), set()
      # First, we sum over the Alice side (not trace, but sum - we don't need the distinction which
      # Alice's bit was here, as we can keep track of this via the indices alone). This helps
      # reducing the number of permutations later.
      lossAssocs = {}
      allTranspositions = {}
      for slot, sd in self._slotData.items():
         onesidedBasis = {x[1:] for x in sd['lossBasis']}
         onesidedBasis = {k: v for k, v in zip(onesidedBasis, range(len(onesidedBasis)))}
         lossAssoc = {}
         # The loss maps are in normalized CSR form - directly use them to obtain the contributions 
         # to the final states
         leftIndices = np.triu_indices(len(sd['lossBasis']))
         for row, (colStart, colEnd) in enumerate(zip(sd['lossMat'].indptr,
                                                      sd['lossMat'].indptr[1:])):
            if colStart == colEnd:
               continue
            inI, inJ = leftIndices[0][row], leftIndices[1][row]
            key = (onesidedBasis[sd['lossBasis'][inI][1:]], onesidedBasis[sd['lossBasis'][inJ][1:]])
            if key[1] < key[0]:
               key = (key[1], key[0])
            key2 = (key[1], key[0]) # we need both for the transpositions
            new = [(self._rightIndices[0][col], self._rightIndices[1][col])
                   for col, val in zip(sd['lossMat'].indices[colStart:colEnd],
                                       sd['lossMat'].data[colStart:colEnd])]
            if key not in lossAssoc:
               lossAssoc[key] = new
            else:
               lossAssoc[key] += new
            if key2 not in lossAssoc:
               lossAssoc[key2] = lossAssoc[key] # by reference, so we only have to take care of one
         worthwhile = False
         for k, origin in lossAssoc.items():
            if k[0] <= k[1]: # the others are by reference
               origin.sort()
            for x, y in origin:
               if self.prepPattern[x][0] == 0 and self.prepPattern[y][0] == 1:
                  alphas.add(x)
                  betas.add(y)
                  worthwhile = True
               elif self.prepPattern[x][0] == 1 and self.prepPattern[y][0] == 0:
                  alphas.add(y)
                  betas.add(x)
                  worthwhile = True
         if worthwhile:
            if len(onesidedBasis) not in allTranspositions:
               allTranspositions[len(onesidedBasis)] = \
                  list(itertools.permutations(range(len(onesidedBasis))))
            lossAssocs[slot] = (lossAssoc, allTranspositions[len(onesidedBasis)])
      if len(alphas) + len(betas) < len(self.prepPattern):
         return False
      assert len(betas) <= len(alphas)
      
      # assign comparison candidate
      canonicalCandidate = None
      permCandidate = [None] * len(lossAssocs)
      for perms in itertools.product(itertools.permutations(alphas), itertools.permutations(betas)):
         for perm in (((*perms[0], *perms[1]),) if len(alphas) > len(betas) else
                      ((*perms[0], *perms[1]), (*perms[1], *perms[0]))):
            for iAssoc, (lossAssoc, transpositions) in enumerate(lossAssocs.values()):
               currentTransSecondHalf = [
                  tuple(sorted((perm[originX], perm[originY]) if perm[originX] < perm[originY] else
                               (perm[originY], perm[originX]) for originX, originY in origins))
                  for origins in lossAssoc.values()
               ]
               compareAgainst = None
               for transposition in transpositions:
                  currentTrans = [
                     ((transposition[recvI], transposition[recvJ]), sh)
                     for (recvI, recvJ), sh in zip(lossAssoc.keys(), currentTransSecondHalf)
                  ]
                  currentTrans.sort()
                  if compareAgainst is None or currentTrans < compareAgainst:
                     compareAgainst = currentTrans
               permCandidate[iAssoc] = compareAgainst
         permCandidate.sort()
         if canonicalCandidate is None or permCandidate < canonicalCandidate:
            canonicalCandidate = permCandidate.copy()
      
      # make everything hashable
      canonicalCandidate = tuple(tuple(tuple(z for z in y) for y in x) for x in canonicalCandidate)
      return (len(alphas), len(betas), canonicalCandidate)
   
   @staticmethod
   def equivalenceHeuristic(optA, optB):
      """
         Checks whether we can for sure say that two prepPatterns are equivalent (this does not check
         for simple equivalence, but instead goes for the final states).
         If this function returns False, the patterns may still be equivalent. If it returns True,
         they are guaranteed to be inequivalent.
         This function is very expensive to run and potentially worse than uniqueIdentifierHeuristic.
         Do not use it.
      """
      assert optA.ditness == optB.ditness
      assert optA.send == optB.send
      assert optA.receive == optB.receive
      assert len(optA.prepPattern) == len(optB.prepPattern)
      if len(optA._uniqueData) != len(optB._uniqueData):
         return False
      optAalphas = sum(1 for x in optA.prepPattern if x[0] == 0)
      optBalphas = sum(1 for x in optB.prepPattern if x[0] == 0)
      if optAalphas == optBalphas:
         inverted = False
      elif optAalphas == len(optB.prepPattern) - optBalphas:
         inverted = True
      else:
         return False
      groupings = {(x['multiplicity'], x['choiBasisLen']) for x in optA._uniqueData}
      groupings = [(tuple(ud for ud in optA._uniqueData if ud['multiplicity'] == group[0] and
                                                           ud['choiBasisLen'] == group[1]),
                    tuple(ud for ud in optB._uniqueData if ud['multiplicity'] == group[0] and
                                                           ud['choiBasisLen'] == group[1]))
                   for group in groupings]
      for ga, gb in groupings:
         if len(ga) != len(gb):
            return False
      for symm in range(2 if not inverted and optAalphas == len(optB.prepPattern) - optBalphas
                          else 1): # if #alpha = #beta, check for inversion or non-inversion
         # we need to obtain the mappings to the final state, where we will only care about those
         # that may give any coherent contribution at all; and then, we are not interested in the
         # particular patterns, only in what we can discriminate against (globally) and how the
         # coefficients occur
         lossMapsA = []
         lossMapsB = []
         alphaIndices = set()
         betaIndices = set()
         for opt, lossMaps, invert in ((optA, lossMapsA, False), (optB, lossMapsB, inverted)):
            for _, lossAssoc in opt._getLossMaps():
               data = []
               hasCoherence = False
               for origin in lossAssoc.values():
                  classification = (set(), set(), set(), set(), set())
                                    # alpha^2, beta^2, alpha alpha, beta beta, alpha beta
                  for x, y in origin:
                     if opt.prepPattern[x][0] == 0:
                        if opt.prepPattern[y][0] == 0:
                           classification[0 if opt.prepPattern[x] == opt.prepPattern[y]
                                            else 2].add((x, y))
                           if invert:
                              betaIndices.add(x)
                           else:
                              alphaIndices.add(x)
                        else:
                           classification[4].add((x, y))
                           hasCoherence = True
                     elif opt.prepPattern[y][0] == 0:
                        classification[4].add((x, y))
                        hasCoherence = True
                     else:
                        classification[1 if opt.prepPattern[x] == opt.prepPattern[y] else 3].add((x, y))
                        if invert:
                           alphaIndices.add(x)
                        else:
                           betaIndices.add(x)
                  if invert:
                     data.append((frozenset(classification[1]), frozenset(classification[0]),
                                  frozenset(classification[3]), frozenset(classification[2]),
                                  frozenset(classification[4])))
                  else:
                     data.append((frozenset(classification[0]), frozenset(classification[1]),
                                  frozenset(classification[2]), frozenset(classification[3]),
                                  frozenset(classification[4])))
               if hasCoherence:
                  lossMaps.append(frozenset(data))
         if len(lossMapsA) != len(lossMapsB):
            break
         # can we find permutations of the indices such that the lossMaps are equal, disregarding order?
         lossMapsA = set(lossMapsA)
         alphaIndices = tuple(alphaIndices)
         betaIndices = tuple(betaIndices)
         for alphaPerm, betaPerm in itertools.product(itertools.permutations(alphaIndices),
                                                      itertools.permutations(betaIndices)):
            indexMap = {x: y for x, y in itertools.chain(zip(alphaIndices, alphaPerm),
                                                         zip(betaIndices, betaPerm))}
            # Warning: this is extremely expensive already for six coefficients, it may be better to
            # just optimize!
            permLossMapsB = set(
               frozenset(
                  tuple(
                     frozenset(
                        (indexMap[m1], indexMap[m2])
                        for m1, m2 in classi
                     )
                     for classi in classification
                  )
                  for classification in data
               )
               for data in lossMapsB
            )
            if lossMapsA == permLossMapsB:
               return True
         inverted = not inverted
      return False