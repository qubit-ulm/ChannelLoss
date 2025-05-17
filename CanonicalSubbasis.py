# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# This function generates the set of canonical subbases, where all of them are mutually simply
# inequivalent.

import numpy as np
import operator
from itertools import product, permutations, combinations, chain

def canonicalSubbasis(d, s, minKets=2, maxKets=None):
   assert d > 1 and s > 1
   if maxKets is None:
      maxKets = d**s
   assert minKets <= maxKets

   for kets in range(minKets, maxKets +1):
      yield from _canonicalPrepPatterns(d, s, kets)

def _canonicalPrepPatterns(d, s, subbasisLength):
   halfLength = subbasisLength >> 1
   if (halfLength << 1) < subbasisLength: # floating-point safe ceil
      halfLength += 1
   perms = tuple(permutations(range(s))) # calculate once
   for basis in _canonicalSubbasis(d, s, subbasisLength):
      basisset = set(basis)
      for a0len in range(halfLength, subbasisLength):
         for a0 in combinations(basis, a0len):
            a1 = tuple(sorted(basisset.difference(a0)))
            if len(a0) == len(a1) and a1 > a0:
               continue
            skip = False
            for perm in perms:
               permuted = tuple(sorted(tuple(ket[p] for p in perm) for ket in a0))
               if permuted < a0:
                  skip = True
                  break
               elif permuted == a0:
                  permuted = tuple(sorted(tuple(ket[p] for p in perm) for ket in a1))
                  if permuted < a1:
                     skip = True
                     break
            if not skip:
               yield [(0, *a0kets) for a0kets in a0] + [(1, *a1kets) for a1kets in a1]

def _canonicalSubbasis(d, s, subbasisLength):
   # (A) We first create a list of how all the slots look like. This extra work allows us to then
   # calculate all possible symmetry transformations beforehand.
   def nonincreasingList(length, maximum, leqThan, total):
      if length == 0:
         yield tuple()
      elif length == 1:
         r = subbasisLength - total
         if r <= maximum and (leqThan is None or leqThan[0] >= r):
            yield (r,)
      else:
         assert total <= subbasisLength
         if leqThan is None:
            for r in range(min(subbasisLength - total, maximum), -1, -1):
               yield from ((r, *n) for n in nonincreasingList(length -1, r, None, total + r))
         else:
            maximum = min(subbasisLength - total, maximum, leqThan[0])
            yield from ((maximum, *n) for n in nonincreasingList(length -1, maximum, leqThan[1:],
                                                                 total + maximum))
            for r in range(maximum -1, -1, -1):
               yield from ((r, *n) for n in nonincreasingList(length -1, r, None, total + r))

   itemgetters = {}
   def cachedItemgetter(items):
      nonlocal itemgetters
      if not items in itemgetters:
         assert len(items) > 1 # We could use our own len-1-aware version for safety, but
                               # a) this should never happen, as s > 1
                               # b) operator.itemgetter is orders of magnitude faster
         itemgetters[items] = operator.itemgetter(*items)
      return itemgetters[items]

   slots = [None] * s
   slotPermutations = None
   slotFlips = None
   def buildSlotList(index=0):
      nonlocal slots
      if index == s:
         # we must take into account additional symmetries:
         nonlocal slotPermutations, slotFlips
         # 1. two (or more) slots may have the same signature and therefore give rise to a
         #    permutation symmetry. find those groups.
         slotPermutations = []
         lastSlot = 0
         lastLevels = slots[0]
         for slot, levels in enumerate(slots):
            if levels != lastLevels:
               if lastSlot == slot -1:
                  slotPermutations.append(((lastSlot,),))
               else:
                  slotPermutations.append(permutations(range(lastSlot, slot)))
               lastSlot, lastLevels = slot, levels
         if lastSlot == s -1:
            slotPermutations.append(((lastSlot,),))
         else:
            slotPermutations.append(permutations(range(lastSlot, s)))
         slotPermutations = tuple(cachedItemgetter(perm)
                                  for perms in product(*slotPermutations)
                                  for perm in (tuple(chain(*perms)),))

         # 2. repeated numbers in slots may give rise to flip symmetries.
         slotFlips = [None] * s
         for i, slot in enumerate(slots):
            # slot contains all the levels, which are populated in a descending order
            # check for all duplicates - those are the ones that may be flipped
            perms = []
            lastLevel = 0
            lastPopulation = slot[0]
            for level, population in enumerate(slot):
               if population != lastPopulation:
                  if lastLevel == level -1:
                     perms.append(((lastLevel,),))
                  else:
                     perms.append(permutations(range(lastLevel, level)))
                  lastLevel, lastPopulation = level, population
            if lastLevel == d -1:
               perms.append(((lastLevel,),))
            else:
               perms.append(permutations(range(lastLevel, d)))
            slotFlips[i] = tuple(tuple(chain(*ps)) for ps in product(*perms))

         # we need mutable slots, but the mutation must not propagate backwards in recursion
         yield from slotsToKets(np.array(slots, dtype='int'))
      else:
         lastSlot = slots[index -1] if index > 0 else (subbasisLength,) * d
         for newSlot in nonincreasingList(d, lastSlot[0], lastSlot, 0):
            assert newSlot <= lastSlot
            slots[index] = newSlot
            yield from buildSlotList(index +1)

   # (B) For every given slot list, we create the corresponding ket list.
   def productGtr(than, productItems):
      # this is like (it for it in itertools.product(*productItems) if it > than), but optimized
      # such that it breaks earlier when the condition can no longer be fulfilled
      # It also ensures that slots are filled consecutively with levels
      assert than is None or len(than) == len(productItems)
      if not productItems:
         yield tuple()
      elif len(productItems) == 1:
         if than is None:
            for p in productItems[0]:
               yield (p,)
         else:
            for p in productItems[0]:
               if p > than[0]:
                  yield (p,)
      else:
         if than is None:
            yield from ((p, *new) for p in productItems[0]
                        for new in productGtr(None, productItems[1:]))
         else:
            if than[0] in productItems[0]:
               yield from ((than[0], *new) for new in productGtr(than[1:], productItems[1:]))
            yield from ((p, *new) for p in productItems[0] if p > than[0]
                        for new in productGtr(None, productItems[1:]))

   kets = [None] * subbasisLength
   def slotsToKets(slots, index=0):
      nonlocal kets
      if index == subbasisLength:
         assert (slots == 0).all()
         yield kets
      else:
         newKets = kets[:index +1] # we need this for the "<" comparison - should still be more
                                   # efficient than growing a ket list
         lastKet = kets[index -1] if index > 0 else (-1,) * s
         levelsForSlot = tuple(tuple(level for level, count in enumerate(slot) if count > 0)
                               for slot in slots)
         for newKet in productGtr(lastKet, levelsForSlot):
            assert newKet not in kets[:index]
            kets[index] = newKet
            newKets[index] = newKet
            keep = True
            for flips in product(*slotFlips):
               flipped = tuple(tuple(levelPerm[slot] for slot, levelPerm in zip(ket, flips))
                               for ket in newKets)
               for permGet in slotPermutations:
                  permuted = sorted(permGet(ket) for ket in flipped)
                  if permuted < newKets:
                     keep = False
                     break
               if not keep:
                  break
            if keep:
               newSlots = slots.copy()
               for sl, l in enumerate(newKet):
                  newSlots[sl, l] -= 1
               yield from slotsToKets(newSlots, index +1)

   yield from buildSlotList()