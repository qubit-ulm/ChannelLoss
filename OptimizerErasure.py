# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# Here, we define the class that performs the optimization using a general solver for the variables
# pertaining to the input state and a convex solver for the maps, where we allow different maps
# depending on the different arrival configurations; r is still fixed. This corresponds to the full
# erasure scenario discussed in section III.I.

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
import mosek
import GenericOptimizer
from math import inf
from itertools import combinations

class Optimizer(GenericOptimizer.Optimizer):
   def __init__(self, env, ditness, send, receive):
      """
         env:     MOSEK environment
         ditness: dimensionality of the carriers
         send:    number of particles to be sent (giving total dimension ditness^send)
         receive: number of particles to be received
      """
      assert ditness == 2
      assert send > receive
      self.ditness = ditness
      self.send = send
      self.receive = receive
      self._slots = tuple(combinations(range(send), receive))

      super().__init__(env)

   def __setupTask(self):
      dtrOut = self.ditness ** self.receive
      slots = len(self._slots)
      r = dtrOut * (dtrOut +1) //2

      self._d = 2 * dtrOut
      self._task.appendcons(slots * (2 + r))
      self._task.appendvars(1)
      self._task.putvarbound(0, mosek.boundkey.fr, -inf, inf)
      self._task.appendbarvars([self._d, dtrOut] * slots)
      self._task.putobjsense(mosek.objsense.maximize)
      self._task.putcj(0, 1.0)

      self._extractions = {
         (i, j): self._task.appendsparsesymmat(self._d, [i], [j], [1.0 if i == j else 0.5])
         for i in range(self._d) for j in range(i +1)
      }
      extractionsHalf = {
         (i, j): self._task.appendsparsesymmat(dtrOut, [i], [j], [1.0 if i == j else 0.5])
         for i in range(dtrOut) for j in range(i +1)
      }

      # linear constraint: fidelity
      self._task.putacol(0, range(slots), [-1.0] * slots)
      self._task.putconboundsliceconst(0, slots, mosek.boundkey.lo, 0.0, inf)
      # linear constraint: traced out map
      self._task.putbaraijlist([x for i in range(2 * slots, slots * (2 + r)) for x in (i, i)],
                               [x for s in range(slots) for i in range(r) for x in (2*s, 2*s +1)],
                               [x for i in range(0, 3*r, 3) for x in (i, i +2)] * slots,
                               [x for i in range(0, 3*r, 3) for x in (i +2, i +3)] * slots,
                               [x for i in range(dtrOut) for j in range(i +1)
                                  for x in (self._extractions[(i, j)],
                                            self._extractions[(max(i + dtrOut, j + dtrOut),
                                                               min(i + dtrOut, j + dtrOut))],
                                            extractionsHalf[(i, j)])],
                               [1.0] * (3 * r))
      constr = [0.0 if i != j else 1.0 for i in range(dtrOut) for j in range(i +1)] * slots
      self._task.putconboundslice(2 * slots, slots * (2 + r), [mosek.boundkey.fx] * len(constr),
                                  constr, constr)

      def calcPT(slots):
         # slots tells which slots are received. Build up an array of dimensions and einsum strings
         # that correctly represent this
         groupedSlots = []
         group = 0
         lastSlot = -1
         for i in sorted(slots):
            if i -1 == lastSlot:
               group += 1
               lastSlot = i
            else:
               if group != 0:
                  groupedSlots.append((True, self.ditness ** group))
               groupedSlots.append((False, self.ditness ** (i - lastSlot -1)))
               group = 1
               lastSlot = i
         if group > 0:
            groupedSlots.append((True, self.ditness ** group))
         if lastSlot != self.send -1:
            groupedSlots.append((False, self.ditness ** (self.send - lastSlot -1)))
         if groupedSlots[0][0]:
            groupedSlots[0] = (True, 2 * groupedSlots[0][1])
         else:
            groupedSlots.insert(0, (True, 2))

         dimensions = []
         einsumStr = ["", ""]
         einsumSymbolsSet = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
         outStr = ["", ""]
         for take, dim in groupedSlots:
            dimensions.append(dim)
            if take:
               chr = einsumSymbolsSet.pop()
               einsumStr[0] += chr
               outStr[0] += chr
               chr = einsumSymbolsSet.pop()
               einsumStr[1] += chr
               outStr[1] += chr
            else:
               chr = einsumSymbolsSet.pop()
               einsumStr[0] += chr
               einsumStr[1] += chr
         dimensions = (*dimensions, *dimensions)
         einsumStr = einsumStr[0] + einsumStr[1] + "->" + outStr[0] + outStr[1]
         path = np.einsum_path(einsumStr, np.empty(dimensions), optimize='greedy')
         return dimensions, einsumStr, path[0]

      self._partialTraces = list(map(calcPT, self._slots))
      self._rhoMatrix = np.empty((2 * (self.ditness ** self.send),) *2)
      self._trius = np.triu_indices(2 * (self.ditness ** self.send))
      self._dtrOut = dtrOut

   def _setupDistillation(self):
      slots = len(self._slots)
      dtrOut = self._dtrOut
      for slot, (dimensions, einsumStr, path) in enumerate(self._partialTraces):
         rhoLoss = np.einsum(einsumStr, self._rhoMatrix.reshape(dimensions), optimize=path).\
            reshape((2, dtrOut, 2, dtrOut))
         # einsum("ijkl, mjnl", choi, rhoLoss)
         # fidelity
         fidmats = [self._extractions.get((i * dtrOut + j, k * dtrOut + l), None)
                    for j in range(dtrOut) for l in range(dtrOut)
                    for i, k in [(0, 0), (1, 0), (1, 1)]]
         fidrhos = [x for j in range(dtrOut) for l in range(dtrOut)
                      for x in (.5*rhoLoss[0, j, 0, l] if j == l else rhoLoss[0, j, 0, l],
                                rhoLoss[1, j, 0, l],
                                .5*rhoLoss[1, j, 1, l] if j == l else rhoLoss[1, j, 1, l])]
         self._task.putbaraij(slot, 2 * slot, [x for x in fidmats if x != None],
                              [y for x, y in zip(fidmats, fidrhos) if x != None])
         # trace
         self._task.putbaraij(slots + slot, 2 * slot,
                              [self._extractions[(i * dtrOut + j, i * dtrOut + l)]
                               for j in range(dtrOut) for l in range(j +1) for i in range(2)],
                              [x for j in range(dtrOut) for l in range(j +1)
                                 for x in ([(rhoLoss[0, j, 0, l] + rhoLoss[1, j, 1, l]) if j == l else
                                            2*(rhoLoss[0, j, 0, l] + rhoLoss[1, j, 1, l])] *2)])

   def _optimizeDistillation(self, pdist, retState=False):
      # trace constraint: trace must be fixed to the given value
      self._task.putconboundsliceconst(len(self._slots), 2*len(self._slots), mosek.boundkey.fx,
                                       pdist, pdist)
      self._task.optimize()
      solsta = self._task.getsolsta(mosek.soltype.itr)
      if solsta == mosek.solsta.optimal:
         fid = self._task.getprimalobj(mosek.soltype.itr) / pdist
         if retState:
            chois = np.empty((len(self._slots), self._d * (self._d +1)//2))
            for i in range(len(self._slots)):
               self._task.getbarxj(mosek.soltype.itr, 2*i, chois[i])
            return (fid, chois)
         else:
            return fid
      else:
         if retState:
            return (0, np.empty((len(self._slots), self._d * (self._d +1)//2)))
         else:
            return 0

   def _minimizer(self, x, pdist, retState=False):
      # make sure the initial state is normalized
      nrm = np.linalg.norm(x)
      if nrm != 1:
         x /= nrm
      np.outer(x, x, out=self._rhoMatrix)

      self._setupDistillation()
      if retState:
         _, choi = self._optimizeDistillation(pdist, True)
         return choi, x
      else:
         f = self._optimizeDistillation(pdist)
         return -f

   def optimize(self, pdist, initialPsi):
      """
         pdist:      distillation success probability
         initialPsi: full ditness^send-dimensional state vector used as initial point

         return: tuple(fidelity, vec of optimal Choi matrix, optimial state vector)
      """
      assert len(initialPsi) == 2 * (self.ditness ** self.send)
      with np.errstate(divide='ignore'):
         mini = spopt.minimize(self._minimizer, initialPsi, args=(pdist,), method='BFGS')
      return (-mini.fun, *self._minimizer(mini.x, pdist, True))