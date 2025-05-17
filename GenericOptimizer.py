# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

import numpy as np
import scipy.linalg as spla
import mosek
import abc

class Optimizer(metaclass=abc.ABCMeta):
   def __init__(self, env):
      if env is None:
         return
      self._task = env.Task(0, 0)
      self._task.putintparam(mosek.iparam.num_threads, 1)
      self._task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
      self._task.putintparam(mosek.iparam.intpnt_scaling, mosek.scalingtype.none)

      self.__setupTask()

   @abc.abstractmethod
   def __setupTask():
      pass

   @staticmethod
   def binaryEntropy(x):
      if x < 1e-5 or x > 1 - 1e-5:
         return 0
      else:
         return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

   __conjugator = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
   @staticmethod
   def entanglementOfFormation(rho):
      return None # currently, we are not interested
      if rho.shape != (4, 4):
         return NotImplemented
      tr = np.trace(rho)
      if tr == 0:
         return 0
      rho /= tr
      sqrtRhoC = spla.sqrtm(rho) @ Optimizer.__conjugator
      # svd( sqrt(rho) @ sqrt(rhoTilde = __conjugator @ rho.conj() @ __conjugator) )
      eigs = spla.eigh(sqrtRhoC @ rho.conj() @ sqrtRhoC.T, eigvals_only=True, overwrite_a=True)
      eigs[eigs < 0] = 0
      concurrence = max(0, [-1, -1, -1, 1] @ np.sqrt(eigs))
      return Optimizer.binaryEntropy(.5 * (1 + np.sqrt(max(0, 1 - concurrence **2))))