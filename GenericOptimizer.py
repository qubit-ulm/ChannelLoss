# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

import mosek
import abc

class Optimizer(metaclass=abc.ABCMeta):
   def __init__(self, env):
      self._task = env.Task(0, 0)
      self._task.putintparam(mosek.iparam.intpnt_multi_thread, mosek.onoffkey.off)
      self._task.putintparam(mosek.iparam.num_threads, 1)
      self._task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
      self._task.putintparam(mosek.iparam.intpnt_scaling, mosek.scalingtype.none)

      self.__setupTask()

   @abc.abstractmethod
   def __setupTask():
      pass