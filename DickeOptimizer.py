# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# The functions mulnomSend and mulnomReceive return dictionaries whose keys are the kets (Alice,
# Dicke) that are present for the initial state and the map, respectively. By default, they give all
# possible combinations, but this may be changed to contain only a subset.

from sympy.ntheory.multinomial import multinomial_coefficients_iterator as mulnomit
import mosek
import abc
import GenericOptimizer

# we always sort these, since the sympy ordering is not the most natural one.
def mulnom(m, n):
   return {k: v for k, v in sorted(mulnomit(m, n))}
def mulnomSend(ditness, s, r):
   return {(i, *k): v for i in (0, 1) for k, v in mulnom(ditness, s).items()}
def mulnomReceive(ditness, s, r):
   return {(i, *k): v for i in (0, 1) for k, v in mulnom(ditness, r).items()}

class Optimizer(GenericOptimizer.Optimizer):
   def __init__(self, env, ditness, send, receive):
      """
         env:     MOSEK environment
         ditness: dimensionality of the carriers
         send:    number of particles to be sent (in Dicke form)
         receive: number of particles to be received (in Dicke form)
      """
      assert ditness >= 2
      assert send > receive
      self.ditness = ditness
      self.send = send
      self.receive = receive
      self._multinomialsR = mulnomReceive(ditness, send, receive)
      self._multinomialsS = mulnomSend(ditness, send, receive)
      self._multinomialsSR = mulnom(ditness, send - receive)

      super().__init__(env)