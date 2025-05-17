# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# This is a sample command line application that can be used for optimization with fixed s and r.

import OptimizerStepwiseKnownMap
import CanonicalSubbasis

# We need to index to probability and do arithmetics, so don't use floats. Every probability is in
# multiples of this.
pmultiple = 1/100

def getPatterns(d, s, r, minKets):
    for sbl in range(minKets, d**s +1):
        uniqueIdentifiers = {}
        for prepPattern in CanonicalSubbasis._canonicalPrepPatterns(d, s, sbl):
            optMap = OptimizerStepwiseKnownMap.Optimizer(None, d, s, r, prepPattern)
            uniqueID = optMap.uniqueIdentifierHeuristic()
            if uniqueID is False or uniqueID in uniqueIdentifiers:
                continue
            else:
                uniqueIdentifiers[uniqueID] = prepPattern
            yield prepPattern