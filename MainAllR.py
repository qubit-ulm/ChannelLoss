# This work is licensed under CC BY 4.0.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
# Copyright (C) 2021 Benjamin Desef

# This is a sample command line application that can be used to postprocess results obtained for the
# optimization with fixed r and adapted to multiple r values.

import numpy as np
import scipy.linalg as spla
import mosek
import OptimizerDickeOuterStateAllR

# We need to index to probability and do arithmetics, so don't use floats. Every probability is in
# multiples of this.
pmultiple = 1/100
env = mosek.Env()

def initialize(*args):
   global opti
   opti = OptimizerDickeOuterStateAllR.Optimizer(env, *args)

def runa(p, files):
   global opti
   return max((opti.optimize(p, np.loadtxt(file)) for file in files), key=lambda args: args[0])

def runb(p, initialVec):
   global opti
   return opti.optimize(p, initialVec)

# command line
if __name__ == '__main__':
   import time
   starttime = time.time()

   import argparse
   import os
   import platform
   import psutil
   import pathlib
   # multiprocessing.Pool seems to be broken and sometimes doesn't catch up with adding new tasks
   # in callbacks, so we use futures.
   import concurrent.futures as fut

   parser = argparse.ArgumentParser("Finds the optimal distillation algorithm for the best possible"
                                    " multiplexed state, using various r values.")
   parser.add_argument("d", help="Ditness", type=int, default=2)
   parser.add_argument("s", help="Send", type=int, default=3)
   parser.add_argument("ptrans", help="Transmission success probability", type=float)
   parser.add_argument("--L", help="Set this flag so that the ptrans parameter is instead interpret"
                       "ed as a length in kilometers and converted using the default attenuation le"
                       "ngth of 22.4 km", action="store_true")
   parser.add_argument("--threshold", help="Relative probability threshold below which r values are"
                       " no longer taken into account", type=float, default=.001)
   parser.add_argument("--initR", help="Take initial states from the folder corresponding to this r"
                       " value", type=int, default=2)
   parser.add_argument("--pmin", help="Minimum total success probability to scan (percent)",
                       type=int, default=1)
   parser.add_argument("--pmax", help="Maximum total success probability to scan (percent)",
                       type=int, default=100)
   parser.add_argument("--workers", help="Number of worker threads", type=int,
                       default=int(os.environ.get('SLURM_NTASK',
                                                  psutil.cpu_count(logical=False) or 1)))
   args = parser.parse_args()
   assert args.d >= 2
   assert args.s >= args.initR
   if args.L:
      assert args.ptrans > 0
      args.ptrans = np.exp(-.046 * args.ptrans)
   else:
      assert 0 < args.ptrans and args.ptrans <= 1

   print("Parameters: d = {:d}, s = {:d}, ptrans = {:.8f}, threshold = {:.8f}, initR = {:d}".
         format(args.d, args.s, args.ptrans, args.threshold, args.initR))

   originalDir = "../{:d} {:d} {:d}".format(args.d, args.s, args.initR)
   try:
      directory = "{:.8f} {:d} {:d}".format(args.ptrans, args.d, args.s)
      if not os.path.exists(directory):
         os.mkdir(directory)
      os.chdir(directory)
   except OSError:
      raise Exception("Could not change directory")
   if not os.path.exists(originalDir):
      raise Exception("No initial data found. Make sure to run the r-independent program first to g"
                      "enerate initial data.")

   # callbacks from multiprocessing (all run in main process)
   # First, every scan starts with the same initial data and triggers the normal callback done().
   # Here, we try to use the output data to re-scan the neighboring probabilities (to avoid missing
   # bifurcations). If the result of the re-scan is able to improve the fidelity, we continue
   # re-scanning.
   results = {p: False for p in range(args.pmin, args.pmax +1)}

   def done(p, direction, f, chois, rho):
      global results, active, pmultiple, pool, args
      if f - results[p] > 0:
         pfrac = p * pmultiple
         print("Success for p = {:.2}: F = {:.8f}".format(pfrac, f))
         newfn = "{:.2f} {:.8f}".format(pfrac, f)
         np.savetxt(newfn + " rho.dat", rho, delimiter=',')
         np.savetxt(newfn + " chois.dat", chois, delimiter=',')
         if results[p] is not False:
            oldfn = "{:.2f} {:.8f}".format(pfrac, results[p])
            if newfn != oldfn:
               os.remove(oldfn + " rho.dat")
               os.remove(oldfn + " chois.dat")
         results[p] = f
         if direction <= 0:
            newp = p - 1
            if newp >= args.pmin:
               active += 1
               pool.submit(runb, newp * pmultiple, rho).add_done_callback(
                  lambda x, p=newp: done(p, -1, *x.result())
               )
         if direction >= 0:
            newp = p + 1
            if newp <= args.pmax:
               active += 1
               pool.submit(runb, newp * pmultiple, rho).add_done_callback(
                  lambda x, p=newp: done(p, 1, *x.result())
               )
      active -= 1

   if platform.system() in {'Windows', 'Darwin'}:
      # Windows: no fork, all processes have to initialize by themselves
      # Mac: broken fork, so Python spawns by default
      pool = fut.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                     initargs=(args.d, args.s, args.ptrans, args.threshold))
   else:
      # *nix:
      initialize(args.d, args.s, args.ptrans, args.threshold)
      pool = fut.ProcessPoolExecutor(max_workers=args.workers)
   # maximum total success probability: cannot exceed 1 - (1 - ptrans)^s
   args.pmax = min(args.pmax, int((1 - (1 - args.ptrans)**args.s) / pmultiple + 1e-8))
   active = 0
   for p in range(args.pmin, args.pmax +1):
      active += 1
      # In principle, we may also try to exclude some initial points here based on the fact that
      # their distillation probability is quite small. However, in the end, we cannot really
      # predict what comes out of the initial states...
      pool.submit(runa, p * pmultiple,
                  tuple(map(str, pathlib.Path(originalDir).glob("* rho.dat")))).\
         add_done_callback(lambda x, p=p: done(p, 0, *x.result()))
   while active > 0:
      time.sleep(.1)
   pool.shutdown()

   print("Total time: {:.2f} s".format(time.time() - starttime))