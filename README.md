# Protecting quantum states against loss

This work is licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) ![CC BY](https://i.creativecommons.org/l/by/4.0/88x31.png).

## Requirements

All scripts run on Python 3.7 or newer. The following packages are required:

- `numpy`
- `scipy`
- `mosek` (+ license)
- `sympy` for the `multinomial_coefficients_iterator` function
- `psutil` for the exemplary command line programs, only used to get the physical CPU count

## Content

The `Optimizer...py` files define a class `Optimizer` that is responsible for carrying out the
optimization.
The filename indicates the problem this corresponds to; this is detailed in the file header.

We additionally provide two command line applications:

- `Main.py` runs the optimizations for given parameters `d`, `s`, and `r`.
  Various configuration options are available, see the help.

  The output is stored in a subfolder named `<d> <s> <r>` according to the parameters; the files are
  named `<pdist> <f> <type>.dat`, where `<pdist>` is the distillation success probability and
  `<f>` the optimized fidelity.
  If `<type>` is `choi`, it contains the vectorized upper triangle of the Choi state of the
  distillation map (in the Dicke basis).
  If `<type>` is `rho`, it contains the vectorized upper triangle of the input density matrix (in
  the Dicke basis).
  If the `--erasure` parameter is specified, the optimization is done in the full computational
  basis and `rho` is replaced by `psi`; the file then contains the full input state vector.
  Additionally, the subfolder is suffixed with ` erasure`.
- `MainAllR.py` runs the optimizations for given parameters `d`, `s`, and `ptrans`.
  Various configuration options are available, see the help.

  This program uses the data created by `Main.py` (without the `--erasure` option) as initial points
  and therefore can only be run afterwards.
  It explores the possibility to use different maps for various `r` values.

  The output is stored in a subfolder named `<ptrans> <d> <s>`; the files are named
  `<ptot> <f> <type>.dat`, where `<ptot>` now is the total success probability.

Both applications are parallelized and by default use either the environment variable `SLURM_NTASK`
(if the SLURM manager is used) or the number of physical cores available.
This can be configured using the `--workers` option.