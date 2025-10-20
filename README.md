# The RIPL Lab Articulation Estimation Tools

Authors: Russell Buchanan, Adrian Röfer

This package contains the implementation of the articulation estimators described in:
 - "Online estimation of articulated objects with factor graphs using vision and proprioceptive sensing", Buchanan et al., ICRA 2024. [ArXiv](https://arxiv.org/pdf/2309.16343).

## Installation

Simply install the package from PyPi as
```bash
pip install ripl_articulation
```

## Usage

The core functionality is estimating an SE3 articulation from a series of SE3 pose pairs like so:
```python
from ripl_articulation import solve_articulation_from_poses

# Poses of Frame A in some reference frame
ref_T_A = ...
# Poses of Frame B in some reference frame
ref_T_B = ...

Xi, thetas = solve_articulation_from_poses(ref_T_A, ref_T_B)
```
From the solution we can reconstruct **B** as `ref_T_B = ref_T_A @ exp(Xi * thetas)`, where `exp` is your favorite SE3 exponential map implementation.

Further, the package provides the individual articulation factor `RelativePoseFactor` which can be added to a `gtsam` factor graph. It can simply be imported as `from ripl_articulation import RelativePoseFactor`.

In addition to these core functionalities of the paper, the package includes a `utils` package, which implements a discrete ground truth articulation container, i.e. `Articulation`, and distance metrics to compare solved articulations to it.

## Citing the Package

If you use or compare to this estimator, please cite our work as
```
@inproceedings{buchanan2024online,
  title={Online estimation of articulated objects with factor graphs using vision and proprioceptive sensing},
  author={Buchanan, Russell and R{\"o}fer, Adrian and Moura, Jo{\~a}o and Valada, Abhinav and Vijayakumar, Sethu},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={16111--16117},
  year={2024},
  organization={IEEE}
}
```
