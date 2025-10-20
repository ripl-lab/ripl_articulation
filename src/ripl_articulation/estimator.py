"""
 Copyright (c) 2025 Russell Buchanan, University of Waterloo
                    Adrian Röfer, University of Freiburg

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
 """


import gtsam.noiseModel
import numpy as np
import gtsam

from .factors  import RelativePoseFactor
from typing    import Tuple


def solve_articulation(poses_a : np.ndarray,
                       poses_b : np.ndarray,
                       prior_xi : np.ndarray=None,
                       prior_theta : float=0.0,
                       xi_noise_model=gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)),
                       pose_noise_model=gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-2)) -> Tuple[np.ndarray, np.ndarray]:
    """Solves for the best articulation explaining N world-space observation pairs w_T_a and w_T_b
       in terms of one constant twist and a scalar variable. Returns the estimated twist Xi (6),
       and a time series of thetas (N). The relation to the original data is as follows:

       w_T_b_est = w_T_a(t) @ exp(Xi * theta(t))

       The estimation is heavily biased towards theta(t) = 0, implying that w_T_a = w_T_b.
       Consider this bias when generating the observations to be processed.

    Args:
        poses_a (np.ndarray): Observations of frame A (N, 6) or (N, 4, 4). 
        poses_b (np.ndarray): Observations of frame B (N, 6) or (N, 4, 4). 
        prior_xi (np.ndarray, optional): Prior to give for the articulation.
                                         If none is given, the estimation of Xi is unbiased but also unconstrained. Defaults to None.
        prior_theta (float, optional): Prior for theta(0). Defaults to 0.0.
        xi_noise_model (gtsam.noiseModel.Diagonal.Sigmas, optional): Noise model for the estimation of Xi. Defaults to gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)).
        pose_noise_model (gtsam.noiseModel.Diagonal.Sigmas, optional): Noise model for the observed poses. Defaults to gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-2).

    Returns:
        (np.ndarray, np.ndarray): Estimate of Xi (6), and thetas (N).
    """
    sym_xi = gtsam.symbol('X', 0)

    thetas = []
    syms_a = []
    syms_b = []

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    if prior_xi is not None:
        graph.addPriorVector(sym_xi, prior_xi, xi_noise_model)
        initial_values.insert(sym_xi, prior_xi)
    else:
        initial_values.insert(sym_xi, np.zeros(6))


    for idx, (w_T_a, w_T_b) in enumerate(zip(poses_a, poses_b)):
        thetas.append(sym_theta:=gtsam.symbol('T', idx))

        syms_a.append(sym_a:=gtsam.symbol('A', idx))
        syms_b.append(sym_b:=gtsam.symbol('B', idx))

        w_T_a_gtsam = gtsam.Pose3(np.require(w_T_a, requirements='F')) if w_T_a.ndim == 2 else gtsam.Pose3.Expmap(w_T_a)
        w_T_b_gtsam = gtsam.Pose3(np.require(w_T_b, requirements='F')) if w_T_b.ndim == 2 else gtsam.Pose3.Expmap(w_T_b)

        graph.addPriorPose3(sym_a, w_T_a_gtsam, pose_noise_model)
        graph.addPriorPose3(sym_b, w_T_b_gtsam, pose_noise_model)

        graph.add(RelativePoseFactor(sym_a, sym_b, sym_theta, sym_xi, pose_noise_model))

        initial_values.insert(sym_a, w_T_a_gtsam)
        initial_values.insert(sym_b, w_T_b_gtsam)
        initial_values.insert(sym_theta, prior_theta)

    graph.addPriorDouble(thetas[0], prior_theta, gtsam.noiseModel.Diagonal.Sigmas(np.deg2rad(np.ones(1) * 25)))

    params = gtsam.LevenbergMarquardtParams()
    # params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)

    results = optimizer.optimize()

    return results.atVector(sym_xi), np.array([results.atDouble(t) for t in thetas])
