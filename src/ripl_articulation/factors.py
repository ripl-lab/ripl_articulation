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

import numpy as np

from functools import partial
from typing import List, Optional

import gtsam
from gtsam import CustomFactor


I_1x1 = np.eye(1)  # Creates a 1-element, 2D array


def articulation_flow_error(
    flow: np.ndarray,
    point: np.ndarray,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    """GPS Factor error function
    :param flow: Flow measurement on a point, to be filled with `partial`
    :param point: The point associated with flow, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    theta = 0.01
    scale = 0.01
    point2_pred = scale * flow + point

    key = this.keys()[0]

    xi_estimate = values.atVector(key)
    omega_estimate = xi_estimate[:3]

    omega_estimate_norm = np.zeros(3)

    if np.linalg.norm(omega_estimate) > 0.0001:
        omega_estimate_norm = omega_estimate / np.linalg.norm(omega_estimate)        

    vee_estimate = xi_estimate[3:]

    xi_estimate = np.concatenate([omega_estimate_norm, vee_estimate])

    expected_pose_diff = gtsam.Pose3.Expmap(theta * xi_estimate)
    

    if jacobians is None:
        expected_point = expected_pose_diff.transformFrom(point)
    else:
        d_r_self = np.zeros((3, 6), order='F')
        d_r_point = np.zeros((3, 3), order='F')
        expected_point = expected_pose_diff.transformFrom(point, d_r_self, d_r_point)
        jacobians[0] = d_r_self
    
    error = expected_point - point2_pred
 
    # if jacobians is not None:
    #     jacobians = d_r_xi
  
    return error


def relative_pose_factor(this : gtsam.CustomFactor,
                         values : gtsam.Values,
                         jacobians : Optional[List[np.ndarray]]) -> np.ndarray:
    """Reimplementation of the RelativePose factor from C++.

    Args:
        this (gtsam.CustomFactor): Interface to GTSAM.
        values (gtsam.Values): Values provided by the optimizer.
        jacobians (Optional[List[np.ndarray]]): Optional Jacobians.

    Returns:
        np.ndarray: Error in estimated twist.
    """
    sym_w_T_a = this.keys()[0]
    sym_w_T_b = this.keys()[1]
    sym_theta = this.keys()[2]
    sym_xi    = this.keys()[3]

    w_T_a_estimate = values.atPose3(sym_w_T_a)
    w_T_b_estimate = values.atPose3(sym_w_T_b)
    xi_estimate    = values.atVector(sym_xi)
    theta_estimate = values.atDouble(sym_theta)

    delta_measurement = w_T_a_estimate.between(w_T_b_estimate)

    if jacobians is not None:
        D_r_xi = np.zeros((6, 6), order='F')
        D_r_pose_a = np.zeros((6, 6), order='F')
        D_r_pose_b = np.zeros((6, 6), order='F')
        Hlog = np.zeros((6, 6), order='F')

        delta_expected = gtsam.Pose3.Expmap(theta_estimate * xi_estimate, D_r_xi)
        error = delta_expected.logmap(delta_measurement)

        gtsam.Pose3.Logmap(w_T_a_estimate, D_r_pose_a)
        gtsam.Pose3.Logmap(w_T_b_estimate, D_r_pose_b)
        gtsam.Pose3.Logmap(delta_expected, Hlog)

        # SUPER FREAKY: The sign of these Jacobians is different from C++
        # Somehow it works with these, but only these.
        # Combination was found experimentally.
        jacobians[0] = -D_r_pose_a 
        jacobians[1] =  D_r_pose_b
        jacobians[2] = -Hlog @ xi_estimate
        jacobians[3] = -Hlog @ D_r_xi
    else:
        delta_expected = gtsam.Pose3.Expmap(theta_estimate * xi_estimate)
        error = delta_expected.logmap(delta_measurement)

    return error


def RelativePoseFactor(sym_w_T_a, sym_w_T_b, sym_theta, sym_xi, noise_model) -> gtsam.CustomFactor:
    """Syntactic sugar for instantiating a relative pose factor for the GTSAM graph.

    Args:
        sym_w_T_a (_type_): Symbol referencing w_T_a(t).
        sym_w_T_b (_type_): Symbol referencing w_T_b(t).
        sym_theta (_type_): Symbol referencing theta(t).
        sym_xi (_type_): Symbol referencing Xi.
        noise_model (_type_): Noise model for the error.

    Returns:
        gtsam.CustomFactor: Instantiated custom factor for insertion in the graph.
    """
    return gtsam.CustomFactor(noise_model,
                              [sym_w_T_a, sym_w_T_b, sym_theta, sym_xi],
                              relative_pose_factor)


def compute_relative_point_error(w_T_a : gtsam.Pose3,
                                 w_P_b : gtsam.Point3,
                                 theta : float,
                                 xi : np.ndarray,
                                 jacobians : Optional[List[np.ndarray]]) -> np.ndarray:
    if jacobians is not None:
        J_a  = np.zeros((6, 6), order='F')
        J_xi_helper = np.zeros((6, 6), order='F')
        # Negative because we do b - predicted
        J_b  = np.eye(3, order='F')
        J_xi_expmap = np.zeros((6, 6), order='F')

        w_T_predicted = w_T_a.compose(gtsam.Pose3.Expmap(xi * theta, J_xi_expmap), J_a, J_xi_helper) # type: gtsam.Pose3
        error = w_P_b - w_T_predicted.translation()

        # Chain rule
        J_xi = J_xi_helper @ J_xi_expmap

        jacobians[0] = -J_a[3:]
        jacobians[1] = J_b
        jacobians[2] = -(J_xi @ xi)[3:]
        jacobians[3] = -J_xi[3:]
    else:
        w_T_predicted = w_T_a.compose(gtsam.Pose3.Expmap(xi * theta)) # type: gtsam.Pose3
        error = w_P_b - w_T_predicted.translation()

    return error


def relative_point_factor(this : gtsam.CustomFactor,
                          values : gtsam.Values,
                          jacobians : Optional[List[np.ndarray]]) -> np.ndarray:
    sym_w_T_a = this.keys()[0]
    sym_w_P_b = this.keys()[1]
    sym_theta = this.keys()[2]
    sym_xi    = this.keys()[3]

    w_T_a_estimate = values.atPose3(sym_w_T_a)
    w_P_b_estimate = values.atPoint3(sym_w_P_b)
    xi_estimate    = values.atVector(sym_xi)
    theta_estimate = values.atDouble(sym_theta)

    return compute_relative_point_error(w_T_a_estimate,
                                        w_P_b_estimate,
                                        theta_estimate,
                                        xi_estimate,
                                        jacobians)

def RelativePointFactor(sym_w_T_a, sym_w_P_b, sym_theta, sym_xi, noise_model) -> gtsam.CustomFactor:
    """Syntactic sugar for instantiating a relative point factor for the GTSAM graph.

    Args:
        sym_w_T_a (_type_): Symbol referencing the reference frame A.
        sym_w_P_b (_type_): Symbol referencing the point observation B.
        sym_theta (_type_): Symbol referencing the current theta estimate.
        sym_xi (_type_): Symbol referencing the Xi estimate.
        noise_model (_type_): Noise model of the error.

    Returns:
        gtsam.CustomFactor: Instantiated custom factor for insertion in the graph. 
    """
    return gtsam.CustomFactor(noise_model,
                              [sym_w_T_a, sym_w_P_b, sym_theta, sym_xi],
                              relative_point_factor)