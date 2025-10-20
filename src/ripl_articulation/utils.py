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
from dataclasses import dataclass


@dataclass
class Articulation:
    # Position of the articulation in space as a point (3,)
    position : np.ndarray
    # Axis of the articulation (3,)
    axis : np.ndarray
    # Discrete type of articulation either PRISMATIC, or REVOLUTE
    type : str


def compute_twist_center(w_T_xi : np.ndarray, xi : np.ndarray) -> np.ndarray:
    xi_P_rotation_origin = np.cross(xi[:3], xi[3:]) / omega_sq_norm if (omega_sq_norm:=(xi[:3]**2).sum()) > 1e-4 else np.zeros(3)
    return w_T_xi[:3, :3] @ xi_P_rotation_origin + w_T_xi[:3, 3]

def compute_articulation_delta_twist(gt : Articulation, w_T_xi : np.ndarray, xi : np.ndarray) -> np.ndarray[float, float]:
    """Computes the delta between a given articulation and a twist model consisting of
       world transform and twist vector. It extracts the rotation or translation axis 
       from the twist and the center of the rotation/a point on the translation.

       Returned errors are angular divergence of the main axis and positional divergence for the centers of rotations.

    Args:
        gt (Articulation): Articulation to compare against.
        w_T_xi (np.ndarray): World offset of the predicted articulation.
        xi (np.ndarray): Predicted articulation in the form of a twist.

    Returns:
        np.ndarray[float, float]: Axis divergens in rad, positional divergence in meters.
    """
    if gt.type == 'PRISMATIC':
        normalized_translation_direction = xi[3:] / np.linalg.norm(xi[3:])

        return compute_articulation_delta_point_and_axis(gt, w_T_xi[:3, 3], w_T_xi[:3, :3] @ normalized_translation_direction)
    elif gt.type == 'REVOLUTE':
        normalized_rotation_axis = xi[:3] / np.linalg.norm(xi[:3])
        w_P_rotation_origin  = compute_twist_center(w_T_xi, xi)
        w_V_rotation         = w_T_xi[:3, :3] @ normalized_rotation_axis
        return compute_articulation_delta_point_and_axis(gt, w_P_rotation_origin, w_V_rotation)
    else:
        raise ValueError(f'Unknown articulation type: "{gt.type}"')


def compute_articulation_delta_point_and_axis(gt : Articulation, w_P_center : np.ndarray, w_V_axis : np.ndarray) -> np.ndarray[float, float]:
    """Computes the delta between a given articulation and a predicted model consisting of a supporting point and 
       an axis. The type of error is computed based on the ground-thruth's articulation type.

       Eeturned errors are angular divergence of the main axis and positional divergence for the centers of rotations.

    Args:
        gt (Articulation): Articulation to compare against.
        w_P_center (np.ndarray): Supporting point of articulation in world frame.
        w_V_axis (np.ndarray): Axis of articulation in world frame.

    Returns:
        np.ndarray[float, float]: Axis divergens in rad, positional divergence in meters.
    """
    if gt.type == 'PRISMATIC':
        # Amount of rotation per translation. Should be 0
        return np.array([np.arccos(np.clip(np.abs(w_V_axis @ gt.axis), 0, 1)), 0])
    elif gt.type == 'REVOLUTE':
        axis_cos = np.clip(np.abs(w_V_axis.T @ gt.axis), 0, 1)
        v_cross_v = np.cross(w_V_axis, gt.axis)
        # If lines are not parallel, calculate their distance
        if (vv_norm:=np.linalg.norm(v_cross_v)) >= 1e-4:
            articulation_distance = np.abs(((w_P_center - gt.position) * v_cross_v).sum()) / vv_norm
        else:
            articulation_distance = np.linalg.norm(np.cross(w_P_center - gt.position, gt.axis))
        return np.array([np.arccos(axis_cos), articulation_distance])
    else:
        raise ValueError(f'Unknown articulation type: "{gt.type}"')
