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
import gtsam

from tqdm import tqdm

from scipy.spatial.transform import Rotation

from ripl_articulation import solve_articulation_from_poses
from ripl_articulation.utils import Articulation, \
                                    compute_articulation_delta_twist


def max_removed_pose(w_Ts_poses : np.ndarray) -> np.ndarray:
    w_Ps_positions = w_Ts_poses[:, :3, 3]
    deltas = np.linalg.norm(w_Ps_positions - w_Ps_positions[0], axis=-1)
    return w_Ts_poses[np.argmax(deltas)]


def test_pose_estimator():
    deltas_lin = []
    deltas_rot = []
    origin_est_error = []
    out_metric_prismatic = []
    out_metric_revolute  = []

    for sample in tqdm(range(1000), desc="Generating test problems..."):
        w_T_b_prismatic = np.array([np.eye(4)] * 10)
        for x in range(len(w_T_b_prismatic)):
            w_T_b_prismatic[x, 0, 3] = x * 0.1
        
        w_T_offset = np.eye(4)
        w_T_offset[:3, 3] = (np.random.random(3) - 0.5) * 12
        
        gt_prismatic = Articulation(w_T_offset[:3, 3],
                                    w_T_offset[:3, 0],
                                    'PRISMATIC')

        w_T_offset[:3, :3] = Rotation.random().as_matrix()

        w_T_b_prismatic = w_T_offset @ w_T_b_prismatic


        w_T_b_prior = max_removed_pose(w_T_b_prismatic)
        a_T_b_prior = gtsam.Pose3.Logmap(gtsam.Pose3(np.linalg.inv(w_T_b_prismatic[0]) @ w_T_b_prior))

        xi, thetas = solve_articulation_from_poses(np.array([w_T_b_prismatic[0]] * len(w_T_b_prismatic)),
                                                   w_T_b_prismatic,
                                                   prior_xi=a_T_b_prior)
        our_error = compute_articulation_delta_twist(gt_prismatic, w_T_b_prismatic[0], xi)
        if our_error[0] > np.deg2rad(10):
            print('Wat')
        out_metric_prismatic.append(our_error)

        for t, w_T_b_gt in zip(thetas, w_T_b_prismatic):
            a_T_b = np.require(gtsam.Pose3.Expmap(xi * t).matrix(), requirements='C')
            w_T_b_pred = w_T_b_prismatic[0] @ a_T_b
            b_T_b_pred = np.linalg.inv(w_T_b_gt) @ w_T_b_pred
            delta_rot = Rotation.from_matrix(b_T_b_pred[:3, :3]).magnitude()
            delta_lin = np.linalg.norm(b_T_b_pred[:3, 3])
            deltas_lin.append(delta_lin)
            deltas_rot.append(delta_rot)

        ### REVOLUTE -------------------
        gt_revolute = Articulation(w_T_offset[:3, 3],
                                   w_T_offset[:3, 2],
                                   'REVOLUTE')


        w_T_b_revolute = np.array([np.eye(4)] * 10)
        w_T_b_revolute[:, 0, 3] = 0.2
        for x, yaw in enumerate(np.linspace(0, np.pi * 0.5, len(w_T_b_revolute))):
            w_R_turn = np.eye(4)
            w_R_turn[:3, :3] = Rotation.from_euler('Z', yaw).as_matrix()
            w_T_b_revolute[x] = w_R_turn @ w_T_b_revolute[x]

        
        w_T_b_revolute = w_T_offset @ w_T_b_revolute        

        w_T_b_prior = max_removed_pose(w_T_b_revolute)
        a_T_b_prior = gtsam.Pose3.Logmap(gtsam.Pose3(np.linalg.inv(w_T_b_revolute[0]) @ w_T_b_prior))

        # print('REVOLUTE')
        xi, thetas = solve_articulation_from_poses(np.array([w_T_b_revolute[0]] * len(w_T_b_revolute)),
                                                   w_T_b_revolute,
                                                   prior_xi=a_T_b_prior)
        out_metric_revolute.append(compute_articulation_delta_twist(gt_revolute, w_T_b_revolute[0], xi))

        xi_P_origin = np.cross(xi[:3], xi[3:]) / (xi[:3]**2).sum()
        w_P_origin  = w_T_b_revolute[0, :3, :3] @ xi_P_origin + w_T_b_revolute[0, :3, 3]
        origin_est_error.append(np.linalg.norm(w_P_origin - w_T_offset[:3, 3]))

        for t, w_T_b_gt in zip(thetas, w_T_b_revolute):
            a_T_b = np.require(gtsam.Pose3.Expmap(xi * t).matrix(), requirements='C')
            w_T_b_pred = w_T_b_revolute[0] @ a_T_b
            b_T_b_pred = np.linalg.inv(w_T_b_gt) @ w_T_b_pred
            delta_rot = Rotation.from_matrix(b_T_b_pred[:3, :3]).magnitude()
            delta_lin = np.linalg.norm(b_T_b_pred[:3, 3])
            deltas_lin.append(delta_lin)
            deltas_rot.append(delta_rot)

    assert np.max(delta_lin) < 1e-4
    assert np.max(delta_rot) < 1e-4
    assert np.max(origin_est_error) < 1e-4

    # print(np.mean(deltas_lin), np.std(deltas_lin), np.median(deltas_lin), np.min(deltas_lin), np.max(deltas_lin))
    # print(np.mean(deltas_rot), np.std(deltas_rot), np.median(deltas_rot), np.min(deltas_rot), np.max(deltas_rot))
    # print(f'Error in estimating rotation origins: {np.mean(origin_est_error)}, {np.std(origin_est_error)}, {np.median(origin_est_error)}, {np.min(origin_est_error)}, {np.max(origin_est_error)}')

    # for f_agg in [np.mean, np.std, np.median, np.min, np.max]:
    #     print(f'{f_agg} prismatic: {f_agg(out_metric_prismatic, axis=0)}')
    #     print(f'{f_agg} revolute: {f_agg(out_metric_revolute, axis=0)}')
    