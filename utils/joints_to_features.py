import torch
import numpy as np
from common.skeleton import Skeleton
from utils.paramUtil import *
from common.quaternion import qrot_np, qinv_np, qmul_np, quaternion_to_cont6d_np, qbetween_np, qfix

def joints_to_features(joints, feet_thre=0.002, dataset='humanml'):
    """
    joints: numpy array of shape (frames, joints_num, 3)
    returns: RIC feature vector (frames-1, feature_dim)
    """

    if dataset == 'humanml':
        n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        kinematic_chain = t2m_kinematic_chain
        joints_num = 22
        fid_r, fid_l = [8, 11], [7, 10]
        face_joint_indx = [2, 1, 17, 16]
    elif dataset == 'kit':
        n_raw_offsets = torch.from_numpy(kit_raw_offsets)
        kinematic_chain = kit_kinematic_chain
        joints_num = 21
        fid_r, fid_l = [14, 15], [19, 20]
        face_joint_indx = [11, 16, 5, 8]
    else:
        raise ValueError('Unknown dataset')

    joints = torch.from_numpy(joints).float()

    # Step 1: Build Skeleton
    skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # Step 2: Uniform skeleton (optional, skipped if already processed)
    tgt_offsets = skel.get_offsets_joints(joints[0])

    # Step 3: Put on floor
    floor_height = joints[:, :, 1].min()
    joints[:, :, 1] -= floor_height

    # Step 4: Face Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = joints[0, r_hip] - joints[0, l_hip]
    across2 = joints[0, sdr_r] - joints[0, sdr_l]
    across = across1 + across2
    across = across / torch.norm(across)
    forward_init = torch.cross(torch.tensor([0.0, 1.0, 0.0]), across)
    forward_init = forward_init / torch.norm(forward_init)
    target = torch.tensor([0.0, 0.0, 1.0])
    root_quat_init = qbetween_np(forward_init.numpy(), target.numpy())
    root_quat_init = np.ones(joints.shape[:-1] + (4,)) * root_quat_init
    joints = qrot_np(root_quat_init, joints.numpy())

    global_positions = joints.copy()

    # Step 5: Foot contact detection
    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)

        return feet_l, feet_r

    feet_l, feet_r = foot_detect(joints, feet_thre)

    # Step 6: Extract features

    # Inverse kinematics to quaternions
    skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    quat_params = skel.inverse_kinematics_np(joints, face_joint_indx, smooth_forward=True)
    quat_params = qfix(quat_params)

    # Continuous 6D
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0]

    # Root linear velocities
    velocity = joints[1:, 0] - joints[:-1, 0]
    velocity = qrot_np(r_rot[1:], velocity)
    l_velocity = velocity[:, [0, 2]]

    # Root rotation velocities
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    r_velocity = np.arcsin(r_velocity[:, 2:3])

    # Root height
    root_y = joints[:-1, 0, 1:2]

    root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

    # Joint positions (RIC)
    joints_local = joints.copy()
    joints_local[..., 0] -= joints_local[:, 0:1, 0]
    joints_local[..., 2] -= joints_local[:, 0:1, 2]
    joints_local = qrot_np(np.repeat(r_rot[:, None], joints_local.shape[1], axis=1), joints_local)
    ric_data = joints_local[:, 1:].reshape(joints_local.shape[0], -1)

    # Joint rotations
    rot_data = cont_6d_params[:, 1:].reshape(cont_6d_params.shape[0], -1)

    # Joint velocities
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(local_vel.shape[0], -1)

    # Combine
    data = np.concatenate([root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1)

    return data
